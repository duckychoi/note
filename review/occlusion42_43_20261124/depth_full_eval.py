"""Paired metric evaluation without target-based scale or shift alignment."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from depth_full_data import make_sample
from depth_full_train import GPU, SOURCES


METRIC_MODES = ("rgb", "camera")
TRAINED_ARMS = ("d2", "fixed", "kf", "real")


def sample_names(archive, indoor):
    candidates = sorted(name.removesuffix("/meta.json") for name in archive.namelist()
                        if name.endswith("/meta.json"))
    if indoor:
        candidates = [name for name in candidates if re.search(r"(?i)(?:^|[/_-])indoors?(?:[/_-]|$)", name)]
        if len(candidates) != 325:
            raise ValueError(f"DIODE indoor frame count is {len(candidates)} instead of 325")
    if not candidates:
        raise ValueError("Evaluation archive has no samples")
    return candidates


def load_sample(archive, name, stride):
    with Image.open(BytesIO(archive.read(f"{name}/image.jpg"))) as file:
        rgb = np.array(file.convert("RGB"))
    with Image.open(BytesIO(archive.read(f"{name}/depth.png"))) as file:
        encoded = np.asarray(file)
        fraction = (encoded.astype(np.float32)-1)/65533
        meters = float(file.info["near"])**(1-fraction)*float(file.info["far"])**fraction
        meters *= float(file.info.get("unit", 1.))
    meta = json.loads(archive.read(f"{name}/meta.json"))
    k = np.array(meta["intrinsics"], dtype=np.float64)
    k[0] *= rgb.shape[1]
    k[1] *= rgb.shape[0]
    valid = (encoded > 0) & (encoded < 65535)
    return make_sample(rgb, meters, k, stride, valid=valid)


def depth_scores(prediction, depth):
    ratio = prediction / depth
    return [float((ratio-1).abs().mean()), float((torch.maximum(ratio, ratio.reciprocal())<1.25).float().mean())]


def metric_denominator(inverse_depth, mask, width):
    values=inverse_depth.float()[mask]
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite normalized inverse depth on a valid target pixel")
    return values.clamp_min(1e-5)*width, int((values<1e-5).sum())


def scale_diagnostics(known_prediction, depth, predicted_focal, known_focal):
    log_ratio = (known_prediction / depth).log()
    focal_error = float((predicted_focal / known_focal).log())
    known_bias = float(log_ratio.mean())
    return dict(focal_log_ratio=focal_error, focal_absolute_log_error=abs(focal_error),
                known_focal_log_depth_bias=known_bias, rgb_log_depth_bias=known_bias+focal_error,
                centered_log_depth_rmse=float((log_ratio-known_bias).square().mean().sqrt()))


@torch.inference_mode()
def evaluate(model, transform, archive, names, stride):
    model.eval()
    results = {}
    for name in names:
        rgb, z, mask, k = load_sample(archive, name, stride)
        depth = torch.as_tensor(z[mask], device=GPU)
        if depth.numel() == 0:
            raise ValueError(f"No valid metric depth: {name}")
        output = model.infer(transform(rgb))
        denominator,floored=metric_denominator(output["inverse_depth"],torch.as_tensor(mask,device=GPU),rgb.shape[1])
        focal, known_focal = output["focallength_px"], float(k[0, 0])
        known_prediction = known_focal/denominator
        results[name] = dict(rgb=depth_scores(focal/denominator, depth), camera=depth_scores(known_prediction, depth),
                             floored_inverse_pixels=floored,valid_pixels=depth.numel(),
                             scale_diagnostics=scale_diagnostics(known_prediction, depth, focal, known_focal))
    return results


def compare(rows, indoor):
    ordered_names = sorted(next(iter(rows.values())))
    scene_ids = {}
    for index, name in enumerate(ordered_names):
        scene = re.search(r"(?i)scene[_-]?\d+", name) if indoor else None
        if indoor and scene is None:
            raise ValueError(f"Missing DIODE scene identifier: {name}")
        scene_ids.setdefault(scene.group() if scene else name, []).append(index)
    metrics = {}
    scale_summary = {}
    for label, samples in rows.items():
        if set(samples) != set(ordered_names):
            raise ValueError(f"Mismatched evaluation images for {label}")
        array = np.array([[samples[name][mode] for mode in METRIC_MODES] for name in ordered_names])
        metrics[label] = np.array([array[index].mean(axis=0) for index in scene_ids.values()])
        if all("scale_diagnostics" in sample for sample in samples.values()):
            names = tuple(samples[ordered_names[0]]["scale_diagnostics"])
            values = np.array([[samples[frame]["scale_diagnostics"][metric] for metric in names]
                               for frame in ordered_names])
            macro = np.mean([values[index].mean(axis=0) for index in scene_ids.values()], axis=0)
            scale_summary[label] = dict(zip(names, macro.tolist()))
    summary = {label: array.mean(axis=0).tolist() for label, array in metrics.items()}
    comparison = {}
    if "kf" in metrics:
        boot = np.random.default_rng(4243).integers(len(scene_ids), size=(20000, len(scene_ids)))
        for label, base in metrics.items():
            if label == "kf":
                continue
            delta = 100*(1 - metrics["kf"][boot, :, 0].mean(axis=1)/base[boot, :, 0].mean(axis=1))
            gap = 100*(metrics["kf"][:, :, 0]-base[:, :, 0])[boot].mean(axis=1)
            comparison[label] = dict(relative_reduction_percent=(100*(1-metrics["kf"][:, :, 0].mean(axis=0)/
                                                                       base[:, :, 0].mean(axis=0))).tolist(),
                                     ci95=np.quantile(delta, [.025, .975], axis=0).T.tolist(),
                                     absrel_gap_percentage_points_ci95=np.quantile(gap, [.025, .975], axis=0).T.tolist())
    return dict(scene_ids=list(scene_ids), scores=summary, compared_with_kf=comparison,
                scale_diagnostics_scene_mean=scale_summary)


def full_run_evidence(run, arms=TRAINED_ARMS):
    """Recover the finalized development checkpoint selection without reading test scores."""
    selected = {}
    for arm in arms:
        with (run / arm / "training.jsonl").open(encoding="utf8") as log:
            entries = [json.loads(line) for line in log if line.strip()]
        metadata = next(entry for entry in entries if "config" in entry)
        config = metadata["config"]
        if config["smoke"]:
            raise ValueError(f"A short execution cannot be used for the full result: {arm}")
        rows = metadata["training_rows"]
        exposure_per_step = 1 if arm == "real" else 2
        scheduled = min(config["steps"] or rows*config["epochs"], rows*config["epochs"])
        development = [entry for entry in entries if "development_absrel" in entry]
        if max(entry["step"] for entry in development) != scheduled:
            raise ValueError(f"Missing final development evaluation: {arm}")
        best = min(development, key=lambda entry: entry["development_absrel"])
        selected[arm] = dict(config=config, selection_digest=metadata["selection_digest"], rows=rows,
                             steps=scheduled, exposures=scheduled*exposure_per_step,
                             development_checkpoint=best)
    matched = ("seed", "stride", "epochs", "steps", "lr", "val_every")
    for arm in (arm for arm in arms if arm not in ("d2", "real")):
        reference = selected["d2"]
        current = selected[arm]
        if arm in ("fixed", "kf") and current["config"].get("fusion_objective") != "half_squared_log_v1":
            raise ValueError(f"Unverified gain gradient in the training objective for {arm}")
        if (reference["selection_digest"] != current["selection_digest"] or reference["rows"] != current["rows"] or
                any(reference["config"][key] != current["config"][key] for key in matched)):
            raise ValueError(f"Unmatched synthetic training runs: d2 and {arm}")
    return selected


def research_decisions(report, evidence):
    candidates = [arm for arm in ("d2", "fixed", "silog") if arm in evidence]
    reference = min(candidates, key=lambda arm: evidence[arm]["development_checkpoint"]["development_absrel"])
    scores = report["scores"]
    comparisons = report["compared_with_kf"]
    decisions = {}
    for index, mode in enumerate(METRIC_MODES):
        reduction = comparisons[reference]["relative_reduction_percent"][index]
        interval = comparisons[reference]["ci95"][index]
        gap = 100*(scores["kf"][index][0] - scores["real"][index][0])
        gap_interval = comparisons["real"]["absrel_gap_percentage_points_ci95"][index]
        decisions[mode] = dict(synthetic_reference=reference, absrel_reduction_percent=reduction,
                               absrel_reduction_ci95=interval, meets_three_percent_point_estimate=reduction >= 3.,
                               lower_ci_reaches_three_percent=interval[0] >= 3.,
                               interval_includes_three_percent=interval[0] <= 3. <= interval[1],
                               kf_minus_real_absrel_percentage_points=gap, matches_or_beats_real_point_estimate=gap <= 0.,
                               kf_minus_real_absrel_percentage_points_ci95=gap_interval,
                               real_gap_upper_nonpositive=gap_interval[1] <= 0.,
                               reduction_vs_real_ci95=comparisons["real"]["ci95"][index])
    return decisions


def main(args):
    sys.path.insert(0, str(SOURCES / "ml-depth-pro-2" / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint

    torch.set_num_threads(2)
    torch.hub.set_dir(str(SOURCES / "ml-depth-pro-2-cache" / "hub"))
    run = Path(args.runs)
    output = Path(args.output)
    trained = [arm for arm in args.arms if arm != "frozen"]
    ready = set(TRAINED_ARMS).issubset(trained) and all((run / arm / "completed").is_file() for arm in trained)
    evidence = full_run_evidence(run, trained) if ready else None
    rows, digests = {}, {}
    model, transform = create_model_and_transforms("depthpro2-s", device=GPU)
    with resolve_checkpoint(PRESETS["depthpro2-s"].checkpoint_uri).open("rb") as stream:
        digests["public_initialization"] = hashlib.file_digest(stream, "sha256").hexdigest()
    with resolve_checkpoint(PRESETS["depthpro2-s"].fov_checkpoint_uri).open("rb") as stream:
        digests["focal_estimator"] = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(args.archive) as archive:
        names = sample_names(archive, args.indoor)
        for label in args.arms:
            checkpoint = run / label / "best.pt"
            if label != "frozen":
                with checkpoint.open("rb") as stream:
                    digests[label] = hashlib.file_digest(stream, "sha256").hexdigest()
                # Training stores the shared depth architecture, preserving the inference-only focal estimator.
                mismatch = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False)
                if mismatch.unexpected_keys or any(not key.startswith("fov.") for key in mismatch.missing_keys):
                    raise ValueError(f"Unexpected checkpoint keys: {mismatch}")
            elif rows:
                raise ValueError("Evaluate the frozen baseline before loading adapted weights")
            rows[label] = evaluate(model, transform, archive, names, args.stride)
            print(label, np.mean([row["rgb"] for row in rows[label].values()], axis=0).tolist(), flush=True)
    with Path(args.archive).open("rb") as stream:
        archive_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(archive=str(args.archive), archive_sha256=archive_digest, input_stride=args.stride, frames=len(names), weight_sha256=digests,
                  metric_order=["AbsRel", "delta1"], mode_order=list(METRIC_MODES),
                  rows=rows, **compare(rows, args.indoor))
    if evidence is not None:
        report["training_evidence"] = evidence
        report["research_decisions"] = research_decisions(report, evidence)
        report["claim_scope"] = dict(stage="exploratory_public_S_adaptation", confirmatory=False,
                                     independent_evaluation_groups=len(report["scene_ids"]),
                                     primary_mode="rgb", development_selection_mode="known_focal_stride2_image_mean"
                                     if all(e["config"]["stride"] == 2 for e in evidence.values())
                                     else "known_focal_at_training_stride_image_mean")
        print(json.dumps(report["research_decisions"], ensure_ascii=False, allow_nan=False), flush=True)
    pending = output.with_suffix(output.suffix+".partial")
    pending.write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf8")
    pending.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="+", choices=("frozen", "d2", "fixed", "kf", "real", "silog"),
                        default=["frozen", "d2", "fixed", "kf", "real"])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--indoor", action="store_true")
    from run_stereo_students import gpu_slot
    with gpu_slot():
        main(parser.parse_args())
