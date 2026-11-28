"""Compare official Depth Pro 2 weights on the existing single-view real test."""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
from common import DEV, TEST, SEEDS, STORE, save
from summarize import load


def family_means(our_scores, apple_scores, subset, budget):
    our_rows = [row for row in our_scores if (row["subset"], row["budget"], row["arm"])
                == (subset, budget, "lines_order")]
    results = {}
    for name in ("whole", "jump", "order"):
        families = {}
        for scene in TEST:
            apple = apple_scores[scene][name]
            ours = [row[name] for row in our_rows if row["scene"] == scene]
            assert len(ours) == len(SEEDS) and all((q is None) == (apple is None) for q in ours)
            if apple is not None:
                families.setdefault(scene[:-1], []).append((float(np.mean(ours)), apple))
        means = {family: np.mean(pairs, axis=0).tolist() for family, pairs in sorted(families.items())}
        ours, apple = np.mean(list(means.values()), axis=0)
        results[name] = dict(ours=float(ours), apple=float(apple), by_family=means)
    return results


def infer(name, output):
    import torch
    sources = Path(__file__).resolve().parents[3] / "research-sources"
    sys.path.insert(0, str(sources / "ml-depth-pro-2" / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint

    torch.set_num_threads(2)
    preset = replace(PRESETS[name], fov_checkpoint_uri=None)
    model, transform = create_model_and_transforms(preset, device="cuda")
    predictions = {}
    with torch.inference_mode():
        for scene in DEV + TEST:
            with np.load(STORE / f"{scene}.npz") as view:
                rgb, fx = view["rgb"], float(view["fx"])
            depth = model.infer(transform(rgb), f_px=fx)["depth"]
            depth = depth.float().cpu().numpy()
            assert depth.shape == rgb.shape[:2] and np.isfinite(depth).all() and (depth > 0).all()
            predictions[scene] = depth
            print("Inferred", scene, depth.shape, flush=True)
    checkpoint = resolve_checkpoint(preset.checkpoint_uri)
    with checkpoint.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    np.savez_compressed(output, **predictions, checkpoint_sha256=digest, gpu=torch.cuda.get_device_name())


def score(name, output):
    from score import load_view, med, metrics

    real = {scene: load_view(scene) for scene in DEV + TEST}
    wide = {scene: load_view(scene, wide=True) for scene in TEST}
    with np.load(output) as data:
        predictions = {scene: data[scene] for scene in DEV + TEST}
        digest, gpu = str(data["checkpoint_sha256"]), str(data["gpu"])
    z = np.concatenate([real[scene]["z"] for scene in DEV])
    p = np.concatenate([predictions[scene][real[scene]["y"], real[scene]["x"]] for scene in DEV])
    assert z.shape == p.shape and len(z) > 0
    calibrated_scale = med(z / p, p)
    result = dict(preset=name, checkpoint_sha256=digest, gpu=gpu,
                  calibration_points=len(z), apple_calibrated_scale=calibrated_scale, comparisons={})
    ours = load("scores")
    previous = load("summary")
    for subset, views in (("strict", real), ("wide", wide)):
        for budget, scale in (("zero", 1.), ("few", calibrated_scale)):
            apple = {scene: metrics(predictions[scene].astype(float) * scale, views[scene]) for scene in TEST}
            both = family_means(ours, apple, subset, budget)
            key = f"{subset}_{budget}"
            reference = previous[key]["means"]["lines_order"]
            assert all(np.isclose(both[name]["ours"], reference[name]) for name in both)
            result["comparisons"][key] = dict(metrics=both, apple_views=apple)
            print(key, {metric: [round(scores[arm], 4) for arm in ("ours", "apple")]
                        for metric, scores in both.items()}, flush=True)
    save(name + "_comparison.json", result)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("infer", "score"))
    cli.add_argument("--preset", choices=("depthpro2-h", "depthpro2-s"), default="depthpro2-h")
    args = cli.parse_args()
    output = Path(__file__).parent / (args.preset + "_predictions.npz")
    (infer if args.stage == "infer" else score)(args.preset, output)


if __name__ == "__main__":
    main()
