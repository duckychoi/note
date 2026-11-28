"""Score published monocular metric checkpoints on the historical iBims cohort."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from common import save
from encoder_phase import SOURCES, load_independent


def metric_scores(name, prediction, depth, valid):
    if prediction.shape != depth.shape or not valid.any():
        raise ValueError(f"Invalid depth output or mask for {name}")
    values = prediction[valid]
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"Invalid metric prediction for {name}")
    ratio = values / depth[valid]
    return [float(np.abs(ratio - 1).mean()), float((np.maximum(ratio, 1 / ratio) < 1.25).mean())]


def score_summary(rows):
    abs_rel, delta1 = np.mean(list(rows.values()), axis=0).tolist()
    return dict(images=len(rows), abs_rel=abs_rel, delta1=delta1, ibims_rows=rows)


def sha256(path):
    with path.open("rb") as weights:
        return hashlib.file_digest(weights, "sha256").hexdigest()


def model_input(rgb):
    import cv2

    # The authors' metric infer_image uses a 518 px lower bound and 14 px patches.
    height, width = rgb.shape[:2]
    scale = 518 / min(height, width)
    target = tuple(round(length * scale / 14) * 14 for length in (width, height))
    resized = cv2.resize(rgb / 255.0, target, interpolation=cv2.INTER_CUBIC)
    normalized = (resized - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw)[None].to("cuda")


def load_depth_anything(size):
    from transformers import DepthAnythingForDepthEstimation

    torch.set_num_threads(2)
    path = SOURCES / "encoder42_eval" / f"da2-metric-hypersim-{size}"
    model = DepthAnythingForDepthEstimation.from_pretrained(
        path, local_files_only=True, use_safetensors=True
    ).to("cuda").eval()
    if model.config.depth_estimation_type != "metric":
        raise ValueError("A metric checkpoint is required")
    return model, path


def predict(model, rgb, shape):
    output = model(pixel_values=model_input(rgb)).predicted_depth
    return F.interpolate(output[:, None], shape, mode="bilinear", align_corners=True)[0, 0].float().cpu().numpy()


@torch.inference_mode()
def run(size):
    model, path = load_depth_anything(size)
    rows = {}
    for name, rgb, _, depth, valid in load_independent(lambda image: image):
        rows[name] = metric_scores(name, predict(model, rgb, depth.shape), depth, valid)
    scores = score_summary(rows)
    model_id = f"depth-anything/Depth-Anything-V2-Metric-Indoor-{size.title()}-hf"
    save(f"da2_hypersim_{size}_ibims.json", dict(
        model=model_id, sha256=sha256(path / "model.safetensors"), calibration="none", raw_rgb_size=[320, 240],
        model_short_side=518, user_supplied_intrinsics=False,
        gpu=torch.cuda.get_device_name(), **scores,
    ))
    print(f"{model_id}: {len(rows)} images, AbsRel {scores['abs_rel']:.5f}, delta1 {scores['delta1']:.5f}", flush=True)


@torch.inference_mode()
def run_diode(size, archive_path, output):
    from depth_full_eval import load_sample, sample_names

    model, path = load_depth_anything(size)
    by_scene, rows = defaultdict(list), {}
    with ZipFile(archive_path) as archive:
        for name in sample_names(archive, indoor=True):
            rgb, depth, valid, _ = load_sample(archive, name, stride=1)
            scores = metric_scores(name, predict(model, rgb, depth.shape), depth, valid)
            rows[name] = scores
            by_scene[name.split("/")[3]].append(scores)
    abs_rel, delta1 = np.mean([np.mean(scores, axis=0) for scores in by_scene.values()], axis=0).tolist()
    result = dict(abs_rel=abs_rel, delta1=delta1, sha256=sha256(path/"model.safetensors"), rows=rows)
    Path(output).write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf8")
    print(f"DA V2 Hypersim {size}: {abs_rel:.5f} / {delta1:.5f}", flush=True)


@torch.inference_mode()
def run_apple(preset_name):
    sys.path.insert(0, str(SOURCES / "ml-depth-pro-2" / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint

    torch.set_num_threads(2)
    torch.hub.set_dir(str(SOURCES / "ml-depth-pro-2-cache" / "hub"))
    preset = PRESETS[preset_name]
    model, transform = create_model_and_transforms(preset, device="cuda")
    rgb_rows, intrinsics_rows = {}, {}
    for name, rgb, focal, depth, valid in load_independent(transform):
        output = model.infer(rgb)
        denominator = rgb.shape[-1] * output["inverse_depth"].clamp_min(1e-5)
        rgb_prediction = (output["focallength_px"] / denominator).cpu().numpy()
        known_prediction = (focal / denominator).cpu().numpy()
        rgb_rows[name] = metric_scores(name, rgb_prediction, depth, valid)
        intrinsics_rows[name] = metric_scores(name, known_prediction, depth, valid)
    scores = {"rgb_only": score_summary(rgb_rows), "known_intrinsics": score_summary(intrinsics_rows)}
    save(f"{preset_name}_ibims_reference.json", dict(
        model=preset_name, sha256=sha256(resolve_checkpoint(preset.checkpoint_uri)),
        fov_sha256=sha256(resolve_checkpoint(preset.fov_checkpoint_uri)),
        calibration="none", raw_rgb_size=[320, 240], inverse_depth_min=1e-5,
        gpu=torch.cuda.get_device_name(), **scores,
    ))
    for mode, values in scores.items():
        print(f"{preset_name} {mode}: AbsRel {values['abs_rel']:.5f}, delta1 {values['delta1']:.5f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("small", "large", "depthpro2-s", "depthpro2-h"))
    parser.add_argument("--diode-archive")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.diode_archive:
        if args.model.startswith("depthpro2-") or not args.output:
            parser.error("--diode-archive needs a Depth Anything V2 model and --output")
        run_diode(args.model, args.diode_archive, args.output)
    else:
        (run_apple if args.model.startswith("depthpro2-") else run)(args.model)
