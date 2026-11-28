"""Compare auxiliary log-depth targets on held-out Hypersim scenes with a frozen S."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from depth_full_data import Hypersim
from depth_full_train import ErrorCovariance, GPU, SOURCES, paired_inference, paired_loss
from encoder_expand_data import selection_for


def calibration_pairs(seed):
    # A fixed prefix was available before download completion. Choose by scene ID
    # and seed, independent of prediction errors and current file availability.
    train = selection_for("train_full")
    rng = np.random.default_rng(seed)
    pairs = []
    for scene in rng.permutation(sorted(train)[:128])[:48].tolist():
        frames = defaultdict(list)
        for camera, index in train[scene]:
            frames[camera].append(index)
        candidates = [(scene, camera, start, stop) for camera, indices in frames.items()
                      for start, stop in zip(indices, indices[3:])]
        pairs.append(candidates[rng.integers(len(candidates))])
    return pairs


@torch.inference_mode()
def run(output, seed):
    sys.path.insert(0, str(SOURCES / "ml-depth-pro-2" / "src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint

    torch.set_num_threads(2)
    torch.hub.set_dir(str(SOURCES / "ml-depth-pro-2-cache" / "hub"))
    preset = replace(PRESETS["depthpro2-s"], fov_checkpoint_uri=None)
    model, transform = create_model_and_transforms(preset, device=GPU)
    model.eval()
    with resolve_checkpoint(preset.checkpoint_uri).open("rb") as f:
        weights = hashlib.file_digest(f, "sha256").hexdigest()
    data = Hypersim("validation_24", 2, pair=True)
    train_rows = calibration_pairs(seed)
    dev_by_scene = defaultdict(list)
    for row in data.rows:
        dev_by_scene[row[0]].append(row)
    rng = np.random.default_rng(seed)
    dev_rows = [rows[int(rng.integers(len(rows)))] for rows in dev_by_scene.values()]
    if {r[0] for r in train_rows} & set(dev_by_scene):
        raise ValueError("Calibration and held-out scenes overlap")

    covariance = ErrorCovariance()
    def examine(row, gamma, held_out):
        a, b, pose = data.read_pair(row)
        if pose is None:
            return dict(pair=row, eligible=0, invalid_pair_pose=True)
        first, second = paired_inference(a, b, model, transform, gamma)
        _, count, stats = paired_loss(first, second, torch.as_tensor(pose, device=GPU), "kf",
                                      covariance, diagnostics=held_out, held_out=held_out)
        return dict(pair=row, eligible=count, **stats)

    calibration = [examine(row, np.exp(rng.uniform(-.45, .6, 2)), False) for row in train_rows]
    before = covariance.moments.clone()
    before_counts = covariance.count.clone()
    scores, records = {}, {}
    for name, gamma in (("native", (1., 1.)), ("exposure_stress", (.8, 1.4))):
        result = [examine(row, gamma, True) for row in dev_rows]
        valid = [item["mean_squared_log_error"] for item in result if item["eligible"]]
        if not valid:
            raise ValueError("No visible held-out correspondences")
        scores[name] = {key: float(np.mean([item[key] for item in valid])) for key in valid[0]}
        scores[name]["covered_scenes"] = len(valid)
        scores[name]["gain_vs_fixed_percent"] = 100*(1-scores[name]["adaptive"]/scores[name]["fixed"])
        records[name] = result
    if not torch.equal(before, covariance.moments) or not torch.equal(before_counts, covariance.count):
        raise RuntimeError("Held-out scores modified calibration statistics")
    report = dict(seed=seed, weight_sha256=weights, calibration=calibration, dev_scenes=len(dev_rows),
                  calibration_counts=before_counts.tolist(), calibration_moments=before.tolist(),
                  scene_mean_squared_log_error=scores, rows=records)
    pending = output.with_suffix(".partial")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    pending.replace(output)
    print(json.dumps(scores, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4243)
    args = parser.parse_args()
    run(args.output, args.seed)
