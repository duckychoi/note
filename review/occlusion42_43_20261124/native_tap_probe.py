"""Official TAPNext++ on physical Middlebury 2005/06 rectified triples."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from repeat_capture_render import BASELINE, FOCAL, geometry, raster_correspondence, sample

ROOT = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval"
DATA = ROOT / "repeat-capture/physical"
OUT = ROOT / "native-surface-direction/tap-probe"


def family(name):
    return re.sub(r"\d+$", "", name)


def image(scene, light, view):
    with Image.open(scene / f"Illum{light}/Exp1/view{view}.png") as f:
        return np.asarray(f.convert("RGB"))[..., ::-1].copy()


def predict(model, scene):
    reference = image(scene, 1, 1)
    height, width = reference.shape[:2]
    ys, xs = np.meshgrid(np.arange(16, height - 16, 24), np.arange(16, width - 16, 24), indexing="ij")
    points = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    rows = {}
    for light in (1, 2):
        for view in (3, 5):
            _, _, state = model.track_frame(reference, points, autocast=False)
            target, visible, _ = model.track_frame(image(scene, light, view), state=state, autocast=False)
            rows[f"L1-L{light}/view{view}"] = (target, visible)
    return points, rows


def predict_sgbm(scene, points):
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                   P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=10,
                                   speckleWindowSize=0, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    anchor = cv2.cvtColor(image(scene, 1, 1), cv2.COLOR_BGR2GRAY)
    xs, ys = points.astype(int).T
    rows = {}
    for light in (1, 2):
        for view in (3, 5):
            target = cv2.cvtColor(image(scene, light, view), cv2.COLOR_BGR2GRAY)
            left = matcher.compute(anchor, target)[ys, xs].astype(np.float32)/16
            right = matcher.compute(np.fliplr(target), np.fliplr(anchor))[:, ::-1].astype(np.float32)/16
            tx = xs-left
            right_x = np.clip(np.rint(tx).astype(int), 0, target.shape[1]-1)
            valid = (left >= 0) & (tx >= 0) & (tx < target.shape[1]) & (right[ys, right_x] >= 0)
            valid &= abs(right[ys, right_x]-left) <= 1
            rows[f"L1-L{light}/view{view}"] = (np.column_stack((tx, ys)).astype(np.float32), valid)
    return rows


def score(scene, points, rows):
    x, disparity, dmin, _, _, _, source_good = geometry(scene)
    ys, xs = points.astype(int).T[::-1]
    scores, ground_truth = {}, {}
    for key, (prediction, confidence) in rows.items():
        fraction = .5 if key.endswith("view3") else 1.
        target_x = x - fraction * disparity
        if fraction == .5:
            inverse_map = raster_correspondence(x, disparity, fraction)
            good = source_good & (abs(sample(inverse_map, target_x) - x) < .6)
        else:
            good = source_good
        eligible = good[ys, xs]
        truth = np.column_stack((target_x[ys, xs], ys))
        valid = eligible & confidence & np.isfinite(prediction).all(axis=1)
        horizontal_error = truth[:, 0] - prediction[:, 0]
        epe = np.linalg.norm(prediction - truth, axis=1)
        predicted_shift = points[:, 0] - prediction[:, 0]
        inv = (predicted_shift / fraction + dmin) / (FOCAL * BASELINE)
        inv_truth = (disparity[ys, xs] + dmin) / (FOCAL * BASELINE)
        ground_truth[key] = dict(xy=truth, eligible=eligible, inverse_depth_per_m=inv_truth,
                                 disparity_1_to_5_px=disparity[ys, xs])
        scores[key] = dict(queries=len(points), eligible=int(eligible.sum()), visible=int(valid.sum()),
                           coverage=float(valid.sum()/max(1, eligible.sum())),
                           pck_3px=float((valid & (epe <= 3)).sum()/max(1, eligible.sum())),
                           epe_px=float(epe[valid].mean()) if valid.any() else None,
                           signed_disparity_error_px=float(horizontal_error[valid].mean()/fraction) if valid.any() else None,
                           sign_error=float((predicted_shift[valid] <= 0).mean()) if valid.any() else None,
                           inverse_depth_abs_error_per_m=float(abs(inv[valid]-inv_truth[valid]).mean()) if valid.any() else None)
    return scores, ground_truth


def aggregate(rows):
    result = {}
    for key in next(iter(rows.values())):
        result[key] = {}
        for metric in ("coverage", "pck_3px", "epe_px", "signed_disparity_error_px", "sign_error",
                       "inverse_depth_abs_error_per_m"):
            groups = defaultdict(list)
            for scene, values in rows.items():
                value = values[key][metric]
                if value is not None:
                    groups[family(scene)].append(value)
            result[key][metric] = float(np.mean([np.mean(values) for values in groups.values()])) if groups else None
    return result


def main(args):
    sys.path[:0] = [str(OUT / "tapnet-source"), str(OUT / "deps")]
    import torch
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    source = OUT / "tapnet-source"
    checkpoint = OUT / "tapnextpp_ckpt.pt"
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    torch.set_num_threads(2)
    model = TAPNextPP.from_checkpoint(checkpoint, device=args.device)
    held_out = sorted(p for p in DATA.iterdir() if p.is_dir() and
                      family(p.name) not in {"Art", "Aloe", "Midd", "Wood"})
    if len(held_out) != 21 or len({family(p.name) for p in held_out}) != 13:
        raise ValueError("Expected 21 held-out scenes from 13 families")
    rows, sgbm, paired, exposures, frames, raw_tracks = {}, {}, {}, {}, {}, {}
    for scene in held_out[:args.limit]:
        points, tracks = predict(model, scene)
        classic = predict_sgbm(scene, points)
        rows[scene.name], gt = score(scene, points, tracks)
        sgbm[scene.name], _ = score(scene, points, classic)
        paired[scene.name] = {}
        frames[scene.name] = dict(reference=f"{scene.name}/Illum1/Exp1/view1.png", target={})
        exposures[scene.name] = [float((scene / f"Illum{light}/Exp1/exposure_ms.txt").read_text())
                                 for light in (1, 2)]
        raw_tracks[f"{scene.name}/query_xy"] = points
        for key, (target, visible) in tracks.items():
            prefix = f"{scene.name}/{key}"
            light, view = key.split("/")
            frames[scene.name]["target"][key] = f"{scene.name}/Illum{light[-1]}/Exp1/{view}.png"
            for name, value in gt[key].items():
                raw_tracks[f"{prefix}/gt_{name}"] = value
            raw_tracks[f"{prefix}/tap_xy"] = target
            raw_tracks[f"{prefix}/tap_visible"] = visible
            classic_xy, classic_visible = classic[key]
            raw_tracks[f"{prefix}/sgbm_xy"] = classic_xy
            raw_tracks[f"{prefix}/sgbm_visible"] = classic_visible
            eligible = gt[key]["eligible"]
            common = eligible & visible & classic_visible & np.isfinite(target).all(axis=1)
            truth = gt[key]["xy"][common]
            paired[scene.name][key] = dict(common=int(common.sum()), eligible=int(eligible.sum()),
                    tap_epe_px=float(np.linalg.norm(target[common]-truth, axis=1).mean()) if common.any() else None,
                    sgbm_epe_px=float(np.linalg.norm(classic_xy[common]-truth, axis=1).mean()) if common.any() else None)
        print(scene.name, flush=True)
    equal_exposure = {name: values for name, values in rows.items() if exposures[name][0] == exposures[name][1]}
    report = dict(checkpoint_sha256=digest, device=args.device, size=256,
                  source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                  exposure_ms=exposures, frame_paths_relative_to_physical=frames,
                  rows=rows, family_average=aggregate(rows), sgbm_rows=sgbm, sgbm_family_average=aggregate(sgbm),
                  paired_rows=paired,
                  equal_exposure_scenes=list(equal_exposure),
                  equal_exposure_family_average=aggregate(equal_exposure) if equal_exposure else None)
    np.savez_compressed(args.output.with_suffix(".npz"), **raw_tracks)
    pending = args.output.with_suffix(".partial")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    pending.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=21)
    parser.add_argument("--output", type=Path, default=OUT / "result.json")
    main(parser.parse_args())
