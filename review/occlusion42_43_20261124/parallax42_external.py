"""Supplementary ETH3D stereo geometry audit and a native public S reference."""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

from light_encoder_score import summary
from light_encoder_study import CKPT, Model, sha, write, yield_memory
from parallax42 import DATASETS, Scale, native_mask, read, previous
from parallax42_data import EPS, GRID, OUT, ROOT, SEED, grid_sample, point_multiplier, read_rgb, robust_delta
from run_stereo_students import gpu_slot


def metric_calibration(path):
    values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    focal = float(values["cam0"].lstrip("[").split()[0])
    return focal * float(values["baseline"]) * .001, float(values["doffs"])


def stereo_points(left, right, fb, doffs, seed):
    left, right = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right))
    low = int(np.floor(fb / 30 - doffs))
    high = int(np.ceil(fb / .25 - doffs))
    width = 16 * int(np.ceil((high - low + 1)/16))
    engine = cv2.StereoSGBM_create(minDisparity=low, numDisparities=width, blockSize=5,
                                   P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=12,
                                   speckleWindowSize=80, speckleRange=2)
    a = engine.compute(left, right).astype(np.float64) / 16
    b = engine.compute(right[:, ::-1].copy(), left[:, ::-1].copy())[:, ::-1].astype(np.float64) / 16
    y, x = np.indices(a.shape)
    mate = np.rint(x-a).astype(np.int32)
    other = b[y, mate.clip(0, a.shape[1]-1)]
    q = a+doffs
    keep = (a >= low) & (other >= low) & (abs(a-other) <= 1) & (mate >= 0) & (mate < a.shape[1])
    keep &= (q > max(fb/30, EPS)) & (q < fb/.25)
    yy, xx = grid_sample(keep, seed)
    return yy, xx, q[yy, xx], float(keep.mean())


@torch.inference_mode()
def eth3d():
    cv2.setNumThreads(1)
    model = Model()
    rows = []
    for index, p in enumerate(sorted((OUT / "eth3d").glob("*/calib.txt"))):
        yield_memory()
        fb, doffs = metric_calibration(p)
        folder = p.parent
        first, second = [read_rgb(folder/f"im{i}.png") for i in (0, 1)]
        y, x, q, area = stereo_points(first, second, fb, doffs, SEED+index)
        pred, _ = Scale.metric(model, first)
        pred = pred.double().cpu().numpy()
        d = pred[y, x]
        lo, hi = d*(q-EPS)/fb, d*(q+EPS)/fb
        bounds = [float(robust_delta(k, lo, hi).mean()) for k in GRID] if len(x) else [0.]
        robust = float(GRID[np.argmax(bounds)])
        point = point_multiplier(d, fb/q) if len(x) else 1.
        # Reference disparity is read only after the stereo teacher has proposed a scale.
        gt = cv2.imdecode(np.fromfile(folder/"disp0GT.pfm", np.uint8), cv2.IMREAD_UNCHANGED)
        if gt is None or gt.shape != pred.shape:
            raise ValueError(folder)
        t = pred*(gt.astype(np.float64)+doffs)/fb
        valid = np.isfinite(gt) & (gt+doffs >= fb/30) & (gt+doffs <= fb/.25)
        full = t[valid]
        use = valid[y, x]
        finite = t[y[use], x[use]]
        inside = int((abs(gt[y[use], x[use]]+doffs-q[use]) <= EPS).sum())
        def delta(v, k):
            return float((abs(v-1)-abs(k*v-1)).mean()) if len(v) else None
        rows.append(dict(key=folder.name, group=folder.name.rsplit("_", 1)[0], n=len(x), area=area,
                         valid=int(use.sum()), inside=inside, all_inside=bool(len(x) and len(finite) == len(x)
                                                                             and inside == len(x)),
                         bound=max(bounds), robust=robust, point=point,
                         full_oracle=point_multiplier(full, np.ones(len(full))),
                         baseline=float(abs(full-1).mean()), full_robust=delta(full, robust),
                         support_robust=delta(finite, robust), full_point=delta(full, point),
                         support_point=delta(finite, point)))
    write(OUT / "eth3d-audit.json", dict(checkpoint=sha(CKPT), rows=rows))
    print("parallax ETH3D", len(rows), flush=True)


@torch.inference_mode()
def official():
    src = ROOT / "research-sources"
    os.environ["TORCH_HOME"] = str(src / "ml-depth-pro-2-cache")
    sys.path.insert(0, str(src / "ml-depth-pro-2/src"))
    from depth_pro_2 import create_model_and_transforms
    net, transform = create_model_and_transforms("depthpro2-s", device=torch.device("cuda"))
    for dataset in DATASETS:
        output = OUT / f"official-{dataset}.json"
        if output.is_file():
            continue
        rows = []
        ref = read(previous.OUT / f"dense-{dataset}.json")["rows"]
        for (name, group, rgb, z, valid), old in zip(previous.frames(dataset), ref, strict=True):
            if (name, group) != (old["key"], old["group"]):
                raise ValueError("Baseline frame key changed")
            yield_memory()
            image = transform(rgb).cuda()
            estimate = net.infer(image)
            q = estimate["inverse_depth"].clamp_min(1e-5)
            depth = estimate["focallength_px"] / (q * q.shape[-1])
            truth, keep = native_mask(z, valid)
            scores = dict(public_s=float((depth[keep]/truth[keep]-1).abs().mean()),
                          frozen=old["scores"]["frozen"])
            if not np.isfinite(list(scores.values())).all():
                raise ValueError(name)
            rows.append(dict(key=name, group=group, scores=scores))
        write(output, dict(**summary(rows), rows=rows))
        print("parallax public S", dataset, summary(rows)["means"], flush=True)


@torch.inference_mode()
def resolution_probe():
    # Same RGB and fixed weights. This is a sensitivity diagnostic, not a new test score.
    model = Scale(False)
    saved = torch.load(OUT / "student.pt", map_location="cuda", weights_only=True)
    arm = "stereo_robust"
    model.arms[arm].load_state_dict(saved["arms"][arm])
    model.heads[arm].load_state_dict(saved["heads"][arm])
    rows = []
    for dataset in DATASETS:
        compared = read(OUT / f"{dataset}.json")["rows"]
        seen = set()
        for (key, group, rgb, _, _), original in zip(previous.frames(dataset), compared, strict=True):
            if group in seen:
                continue
            if (key, group) != (original["key"], original["group"]):
                raise ValueError("Resolution probe keys differ")
            seen.add(group)
            yield_memory()
            h, w = rgb.shape[:2]
            small = cv2.resize(rgb, (450, round(450*h/w)), interpolation=cv2.INTER_AREA)
            _, features = model.input(small)
            rows.append(dict(dataset=dataset, key=key, group=group, width=w,
                             native=original["scales"][arm], resized=float(model.scale(features, arm).exp()),
                             oracle=original["oracle"]))
    write(OUT / "resolution-probe.json", dict(rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("eth3d", "official", "resolution", "all"))
    stage = parser.parse_args().stage
    with gpu_slot():
        if stage in ("all", "eth3d"):
            eth3d()
        if stage in ("all", "official"):
            official()
        if stage == "resolution":
            resolution_probe()
