"""Cache public Depth Pro 2-S features with geometry and disjoint sensor observations."""
import collections
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from light_encoder_study import yield_memory
from lightview42 import frames
from parallax42 import previous, read
from parallax42_data import FOCAL_BASELINE, REAL, ROOT, read_rgb
from run_stereo_students import gpu_slot


OUT = REAL.parents[1] / "pf42-trials"
SEED = 421327


class Reader:
    def __init__(self, device="cuda"):
        source = ROOT / "research-sources"
        os.environ["TORCH_HOME"] = str(source / "ml-depth-pro-2-cache")
        sys.path.insert(0, str(source / "ml-depth-pro-2/src"))
        from depth_pro_2 import create_model_and_transforms
        from depth_pro_2.network.encoder import autocast_for
        self.device = torch.device(device)
        self.net, self.transform = create_model_and_transforms("depthpro2-s", device=self.device)
        self.cast = autocast_for(self.device, self.net.precision)
        dim = self.net.encoder.blocks[-1].norm1.normalized_shape[0]
        matrix, _ = np.linalg.qr(np.random.default_rng(SEED).normal(size=(dim, 16)))
        self.project = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        self.checked = False

    @torch.inference_mode()
    def sample(self, rgb, y, x):
        if self.device.type == "cuda":
            yield_memory()
        h, w = rgb.shape[:2]
        image = self.transform(rgb).to(self.device)[None]
        padded = F.pad(image, (0, -w % 32, 0, -h % 32), mode="replicate")
        with self.cast():
            maps, grid = self.net.encoder.encode(padded)
        q = self.net.decode_frame(maps, grid)[0, 0, :h, :w].float().clamp_min(1e-5)
        focal = self.net.fov.focal_length_px(image).float()
        if not self.checked:
            official = self.net.infer(image[0])
            torch.testing.assert_close(q, official["inverse_depth"].clamp_min(1e-5), rtol=0, atol=2e-6)
            self.checked = True
        base = focal/(w*q)
        yy = torch.as_tensor(y, device=self.device)
        xx = torch.as_tensor(x, device=self.device)
        ids = yy.div(self.net.encoder.patch_size, rounding_mode="floor")*grid[1] + xx.div(
            self.net.encoder.patch_size, rounding_mode="floor")
        depth = base[yy, xx].cpu().numpy()
        feature = (maps[-1][0, ids].float() @ self.project).cpu().numpy()
        image_depth = base[::16, ::16].log()
        data = np.column_stack((feature, np.log(depth), x/w, y/h, rgb[y, x]/255.,
                                np.full(len(x), float((focal/w).log())),
                                np.full(len(x), float(image_depth.median()))))
        return depth, data.astype(np.float32)


def save(name, rows):
    data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    path = OUT / f"{name}.npz"
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp, **data)
    temp.replace(path)
    print(name, len(rows), len(data["base"]), flush=True)


def synthetic(reader):
    previous_rows = read(previous.OUT / "manifest.json")["synthetic"]
    candidates = collections.defaultdict(list)
    for i, row in enumerate(previous_rows):
        candidates[row[0]].append(i)
    rng = np.random.default_rng(SEED)
    rows = []
    for scene, indices in sorted(candidates.items()):
        for index in rng.permutation(indices):
            rgb, z, valid, _ = previous.read_training("s", int(index), previous_rows, None)
            yy, xx = np.nonzero(valid & np.isfinite(z) & (z >= .25) & (z <= 30))
            if len(yy) >= 64:
                chosen = rng.choice(len(yy), 64, replace=False)
                y, x = yy[chosen], xx[chosen]
                base, features = reader.sample(rgb, y, x)
                rows.append(dict(base=base, features=features, truth=z[y, x], group=np.repeat(scene, len(y))))
                break
        if len(rows) % 100 == 0:
            print("synthetic scenes", len(rows), flush=True)
    save("synthetic", rows)


def stereo_match(first, second):
    first, second = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (first, second))
    engine = cv2.StereoSGBM_create(minDisparity=-16, numDisparities=112, blockSize=5,
                                   P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=12,
                                   speckleWindowSize=80, speckleRange=2)
    a = engine.compute(first, second).astype(np.float32)/16
    b = engine.compute(second[:, ::-1].copy(), first[:, ::-1].copy())[:, ::-1].astype(np.float32)/16
    y, x = np.indices(a.shape)
    partner = np.rint(x-a).astype(np.int32)
    other = b[y, partner.clip(0, a.shape[1]-1)]
    keep = (a >= -16) & (other >= -16) & (abs(a-other) <= 1) & (partner >= 0) & (partner < a.shape[1])
    return np.where(keep, a, np.nan)


def real(reader):
    rows = collections.defaultdict(list)
    mapping = {p.name: re.sub(r"\d+$", "", p.name) for p in REAL.iterdir() if p.is_dir()}
    group_ids = {g: i for i, g in enumerate(sorted(set(mapping.values())))}
    exposures = [f"Illum{i}/Exp{e}" for i in (1, 2) for e in (1, 2)]
    for scene, group in sorted(mapping.items()):
        folder = REAL / scene
        split = {0: "held", 1: "cal"}.get(group_ids[group] % 5, "fit")
        dmin = float((folder / "dmin.txt").read_text())/3
        disparities = []
        for p in exposures:
            left, right = (read_rgb(folder / p / name) for name in ("view1.png", "view5.png"))
            if not disparities:
                rgb = left
            disparities.append(stereo_match(left, right))
        h, w = rgb.shape[:2]
        y, x = np.mgrid[8:h:16, 8:w:16].reshape(2, -1)
        q = np.stack(disparities)[:, y, x].T + dmin
        distance = FOCAL_BASELINE/np.maximum(q, .001)
        ok = (distance >= .25) & (distance <= 30)
        distance = np.where(ok, distance, np.nan)
        use = ok.sum(axis=1) >= 2
        y, x, distance = y[use], x[use], distance[use]
        base, features = reader.sample(rgb, y, x)
        truth = np.full(len(x), np.nan, dtype=np.float32)
        if split != "fit":
            gt = cv2.imdecode(np.fromfile(folder / "disp1.png", np.uint8), cv2.IMREAD_UNCHANGED)[y, x]
            truth = np.where(gt > 0, FOCAL_BASELINE/(gt.astype(np.float32)/3+dmin), np.nan)
        rows[split].append(dict(base=base, features=features, y=distance, truth=truth,
                                group=np.repeat(group, len(y)), scene=np.repeat(scene, len(y))))
    for split, data in rows.items():
        save(split, data)


def external(reader, name):
    rows = []
    for key, group, rgb, z, valid in frames(name):
        h, w = rgb.shape[:2]
        y, x = np.meshgrid(np.linspace(0, h-1, 16, dtype=int), np.linspace(0, w-1, 24, dtype=int), indexing="ij")
        keep = valid[y, x] & np.isfinite(z[y, x]) & (z[y, x] >= .25) & (z[y, x] <= 30)
        y, x = y[keep], x[keep]
        if len(x):
            base, features = reader.sample(rgb, y, x)
            rows.append(dict(base=base, features=features, truth=z[y, x],
                             group=np.repeat(group, len(y)), key=np.repeat(key, len(y))))
        if len(rows) % 200 == 0:
            print(name, len(rows), flush=True)
    save(name, rows)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    with gpu_slot():
        model = Reader()
        if not (OUT / "synthetic.npz").is_file():
            synthetic(model)
        if not all((OUT / f"{s}.npz").is_file() for s in ("cal", "fit", "held")):
            real(model)
        for dataset in ("hammer", "ibims", "diode", "diode_outdoor"):
            if not (OUT / f"{dataset}.npz").is_file():
                external(model, dataset)
