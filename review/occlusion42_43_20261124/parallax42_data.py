"""Read only RGB and metric camera calibration to generate stereo observations."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "research-sources/encoder42_eval"
REAL = SRC / "repeat-capture/physical"
OUT = SRC / "parallax42-third"
SOURCE_SCALE = 3
FOCAL_BASELINE = (3740.0 / SOURCE_SCALE) * .160
EPS = 2.0
LIMIT = .20
SEED = 421130
GRID = np.exp([0., -.02, .02, -.05, .05, -.10, .10, -LIMIT, LIMIT])


def read_rgb(path):
    # imdecode handles Unicode parent directories on Windows.
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()


def capture():
    folders = sorted(REAL.iterdir())
    families = {p.name: re.sub(r"\d+$", "", p.name) for p in folders}
    held = set(sorted(set(families.values()))[::4])
    for scene in folders:
        dmin = float((scene / "dmin.txt").read_text()) / SOURCE_SCALE
        for a in sorted(scene.glob("Illum*/Exp*/view1.png")):
            b = a.with_name("view5.png")
            if b.is_file():
                for ref, mate in ((a, b), (b, a)):
                    yield (str(ref.relative_to(REAL).as_posix()), str(mate.relative_to(REAL).as_posix()),
                           families[scene.name], families[scene.name] in held, dmin)


def grid_sample(valid, seed):
    rng = np.random.default_rng(seed)
    h, w = valid.shape
    yy, xx = np.linspace(0, h, 5, dtype=int), np.linspace(0, w, 5, dtype=int)
    points = []
    for y0, y1 in zip(yy[:-1], yy[1:]):
        for x0, x1 in zip(xx[:-1], xx[1:]):
            ys, xs = np.nonzero(valid[y0:y1, x0:x1])
            if len(ys):
                keep = rng.choice(len(ys), min(2, len(ys)), replace=False)
                points.extend(zip(y0 + ys[keep], x0 + xs[keep]))
    return np.asarray(points, dtype=np.int32).reshape(-1, 2).T


def stereo():
    cv2.setNumThreads(1)
    matcher = cv2.StereoSGBM_create(minDisparity=-16, numDisparities=112, blockSize=5,
                                   P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=12,
                                   speckleWindowSize=80, speckleRange=2)
    data = defaultdict(list)
    for j, (ref, mate, family, held, dmin) in enumerate(capture()):
        rev = ref.endswith("view5.png")
        if not rev:
            first, second = [cv2.cvtColor(read_rgb(REAL / p), cv2.COLOR_RGB2GRAY) for p in (ref, mate)]
            left = matcher.compute(first, second).astype(np.float32) / 16
            right = matcher.compute(second[:, ::-1].copy(), first[:, ::-1].copy())[:, ::-1].astype(np.float32) / 16
        primary, opposite = (right, left) if rev else (left, right)
        iy, ix = np.indices(primary.shape)
        match = np.rint(ix + primary if rev else ix - primary).astype(np.int32)
        keep = (primary >= -16) & (match >= 0) & (match < primary.shape[1])
        back = opposite[iy, match.clip(0, primary.shape[1]-1)]
        q = primary + dmin
        keep &= (back >= -16) & (abs(back-primary) <= 1) & (q > EPS)
        keep &= (q > FOCAL_BASELINE / 30) & (q < FOCAL_BASELINE / .25)
        seed = int.from_bytes(hashlib.sha256(ref.encode()).digest()[:8], "big") ^ SEED
        y, x = grid_sample(keep, seed)
        for key, value in dict(ref=ref, family=family, held=held, dmin=dmin, area=keep.mean(),
                               n=len(y), y=np.pad(y, (0, 32-len(y)), constant_values=-1),
                               x=np.pad(x, (0, 32-len(x)), constant_values=-1),
                               disparity=np.pad(primary[y, x], (0, 32-len(y)))).items():
            data[key].append(value)
        if (j + 1) % 32 == 0:
            print("captured", j + 1, flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "stereo.npz", **data)


def point_multiplier(pred, truth):
    ratio = truth / pred
    order = np.argsort(ratio)
    weights = 1 / ratio[order]
    middle = np.searchsorted(weights.cumsum(), .5 * weights.sum())
    return float(ratio[order[middle]].clip(np.exp(-LIMIT), np.exp(LIMIT)))


def robust_delta(k, lo, hi):
    def delta(t):
        return abs(t-1) - abs(k*t-1)
    return np.minimum.reduce([delta(lo), delta(hi), delta(np.clip(1., lo, hi)),
                              delta(np.clip(1./k, lo, hi))])


def robust_multiplier(pred, q):
    lo = pred * (q-EPS) / FOCAL_BASELINE
    hi = pred * (q+EPS) / FOCAL_BASELINE
    values = np.array([robust_delta(k, lo, hi).mean() for k in GRID])
    ix = np.argmax(values)
    return float(GRID[ix]), float(values[ix])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stereo",))
    parser.parse_args()
    stereo()
