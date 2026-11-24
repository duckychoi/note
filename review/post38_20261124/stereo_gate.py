"""Sparse physical depth queries on Middlebury trainingQ. Run from any directory."""
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[2] / "research-sources/post38_20261124/middlebury"
METHODS = ("uniform", "texture", "consistency")
SCALES = (0.8, 1.0, 1.2)


def read_pfm(data):
    stream = io.BytesIO(data)
    assert stream.readline().strip() == b"Pf"
    width, height = map(int, stream.readline().split())
    scale = float(stream.readline())
    pixels = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
    # Middlebury SDK reads the stored disparities directly in pixel units.
    # The official Eval3 PFM specification ignores the magnitude of this field.
    return np.flipud(pixels.reshape(height, width))


def match(first, second, disparities):
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=disparities,
                                    blockSize=5, P1=200, P2=800, disp12MaxDiff=1,
                                    uniquenessRatio=8, speckleWindowSize=50,
                                    speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM)
    return matcher.compute(first, second).astype(float) / 16


def absrel_calculator(ratios):
    sorted_ratios = np.sort(ratios.astype(float))
    prefix = np.r_[0, np.cumsum(sorted_ratios)]
    count = len(sorted_ratios)

    def score(multiplier):
        pivot = np.searchsorted(sorted_ratios, 1 / multiplier)
        return (2 * pivot - count + multiplier * (prefix[-1] - 2 * prefix[pivot])) / count

    return score


def evaluate(left, right, gt_disp, calib, overlap, scene_number):
    height, width = left.shape
    y, x = np.indices(left.shape)
    get = lambda key: float(re.search(r"^" + key + r"=([^\r\n]+)", calib, re.M)[1])
    numerator = float(re.search(r"cam0=\[([\d.]+)", calib)[1]) * get("baseline")
    offset = get("doffs")
    nd = int(np.ceil(get("ndisp") / 16) * 16)
    start, end = (0, width) if overlap == "full" else (width // 4, 3 * width // 4)
    right = right.copy()
    right[:, :start] = right[:, end:] = 0
    disparity = match(left, right, nd)
    reverse = match(np.ascontiguousarray(right[:, ::-1]),
                    np.ascontiguousarray(left[:, ::-1]), nd)[:, ::-1]
    projection = np.rint(x - disparity).astype(int)
    support = (disparity >= 0) & (projection >= start) & (projection < end) & (disparity + offset > 0)
    rev_at_match = reverse[y, np.clip(projection, 0, width - 1)]
    disagreement = np.where(rev_at_match >= 0, np.abs(disparity - rev_at_match), np.inf)
    texture = np.hypot(cv2.Sobel(left, cv2.CV_32F, 1, 0), cv2.Sobel(left, cv2.CV_32F, 0, 1))
    lattice = (x % 4 == 0) & (y % 4 == 0)
    truth = np.isfinite(gt_disp) & (gt_disp >= 0) & (gt_disp + offset > 0)
    test = ~lattice & truth
    rated = test & support
    query = lattice & support
    depths = np.divide(numerator, disparity + offset, out=np.zeros_like(disparity), where=support)
    gt = np.divide(numerator, gt_disp.astype(float) + offset, out=np.zeros_like(disparity), where=truth)
    score = absrel_calculator(depths[rated] / gt[rated])
    assert abs(score(1.2) - np.mean(np.abs(1.2 * depths[rated] / gt[rated] - 1))) < 1e-8
    scores, valid_rates = {}, {}
    for scale in SCALES:
        scores[f"0/{scale}"] = float(score(scale))
    for budget in (1, 4):
        regions = [(query if budget == 1 else query & ((x >= width // 2) == (k % 2)) &
                    ((y >= height // 2) == (k // 2))) for k in range(budget)]
        for method in METHODS:
            pools = []
            for region in regions:
                pool = np.flatnonzero(region)
                if method != "uniform":
                    ordering = (np.argsort(-texture.flat[pool], kind="stable") if method == "texture" else
                                np.lexsort((-texture.flat[pool], disagreement.flat[pool])))
                    pool = pool[ordering[:max(1, int(np.ceil(len(pool) * 0.1)))]]
                pools.append(pool)
            samples = {scale: [] for scale in SCALES}
            hits = 0
            for seed in range(12):
                rng = np.random.default_rng([61124, scene_number, budget, seed])
                log_ratios = []
                for pool in pools:
                    pick = rng.random()
                    noise = rng.normal(0, .02)
                    if len(pool):
                        point = pool[int(pick * len(pool))]
                        if truth.flat[point]:
                            log_ratios.append(np.log(gt.flat[point] / depths.flat[point]) + noise)
                hits += len(log_ratios)
                for scale in SCALES:
                    factor = np.exp(np.median(log_ratios)) if log_ratios else scale
                    samples[scale].append(float(score(factor)))
            valid_rates[f"{method}/{budget}"] = hits / (12 * budget)
            for scale in SCALES:
                scores[f"{method}/{budget}/{scale}"] = float(np.mean(samples[scale]))
    return dict(coverage=float(rated.sum() / test.sum()), score=scores, valid_query=valid_rates)


def main():
    cv2.setNumThreads(1)
    inputs = sorted(DATA.glob("*.zip"))
    document = {"sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "opencv": cv2.__version__, "numpy": np.__version__, "scene": {}, "summary": {}}
    with ZipFile(DATA / "MiddEval3-data-Q.zip") as photos, ZipFile(DATA / "MiddEval3-GT0-Q.zip") as labels:
        scenes = sorted(name.rsplit("/", 1)[0] for name in labels.namelist() if name.endswith("/disp0GT.pfm") and "/trainingQ/" in name)
        assert len(scenes) == 15
        for index, folder in enumerate(scenes):
            name = folder.split("/")[-1]
            read = lambda fname: cv2.imdecode(np.frombuffer(photos.read(f"{folder}/{fname}"), np.uint8), cv2.IMREAD_GRAYSCALE)
            left, right = read("im0.png"), read("im1.png")
            disparity = read_pfm(labels.read(folder + "/disp0GT.pfm"))
            calib = photos.read(folder + "/calib.txt").decode()
            document["scene"][name] = {condition: evaluate(left, right, disparity, calib, condition, index)
                                       for condition in ("full", "half")}
    families = {}
    for name in document["scene"]:
        key = {"MotorcycleE": "Motorcycle", "PianoL": "Piano", "PlaytableP": "Playtable"}.get(name, name)
        families.setdefault(key, []).append(name)
    assert len(families) == 12
    for condition in ("full", "half"):
        family_mean = lambda fn: np.array([np.mean([fn(document["scene"][n][condition]) for n in names])
                                           for names in families.values()])
        means = {}
        for key in next(iter(document["scene"].values()))[condition]["score"]:
            means[key] = float(family_mean(lambda r: r["score"][key]).mean())
        contrasts = {}
        boot = np.random.default_rng(924).integers(12, size=(10000, 12))
        for scale in SCALES:
            ours = family_mean(lambda r: r["score"][f"consistency/4/{scale}"])
            for method in METHODS[:2]:
                ref = family_mean(lambda r: r["score"][f"{method}/4/{scale}"])
                diff = ref - ours
                contrasts[f"{method}/{scale}"] = dict(relative=float(diff.mean()/ref.mean()),
                                                       ci=np.quantile(diff[boot].mean(1), [.025,.975]).tolist())
        document["summary"][condition] = dict(coverage=float(family_mean(lambda r:r["coverage"]).mean()),
                                               means=means, paired_4=contrasts)
    (OUT / "stereo_results.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
