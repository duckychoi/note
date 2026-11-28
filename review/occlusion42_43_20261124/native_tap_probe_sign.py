"""Evaluate the sign of depth corrections at frozen Middlebury tracker queries."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval/native-surface-direction"
FOCAL_BASELINE = (3740 / 3) * .16


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def counts(truth, prediction, eligible):
    valid = eligible & np.isfinite(prediction) & (prediction != 0)
    near, far = truth > 0, truth < 0
    return dict(eligible=int(eligible.sum()), near=int((eligible & near).sum()), far=int((eligible & far).sum()),
                predicted=int(valid.sum()), correct=int((valid & (prediction == np.sign(truth))).sum()),
                covered_near=int((valid & near).sum()), covered_far=int((valid & far).sum()),
                hit_near=int((valid & near & (prediction > 0)).sum()),
                hit_far=int((valid & far & (prediction < 0)).sum()))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def rates(row):
    near, far = row["near"], row["far"]
    covered_near, covered_far = row["covered_near"], row["covered_far"]
    return dict(coverage=ratio(row["predicted"], row["eligible"]),
                accuracy=ratio(row["correct"], row["predicted"]),
                near_recall=ratio(row["hit_near"], near), far_recall=ratio(row["hit_far"], far),
                covered_near_recall=ratio(row["hit_near"], covered_near),
                covered_far_recall=ratio(row["hit_far"], covered_far),
                always_near=ratio(near, row["eligible"]), always_far=ratio(far, row["eligible"]),
                majority_on_covered_points=ratio(max(covered_near, covered_far), row["predicted"]),
                always_near_on_covered_points=ratio(covered_near, row["predicted"]),
                always_far_on_covered_points=ratio(covered_far, row["predicted"]))


def average(stats, group_by_family):
    by_group = defaultdict(list)
    for scene, row in stats.items():
        by_group[re.sub(r"\d+$", "", scene) if group_by_family else scene].append(rates(row))
    defined = {metric: [] for metric in next(iter(by_group.values()))[0]}
    for group in by_group.values():
        for metric in defined:
            available = [row[metric] for row in group if row[metric] is not None]
            if available:
                defined[metric].append(float(np.mean(available)))
    return {key: float(np.mean(values)) if values else None for key, values in defined.items()}


def measure(scene, source, tracker):
    x, y = tracker[f"{scene}/query_xy"].astype(int).T
    with np.load(source) as student:
        base = student["base"][y, x].astype(float)
        k = float(student["k"])
        h = student["h"][y, x]
        student_sha256 = str(student["sha256"])
    rho = 1 / (k * base)
    calibration = float((ROOT.parent / f"repeat-capture/physical/{scene}/dmin.txt").read_text()) / 3
    rows = {}
    for view, fraction in ((3, .5), (5, 1.)):
        estimates = {"tap": {}, "sgbm": {}}
        for light in (1, 2):
            prefix = f"{scene}/L1-L{light}/view{view}"
            truth = tracker[f"{prefix}/gt_inverse_depth_per_m"] - rho
            eligible = tracker[f"{prefix}/gt_eligible"] & np.isfinite(rho) & (rho > 0) & (abs(truth) > 1e-10)
            if light == 1:
                rows[f"student/view{view}"] = counts(truth, -np.sign(h), eligible)
            for model in estimates:
                shift = x - tracker[f"{prefix}/{model}_xy"][:, 0]
                value = (shift / fraction + calibration) / FOCAL_BASELINE - rho
                visible = tracker[f"{prefix}/{model}_visible"]
                for threshold in (0, 1):
                    key = f"{model}/view{view}/{threshold}px"
                    sign = np.where(visible & (abs(value) > threshold / (fraction * FOCAL_BASELINE)),
                                    np.sign(value), 0.)
                    estimates[model].setdefault(threshold, []).append(sign)
                    rows[f"{key}/light{light}"] = counts(truth, sign, eligible)
                    if light == 2:
                        prior = estimates[model][threshold][0]
                        same = np.where(prior == sign, sign, 0.)
                        rows[f"{key}/agreement"] = counts(truth, same, eligible)
    return rows, k, student_sha256


def main():
    tracker_path = ROOT / "tap-probe/result.npz"
    manifest = json.loads((ROOT / "tap-probe/result.json").read_text(encoding="utf8"))
    sources = ROOT / "middlebury-native"
    rows, version, k_values = {}, set(), set()
    file_hashes = {}
    with np.load(tracker_path) as tracker:
        for scene in manifest["rows"]:
            path = sources / f"{scene}.npz"
            result, k, student_sha256 = measure(scene, path, tracker)
            rows[scene] = result
            version.add(student_sha256)
            k_values.add(k)
            file_hashes[scene] = digest(path)
    assert len(rows) == 21 and len(version) == len(k_values) == 1
    summaries = {}
    for key in next(iter(rows.values())):
        by_scene = {scene: row[key] for scene, row in rows.items()}
        pooled = sum((Counter(row) for row in by_scene.values()), Counter())
        summaries[key] = dict(family_average=average(by_scene, True), pooled_counts=dict(pooled), pooled=rates(pooled))
    report = dict(k=k_values.pop(), student_sha256=version.pop(), tracker_npz_sha256=digest(tracker_path),
                  depth_npz_sha256_by_scene=file_hashes, summary=summaries, scene_counts=rows)
    out = ROOT / "tap-probe/correction-sign.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
