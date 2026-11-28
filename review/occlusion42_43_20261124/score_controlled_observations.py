"""Fit synthetic-only gains and evaluate them on scene-balanced paired controls."""
from collections import defaultdict
import json
import re

import numpy as np

from common import ALL
from controlled_observations import MODES, RUN, write_json


def scene_rows(split):
    rows = [json.loads(p.read_text(encoding="utf8")) for p in sorted(RUN.glob(f"{split}-*.json"))]
    expected = {"train": 48, "dev": 46, "real": len(ALL)}
    if len(rows) != expected[split] or len({r["scene"] for r in rows}) != len(rows):
        raise ValueError(f"Incomplete {split}, found {len(rows)} rows")
    return rows


def case_moments(case, mode):
    return np.asarray(case["moments"][mode], dtype=np.float64)/case["eligible"]


def fit(rows, modes=MODES):
    # Within a scene all valid orientations and capture conditions get equal weight.
    per_scene = []
    for row in rows:
        cases = [c for c in row["cases"] if c["eligible"]]
        if cases:
            per_scene.append([np.mean([case_moments(c, m) for c in cases], axis=0) for m in modes])
    moments = np.mean(per_scene, axis=0)
    cross, spread = moments[:, :, 2], moments[:, :, 3]
    pooled = float(np.clip(-cross.sum()/spread.sum(), 0, 1))
    fixed = np.clip(-cross.sum(axis=1)/spread.sum(axis=1), 0, 1)
    ridge = .05*spread.sum(axis=1, keepdims=True)/spread.shape[1]
    adaptive = np.clip((-cross+ridge*fixed[:, None])/(spread+ridge), 0, 1)
    return pooled, fixed, adaptive


def evaluate(mom, pooled, fixed, adaptive):
    freq, square, cross, spread, oracle_endpoints = mom.T
    def mse(k):
        return (square+2*k*cross+k*k*spread).sum()
    # Exact expected MSE over all permutations within this one pair and condition.
    shuffled = square.sum()+2*np.dot(freq, adaptive)*cross.sum()+np.dot(freq, adaptive**2)*spread.sum()
    return np.array([mse(0.), mse(1.), mse(pooled), mse(fixed), mse(adaptive), shuffled, oracle_endpoints.sum()])


def score(rows, gains, split, conditions=MODES, with_fallback=False):
    pooled, fixed, adaptive = gains
    groups = defaultdict(list)
    total = eligible = 0
    for row in rows:
        cases = [c for c in row["cases"] if c["eligible"] or (with_fallback and c["full_valid"])]
        total += sum(c["grid"] for c in row["cases"])
        eligible += sum(c["eligible"] for c in cases)
        if not cases:
            continue
        key = re.sub(r"\d+$", "", row["scene"]) if split == "real" else row["scene"]
        by_case = []
        for case in cases:
            risk = np.array([evaluate(case_moments(case, mode), pooled, fixed[i], adaptive[i])
                             for i, mode in enumerate(conditions)]) if case["eligible"] else np.zeros((len(conditions), 7))
            if with_fallback:
                risk = (risk*case["eligible"]+case["unsupported_anchor_sse"])/case["full_valid"]
            by_case.append(risk)
        groups[key].append(np.mean(by_case, axis=0))
    data = np.stack([np.mean(values, axis=0) for values in groups.values()])
    data = np.concatenate((data, data.mean(axis=1, keepdims=True)), axis=1)
    modes = [*conditions, "equal_condition_mean"]
    methods = ["anchor", "neighbor", "global_fixed", "condition_fixed", "adaptive", "shuffled", "oracle_best_endpoint"]
    indices = np.random.default_rng(4342).integers(len(data), size=(10000, len(data)))
    means = data.mean(axis=0)
    summary = {}
    for index, mode in enumerate(modes):
        paired = data[:, index, :][indices].mean(axis=1)
        delta = 100*(1-paired[:, 4]/paired[:, 3])
        summary[mode] = dict(log_mse=dict(zip(methods, means[index].tolist())),
            reduction_vs_condition_fixed_pct=100*(1-means[index, 4]/means[index, 3]),
            reduction_vs_shuffled_pct=100*(1-means[index, 4]/means[index, 5]),
            paired_scene_ci95_pct=np.quantile(delta, [.025, .975]).tolist(),
            improved_groups=int((data[:, index, 4] < data[:, index, 3]).sum()))
    return dict(scenes=len(groups), valid_captures=sum(map(len, groups.values())), captures=len(rows),
                eligible=eligible, grid=total, scenes_in_order=list(groups), mode_order=modes,
                method_order=methods, scene_log_mse=data.tolist(), summary=summary)


def main():
    train, dev, real = (scene_rows(s) for s in ("train", "dev", "real"))
    assert not ({r["scene"] for r in train} & {r["scene"] for r in dev})
    assert len({r["sha256"] for r in train+dev+real}) == 1
    gains = fit(train)
    domains = {s: score(rows, gains, s) for s, rows in (("dev", dev), ("real", real))}
    decision = all(min(r["summary"]["equal_condition_mean"][key] for key in
                   ("reduction_vs_condition_fixed_pct", "reduction_vs_shuffled_pct")) > 0 for r in domains.values())
    result = dict(calibration_scene_ids=[r["scene"] for r in train if any(c["eligible"] for c in r["cases"])],
                  sha256=train[0]["sha256"], pooled_gain=gains[0], condition_gain=gains[1].tolist(),
                  pixel_gain=gains[2].tolist(), fit_ridge_fraction=.05, **domains,
                  observation_pass_for_student_exploration=decision, independent_student_3pct_confirmed=False)
    write_json(RUN/"results.json", result)
    for s, r in domains.items():
        print(s, r["scenes"], r["summary"]["equal_condition_mean"])
    print("student exploration gate", decision)


if __name__ == "__main__":
    main()
