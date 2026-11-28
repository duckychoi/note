"""Recursive rare-category test, common random numbers and fixed total real budget."""
import json
from pathlib import Path

import numpy as np


def simulate(target, anchors, copy, seed):
    rng = np.random.default_rng(seed)
    paths = 4096
    counts = np.array([2, 3, 5, 8, 12, 20, 30, 48])
    ids = np.repeat(np.arange(len(counts)), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    real_hits = rng.binomial(anchors, target, paths)
    initial = (32 * .08 + real_hits) / (32 + anchors)
    current = {name: initial.copy() for name in ("flat", "equal_origin", "known_corr", "accumulate", "keep_initial")}
    history = {name: [] for name in current}
    weights = counts / (1 + (counts - 1) * copy**2)
    prior_mass = 32 + anchors
    positives = prior_mass * initial
    for t in range(24):
        parent_draw = rng.random((paths, len(counts)))
        copy_mask = rng.random((paths, len(ids))) < copy
        independent_draw = rng.random(copy_mask.shape)
        for name, prob in current.items():
            if name != "keep_initial":
                parents = parent_draw < prob[:, None]
                labels = np.where(copy_mask, parents[:, ids], independent_draw < prob[:, None])
                group_means = np.add.reduceat(labels.astype(np.int16), starts, axis=1) / counts
                if name == "accumulate":
                    positives += labels.sum(1)
                    updated = positives / (prior_mass + (t + 1) * len(ids))
                elif name == "flat":
                    updated = labels.mean(1)
                elif name == "equal_origin":
                    updated = group_means.mean(1)
                else:
                    updated = np.average(group_means, axis=1, weights=weights)
                prob = current[name] = updated
            history[name].append(dict(below_one_percent=float(np.mean(prob < .01)),
                                      mae=float(np.mean(np.abs(prob-target))),
                                      mean_probability=float(prob.mean())))
    return history


if __name__ == "__main__":
    results = {}
    for target in (.08, .03):
        for anchors in (0, 8):
            for copy in (0., .6):
                key = f"target{target}_anchors{anchors}_copy{copy}"
                results[key] = simulate(target, anchors, copy, 42071)
    Path(__file__).with_name("rare42.json").write_text(json.dumps(results, indent=1), encoding="utf8")
