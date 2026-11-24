"""Exploratory control with the same strong-world calibration labels as the timing selector."""
import json

import numpy as np

from simulate import CONFIRM_SEEDS, HERE, TEST_SEEDS
from swing import BAND


def load(seed):
    with np.load(HERE / f"strong_{seed}.npz") as raw:
        bucket = np.floor((np.rad2deg(raw["goal"]) - 32) / 5).astype(int)
        wins = np.abs(raw["peak"] - raw["goal"][:, None]) <= BAND
        return bucket, wins


def run():
    # Fixed five-degree bins. Learn one action per goal bin, using no camera.
    counts = np.zeros((12, 4), dtype=int)
    sizes = np.zeros(12, dtype=int)
    for seed in TEST_SEEDS:
        bins, wins = load(seed)
        np.add.at(counts, bins, wins)
        np.add.at(sizes, bins, 1)
    assert np.all(sizes)
    actions = counts.argmax(axis=1)
    selected = json.loads((HERE / "strong_selection.json").read_text())["idx"]
    success = []
    paired = []
    for seed in CONFIRM_SEEDS:
        bins, wins = load(seed)
        baseline = wins[np.arange(len(wins)), actions[bins]]
        with np.load(HERE / f"strong_{seed}_choices.npz") as policy:
            chosen = wins[np.arange(len(wins)), policy["prior"][:, selected]].mean(0)
        success.extend(baseline)
        paired.extend(chosen - baseline)
    return dict(bin_start_deg=32, bin_width_deg=5, actions=(actions+1).tolist(),
                counts=sizes.tolist(), rate=float(np.mean(success)),
                timing_minus_goal=float(np.mean(paired)))


if __name__ == "__main__":
    (HERE / "goal_control.json").write_text(json.dumps(run(), indent=2), encoding="utf8")
