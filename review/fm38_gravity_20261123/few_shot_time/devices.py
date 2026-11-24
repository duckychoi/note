"""Generate new fixed devices. The simulator records potential outcomes only for offline scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "observation_time"
sys.path.insert(0, str(PREVIOUS))
from simulate import INERTIA, SNAP, SPEED, build, observe, practice

N_DEVICES = 96
N_HISTORIES = 8
N_PAIRS = 8
N_TEST = 128
N_CAL = N_HISTORIES * N_PAIRS
BUDGETS = (2, 4, 8, 16)
TIMES = (.32, .40, .48)
DOMAINS = ("original", "strong")


def make_device(index):
    family = index // N_DEVICES
    physics, calibration, evaluation = [
        np.random.default_rng(s) for s in np.random.SeedSequence([386704, index]).spawn(3)
    ]
    slope = physics.uniform(25, 45)
    damping = physics.uniform(*((.6, 1.4), (2., 2.8))[family])
    model = build(slope)
    model.dof_damping[0] = INERTIA * damping
    size = N_CAL + N_TEST
    starts, goals = [], []
    for rng, n in ((calibration, N_CAL), (evaluation, N_TEST)):
        starts.extend(np.deg2rad(rng.uniform(-8, 8, n)))
        goals.extend(np.deg2rad(rng.uniform(32, 91, n)))
    probe = np.empty((size, len(TIMES)))
    peak = np.empty((size, len(SPEED)))
    cols = [int(np.flatnonzero(SNAP == t)[0]) for t in TIMES]
    for i, start in enumerate(starts):
        probe[i] = np.asarray(practice(model, start))[cols]
        for a, speed in enumerate(SPEED):
            _, peak[i, a] = observe(model, start, speed)
    return dict(slope=slope, damping=damping, start=starts, goal=goals, probe=probe, peak=peak)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        devices = []
        for item in pool.map(make_device, range(2*N_DEVICES)):
            devices.append(item)
            if len(devices) % 32 == 0:
                print(f"{len(devices)} fixed devices", flush=True)
    np.savez_compressed(HERE/"devices.npz", **{k: np.stack([d[k] for d in devices]) for k in devices[0]})
