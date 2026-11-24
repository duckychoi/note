"""Run the frozen single-snapshot policies. Evaluate the held-out starts only after locking times."""
import argparse
import json
import sys

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from devices import HERE, PREVIOUS, N_CAL, SNAP, TIMES
from choose import digest
sys.path.insert(0, str(PREVIOUS))
from policy import Camera, features, inputs, pick
from swing import BAND


def run(calibration):
    if not calibration:
        locks = json.loads((HERE/"lock.json").read_text())
        assert all(digest(HERE/file) == h for file, h in locks.items())
    model_file = PREVIOUS/"policies.joblib"
    original = json.loads((PREVIOUS/"selection.json").read_text())
    assert digest(model_file) == original["model_sha256"]
    models = joblib.load(model_file)
    camera = Camera()
    sl = slice(None, N_CAL) if calibration else slice(N_CAL, None)
    times = (TIMES[0], TIMES[2]) if calibration else TIMES
    with np.load(HERE/"devices.npz") as raw:
        start = raw["start"][:, sl]
        goals = raw["goal"][:, sl]
        photos = raw["probe"][:, sl]
        peak = raw["peak"][:, sl]
    d, n = start.shape
    shots = camera.decode(photos[:, :, [TIMES.index(t) for t in times]])
    starting = camera.decode(start)
    choices = np.empty((d, len(models), n, len(times)), dtype=np.uint8)
    for slot, t in enumerate(times):
        j = int(np.flatnonzero(SNAP==t)[0])
        for first in range(0, d, 16):
            section = slice(first, first+16)
            s = starting[section].ravel()
            snapshot = np.full((len(s),len(SNAP)), np.nan)
            snapshot[:, j] = shots[section, :, slot].ravel()
            data = dict(start=s, snap=snapshot, goal=goals[section].ravel(),
                        action=np.broadcast_to(np.arange(4),(len(s),4)))
            p = inputs(data, j)
            x = features(p)
            for k, model in enumerate(models):
                pred = p["prior"] + model["prior"][j].predict(x)
                choices[section, k, :, slot] = pick(pred, data).reshape(-1, n)
    win = abs(peak-goals[...,None]) <= BAND
    observed = win[np.arange(d)[:,None,None,None], np.arange(n)[None,None,:,None], choices]
    name = "calibration" if calibration else "test"
    np.savez_compressed(HERE/f"{name}.npz", actions=choices, observed=observed, times=times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration","test"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    with threadpool_limits(limits=2):
        run(args.stage=="calibration")
