"""A device's camera time depends only on its own binary calibration outcomes."""
import hashlib
import json

import numpy as np

from devices import BUDGETS, HERE, N_CAL, N_HISTORIES, N_PAIRS, TIMES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(observed):
    # Shape: device, model, independent history, paired start/goal, camera time.
    paired_difference = observed[..., 0].astype(int) - observed[..., 1]
    margins = np.stack([paired_difference[..., :n//2].sum(-1) for n in BUDGETS], axis=-1)
    return np.where(margins > 0, TIMES[0], TIMES[2]), margins


def lock():
    with np.load(HERE/"calibration.npz") as logged:
        outcomes = logged["observed"]
        assert outcomes.shape[-2:] == (N_CAL, 2)
        outcomes = outcomes.reshape(*outcomes.shape[:2], N_HISTORIES, N_PAIRS, 2)
    selected, margins = choose(outcomes)
    path = HERE/"locked_times.npz"
    np.savez_compressed(path, selected=selected, margins=margins)
    hashes = {p.name: digest(p) for p in (path, HERE/"calibration.npz", HERE/"protocol.md")}
    (HERE/"lock.json").write_text(json.dumps(hashes, indent=2), encoding="utf8")


if __name__ == "__main__":
    lock()
