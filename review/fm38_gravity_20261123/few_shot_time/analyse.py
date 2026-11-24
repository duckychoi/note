"""Compare device-level paired rates, preserving nested budgets and reused test scenes."""
import json

import numpy as np

from devices import BUDGETS, DOMAINS, HERE, N_CAL, N_DEVICES, N_HISTORIES, N_PAIRS, N_TEST, TIMES
from swing import BAND

FAMILIES = (*DOMAINS, "balanced")


def intervals(by_device):
    # Device x budget x fixed-time contrast in each of two independently sampled strata.
    n = N_DEVICES
    rng = np.random.default_rng(386705)
    draws = np.empty((64000, len(FAMILIES), len(BUDGETS), len(TIMES)))
    for start in range(0, len(draws), 500):
        batch = draws[start:start+500]
        for group, x in enumerate(by_device):
            w = rng.multinomial(n, np.full(n, 1/n), size=len(batch))
            batch[:,group] = (w @ x.reshape(n, -1) / n).reshape(len(batch), len(BUDGETS), -1)
        batch[:,2] = batch[:,:2].mean(1)
    alpha = .05/(2*int(np.prod(draws.shape[1:])))
    return np.quantile(draws, [alpha, 1-alpha], axis=0)


def evaluate():
    with np.load(HERE/"locked_times.npz") as f:
        early = f["selected"] == TIMES[0]
        ties = f["margins"] == 0
    with np.load(HERE/"test.npz") as f:
        np.testing.assert_array_equal(f["times"],TIMES)
        fixed = f["observed"].mean(axis=2)  # device, trained model, time
    with np.load(HERE/"calibration.npz") as f:
        informative = (f["observed"][...,0] != f["observed"][...,1]).mean((1,2))
    with np.load(HERE/"devices.npz") as raw:
        reachable = (abs(raw["peak"][:,N_CAL:]-raw["goal"][:,N_CAL:,None]) <= BAND).any(2).mean(1)
    freq = early.mean(axis=2)
    adaptive = fixed[:,:,2,None] + freq*(fixed[:,:,0,None]-fixed[:,:,2,None])
    paired = adaptive.mean(1)[:,:,None] - fixed.mean(1)[:,None,:]
    groups = np.split(paired, 2)
    bounds = intervals(groups)
    report = dict(devices_per_domain=N_DEVICES, test_per_device=N_TEST, histories=N_HISTORIES,
                  models=fixed.shape[1], calibration_pairs=N_PAIRS, domains={})
    for g, domain in enumerate(FAMILIES):
        ix = slice(None) if g == 2 else slice(g*N_DEVICES,(g+1)*N_DEVICES)
        data = dict(fixed=dict(zip(map(str,TIMES), fixed[ix].mean((0,1)).tolist())),
                    reachable=float(reachable[ix].mean()), informative=float(informative[ix].mean()), budgets={})
        for b, budget in enumerate(BUDGETS):
            result = dict(rate=float(adaptive[ix,:,b].mean()), early=float(early[ix,...,b].mean()),
                          tie=float(ties[ix,...,b].mean()), comparisons={})
            for j, t in enumerate(TIMES):
                delta = float(paired[ix,b,j].mean())
                lo, hi = bounds[:,g,b,j]
                result["comparisons"][str(t)] = dict(delta=delta, ci=[float(lo),float(hi)],
                    positive=bool(lo>0), practical=bool(lo>0 and delta>=.03))
            data["budgets"][str(budget)] = result
        report["domains"][domain] = data
    return report


if __name__ == "__main__":
    (HERE/"results.json").write_text(json.dumps(evaluate(), indent=2), encoding="utf8")
