"""STSA (#32, docs/02) sanity: add the missing linear integral baseline and heavy-tailed noise.

Same scalar tracking model as src/stsa_robust_sa.py:
  z_t = target_t + noise,  d_t = z_t - th_t,  th <- th + update(d) + dist_t
Methods
  sa_const : th += a d
  pi_lin   : u += a k2 d ; th += a (k1 d + u)          (linear PI / 2nd-order SA = steady-state Kalman with drift state)
  pi_huber : same as pi_lin but d -> clip(d, -c, c)    (bounded-influence linear PI)
  stw      : u += a k2 sign(d) ; th += a (k1 sqrt|d| sign d + u)   (original proposal)
Gains selected on validation seeds 0..99 (tail RMSE), reported on test seeds 1000..1399.
Noise: Gaussian sigma=1 (original) and Student-t nu=1.2 (heavy tail, scaled to same median |noise|).
numpy only, vectorized over seeds.  Writes stsa_sanity.json.
"""
import itertools
import sys
import json
from pathlib import Path
import numpy as np

T = 3000
VAL = np.arange(0, 100)
TEST = np.arange(1000, 1400)

SCEN = {
    "A_stationary": (lambda t: 2.0, lambda t: 0.0),
    "B_ramp": (lambda t: 2.0 + 0.004 * t, lambda t: 0.0),
    "D_const_dist": (lambda t: 2.0, lambda t: 0.05),
    "E_sine_dist": (lambda t: 2.0, lambda t: 0.05 * np.sin(0.01 * t)),
}

A_GRID = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
GRIDS = {
    "sa_const": [dict(a=a) for a in A_GRID],
    "pi_lin": [dict(a=a, k1=k1, k2=k2) for a in A_GRID for k1 in (0.25, 0.5, 1.0, 1.5) for k2 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)],
    "pi_huber": [dict(a=a, k1=k1, k2=k2, c=c) for a in A_GRID for k1 in (0.25, 0.5, 1.0, 1.5) for k2 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) for c in (0.5, 1.5)],
    "stw": [dict(a=a, k1=k1, k2=k2) for a in (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4) for k1 in (0.25, 0.5, 1.0, 1.5) for k2 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)],
}


def noise(seeds, kind):
    rngs = [np.random.default_rng(int(s)) for s in seeds]
    if kind == "gauss":
        return np.stack([r.normal(0, 1, T) for r in rngs], 1)
    x = np.stack([r.standard_t(1.2, T) for r in rngs], 1)
    return x * (0.6745 / np.median(np.abs(np.random.default_rng(99).standard_t(1.2, 200000))))  # same median |noise| as N(0,1)


def run(method, p, scen, eps):
    tf, df = SCEN[scen]
    S = eps.shape[1]
    th = np.zeros(S); u = np.zeros(S); err2 = np.empty((T, S))
    for t in range(T):
        tgt = tf(t); d = tgt + eps[t] - th
        if method == "sa_const":
            th = th + p["a"] * d
        elif method == "pi_lin":
            u = u + p["a"] * p["k2"] * d; th = th + p["a"] * (p["k1"] * d + u)
        elif method == "pi_huber":
            dc = np.clip(d, -p["c"], p["c"]); u = u + p["a"] * p["k2"] * dc; th = th + p["a"] * (p["k1"] * dc + u)
        else:
            sg = np.sign(d); u = u + p["a"] * p["k2"] * sg; th = th + p["a"] * (p["k1"] * np.sqrt(np.abs(d)) * sg + u)
        th = th + df(t)
        th = np.clip(th, -1e6, 1e6)
        err2[t] = (th - tgt) ** 2
    return err2


def tail(err2):  # tail RMSE averaged over seeds, plus per-seed tail mean square (for CI)
    e = err2[int(0.8 * T):]
    return float(np.sqrt(e.mean())), e.mean(0)


ONLY = sys.argv[1:]
out = {}
for nk in ("gauss", "t1.2"):
    ev, et = noise(VAL, nk), noise(TEST, nk)
    for sc in SCEN:
        row = {}
        for m, grid in GRIDS.items():
            if ONLY and m not in ONLY: continue
            best = min(grid, key=lambda p: tail(run(m, p, sc, ev))[0])
            r, per = tail(run(m, best, sc, et))
            row[m] = dict(params=best, test_tail_rmse=r, per_seed_ms=per)
        # paired comparison stw vs pi_huber (per-seed tail mean-square)
        if "stw" in row and "pi_huber" in row:
            dd = row["stw"]["per_seed_ms"] - row["pi_huber"]["per_seed_ms"]
            row["stw_minus_pihuber_ms"] = [float(dd.mean()), float(1.96 * dd.std() / np.sqrt(len(dd)))]
        for m in row:
            if isinstance(row[m], dict): row[m].pop("per_seed_ms")
        out[f"{nk}/{sc}"] = row
        print(nk, sc, {m: (round(row[m]["test_tail_rmse"], 4), row[m]["params"]) for m in row if isinstance(row[m], dict)}, flush=True)

Path(__file__).with_name("stsa_sanity_wide_" + "_".join(ONLY) + ".json" if ONLY else "stsa_sanity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
