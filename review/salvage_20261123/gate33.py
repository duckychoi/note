"""#33 gate 1: correlated-sensor read-budget gate. Preregistered in PREREG_33.md (written before running).

Every arm picks its 12 reads BEFORE seeing x_t, y_t; predicts from read coords only; then y_t is revealed.
Common learner: LMS on read coordinates.  numpy only, single process.
Usage: python gate33.py            -> gate33.json
"""
import json
import os
from pathlib import Path

import numpy as np

D, K, S, TAU, T, NOISE = 200, 12, 3, 480, 1920, 0.3
N_EXPLOIT, N_EXPLORE = 8, 4
VAL = list(range(2000, 2024))
TEST = list(range(3000, 3096))
LRS = (.01, .025, .05, .1, .2)
BASELINES = ("random", "kale_static", "kale_forget", "kale_restart", "signed_corr")
GRIDS = {
    "random": [dict(lr=a) for a in LRS + (.5, 1.0)],
    "kale_static": [dict(lr=a) for a in LRS],
    "kale_forget": [dict(lr=a, g=g) for a in LRS for g in (.99, .995, .998)],
    "kale_restart": [dict(lr=a, R=r) for a in LRS for r in (240, 480, 960)],
    "signed_corr": [dict(lr=a) for a in LRS],
    "resid_ucb": [dict(lr=a, kappa=kp) for a in LRS for kp in (.1, .3, 1.0)],
    "oracle_support": [dict(lr=a) for a in LRS],
}


def streams(seeds, rho):
    xs, ys, act = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        e = rng.normal(size=(T, D))
        x = np.empty_like(e); x[:, 0] = e[:, 0]
        c = np.sqrt(1 - rho**2)
        for j in range(1, D):
            x[:, j] = rho * x[:, j - 1] + c * e[:, j]
        beta = np.zeros((T // TAU, D))
        for b in beta:
            b[rng.choice(D, S, replace=False)] = rng.normal(size=S)
        W = beta[np.arange(T) // TAU]
        xs.append(x); ys.append((x * W).sum(1) + NOISE * rng.normal(size=T)); act.append(W != 0)
    return np.stack(xs), np.stack(ys), np.stack(act)


def topk_excluding(score, mask_excl, k):
    s = np.where(mask_excl, -np.inf, score)
    return np.argsort(-s, axis=1)[:, :k]


def simulate(data, arm, p):
    xs, ys, act = data
    n = len(xs); row = np.arange(n)[:, None]
    rng = np.random.default_rng(777)            # selection randomness, independent of data
    w = np.zeros((n, D)); cred = np.zeros((n, D)); cnt = np.zeros((n, D))
    Sb = np.zeros((n, D)); Nb = np.zeros((n, 1))
    tie = rng.uniform(0, 1e-9, (n, D))
    sq = np.empty((n, T)); allin = np.zeros((n, T), bool); rec = np.empty((n, T))
    pin = N_EXPLORE / (D - N_EXPLOIT)
    for t in range(T):
        u = rng.random((n, D))
        if arm == "random":
            C = np.argsort(u, axis=1)[:, :K]
        elif arm.startswith("kale"):
            if arm == "kale_restart" and t % p["R"] == 0:
                Sb[:] = 0; Nb[:] = 0
            bhat = Sb / np.maximum(Nb, 1e-12)
            E = np.argsort(-(np.abs(bhat) + tie), axis=1)[:, :N_EXPLOIT]
            inE = np.zeros((n, D), bool); inE[row, E] = True
            X = topk_excluding(u, inE, N_EXPLORE)          # uniform among non-E
            C = np.concatenate((E, X), 1)
        elif arm == "signed_corr":
            E = np.argsort(-(np.abs(cred) + tie), axis=1)[:, :K - 2]
            inE = np.zeros((n, D), bool); inE[row, E] = True
            C = np.concatenate((E, topk_excluding(u, inE, 2)), 1)
        elif arm == "resid_ucb":
            E = np.argsort(-(np.abs(w) + tie), axis=1)[:, :N_EXPLOIT]
            inE = np.zeros((n, D), bool); inE[row, E] = True
            ucb = np.abs(cred) + p["kappa"] * np.sqrt(np.log(t + 2) / (cnt + 1))
            X1 = topk_excluding(ucb + tie, inE, 2)
            inE[row, X1] = True
            X2 = topk_excluding(u, inE, 2)
            C = np.concatenate((E, X1, X2), 1)
        elif arm == "oracle_support":
            a = act[:, t]
            pri = np.where(a, -1.0, u)
            C = np.argsort(pri, axis=1)[:, :K]
        xv = xs[row, t, C]                                   # the only data read this round
        err = ys[:, t] - (xv * w[row, C]).sum(1)
        sq[:, t] = err ** 2
        a_t = act[:, t]
        got = a_t[row, C].sum(1)
        rec[:, t] = got / S; allin[:, t] = got == S
        w[row, C] += p["lr"] / np.sqrt(K) * err[:, None] * xv
        if arm.startswith("kale"):
            pj = np.full((n, K), pin); pj[:, :N_EXPLOIT] = 1.0
            g = p.get("g", 1.0)
            Sb *= g; Nb *= g; Nb += 1
            Sb[row, C] += xv * ys[:, t, None] / pj
        elif arm == "signed_corr":
            cred *= .995
            cred[row, C] += .2 * (xv * ys[:, t, None] - cred[row, C])
        elif arm == "resid_ucb":
            cred *= .995
            cred[row, C] += .2 * (xv * err[:, None] - cred[row, C])
            cnt[row, C] += 1
    # recovery after each switch: first round all S active in C for 20 consecutive rounds
    recov = []
    for b in range(1, T // TAU):
        seg = allin[:, b * TAU:(b + 1) * TAU].astype(int)
        cs = np.cumsum(np.pad(seg, ((0, 0), (1, 0))), 1)
        run20 = (cs[:, 20:] - cs[:, :-20]) == 20                    # window [i, i+20)
        first = np.where(run20.any(1), run20.argmax(1), TAU)
        recov.append(first)
    return sq.mean(1), rec.mean(1), np.stack(recov, 1)


def boot(x, seed=98):
    r = np.random.default_rng(seed).choice(x, (5000, len(x))).mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def run_rho(rho):
    val, test = streams(VAL, rho), streams(TEST, rho)
    out = {}
    for arm, grid in GRIDS.items():
        vs = [float(simulate(val, arm, p)[0].mean()) for p in grid]
        best = grid[int(np.argmin(vs))]
        mse, rec, recov = simulate(test, arm, best)
        out[arm] = dict(params=best, val_mse=min(vs), test_mse=float(mse.mean()), test_ci=boot(mse),
                        recall=float(rec.mean()), recovery_median=float(np.median(recov)),
                        recovery_unrecovered_frac=float((recov == TAU).mean()), _mse=mse)
        print(rho, arm, best, "val", round(min(vs), 4), "test", round(float(mse.mean()), 4),
              "recall", round(float(rec.mean()), 3), "recov_med", float(np.median(recov)), flush=True)
    strongest = min(BASELINES, key=lambda a: out[a]["val_mse"])
    new, base = out["resid_ucb"]["_mse"], out[strongest]["_mse"]
    diff = base - new
    rel = (new - base) / base
    verdict = dict(strongest_baseline=strongest, ratio_new_over_base=float(new.mean() / base.mean()),
                   paired_base_minus_new=float(diff.mean()), paired_ci95=boot(diff),
                   rel_degradation_mean=float(rel.mean()), rel_degradation_ci95=boot(rel))
    for v in out.values():
        v.pop("_mse")
    return out, verdict


if __name__ == "__main__":
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    res = {}
    for rho in (0.5, 0.0):
        arms, v = run_rho(rho)
        res[str(rho)] = dict(arms=arms, verdict=v)
        print("VERDICT", rho, json.dumps(v), flush=True)
    v5, v0 = res["0.5"]["verdict"], res["0.0"]["verdict"]
    main_pass = v5["ratio_new_over_base"] <= 0.95 and v5["paired_ci95"][0] > 0
    ctrl_pass = v0["rel_degradation_ci95"][1] <= 0.02
    res["gate"] = dict(main_pass=main_pass, control_pass=ctrl_pass, pass_=main_pass and ctrl_pass)
    print("GATE", res["gate"])
    Path(__file__).with_name("gate33.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
