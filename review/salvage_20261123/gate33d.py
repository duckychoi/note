"""#33 gate 4 (PREREG_33.md "관문 4"): unannounced IRREGULAR support switches.
Switch gaps = 60 + Geometric(mean 420) rounds (mean ~480), no period hint to any arm. Same 12/200 read budget.
Reuses gate33c.simulate / kale_ipw_lasso (they only see x, y; |beta| is used for scoring only).
numpy only, single process.  Fresh test seeds 6000-6095; validation 2100-2123 (new, irregular streams).
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import gate33 as G
import gate33c as GC

D, K, S, T, NOISE = G.D, G.K, G.S, G.T, G.NOISE
VAL, TEST = list(range(2100, 2124)), list(range(6000, 6096))
LRS = GC.LRS
CAP, WIN = 240, 20


def streams(seeds, rho):
    xs, ys, ab, sw = [], [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        e = rng.normal(size=(T, D))
        x = np.empty_like(e); x[:, 0] = e[:, 0]
        c = np.sqrt(1 - rho**2)
        for j in range(1, D):
            x[:, j] = rho * x[:, j - 1] + c * e[:, j]
        starts = [0]
        while True:
            g = 60 + rng.geometric(1 / 420)
            if starts[-1] + g >= T:
                break
            starts.append(starts[-1] + g)
        W = np.zeros((T, D))
        bounds = starts + [T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            beta = np.zeros(D); beta[rng.choice(D, S, replace=False)] = rng.normal(size=S)
            W[a:b] = beta
        xs.append(x); ys.append((x * W).sum(1) + NOISE * rng.normal(size=T)); ab.append(np.abs(W)); sw.append(starts[1:])
    return (np.stack(xs), np.stack(ys), np.stack(ab)), sw


def recovery_times(wr, sw, theta):
    """Per switch with >= CAP rounds until the next switch/end: first r with mean(wr[s+r : s+r+WIN]) >= theta, capped at CAP."""
    out = []
    for i, ss in enumerate(sw):
        nxt = list(ss[1:]) + [T]
        for s, e in zip(ss, nxt):
            if e - s < CAP + WIN:
                continue
            seg = wr[i, s:s + CAP + WIN]
            cs = np.concatenate(([0], np.cumsum(seg)))
            m = (cs[WIN:] - cs[:-WIN]) / WIN
            ok = np.where(m[:CAP] >= theta)[0]
            out.append((i, ok[0] if len(ok) else CAP))
    return out


def per_seed_rmst(rt, n):
    v = np.full(n, np.nan); acc = {}
    for i, r in rt:
        acc.setdefault(i, []).append(r)
    for i, rs in acc.items():
        v[i] = np.mean(rs)
    return v


def run(data, name, arm, q):
    return GC.kale_ipw_lasso(data, q) if arm == "kipw" else GC.simulate(data, arm, q)


def main():
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    t0 = time.time(); res = {}
    fam = {
        "ucb_resid": [dict(lr=a, explore="ucb_resid", kappa=k) for a in LRS for k in (1, 2, 4, 8, 16, 32)],
        "ucb_target": [dict(lr=a, explore="ucb_target", kappa=k) for a in LRS for k in (1, 2, 4, 8, 16, 32)],
        "count2rand2": [dict(lr=a, explore="count2rand2") for a in LRS],
        "rand4": [dict(lr=a, explore="rand4") for a in LRS],
    }
    base = {
        "random": [dict(lr=a) for a in (0.0, .001, .003, .01, .025, .05)],
        "signed_corr": [dict(lr=a, decay=dc, eta=e) for a in LRS for dc in (.98, .99, .995, .999) for e in (.05, .2, .5)],
        "kale_forget": [dict(lr=a, g=g) for a in LRS for g in (.98, .99, .995, .998)],
        "kale_restart": [dict(lr=a, R=r) for a in LRS for r in (120, 240, 480, 960)],
    }
    kl = [dict(lam=l, mu=m) for l in (.2, .5, 1.0, 2.0) for m in (.5, 2.0)]
    kg = {"kaleIPWLasso_forget": [dict(mode="forget", g=g, **q) for q in kl for g in (.99, .995)],
          "kaleIPWLasso_restart": [dict(mode="restart", R=r, **q) for q in kl for r in (240, 480)]}
    for rho in (0.5, 0.0):
        val, swv = streams(VAL, rho)
        tuned = {}
        for name, grid in list(fam.items()) + list(base.items()) + list(kg.items()):
            arm = "wret" if name in fam else ("kipw" if name in kg else name)
            vs = [float(run(val, name, arm, q)[0].mean()) for q in grid]
            tuned[name] = (arm, grid[int(np.argmin(vs))], min(vs))
            print(rho, "val", name, tuned[name][1], round(min(vs), 4), round(time.time() - t0), "s", flush=True)
        rep = min(fam, key=lambda k: tuned[k][2])
        strongest = min(list(base) + list(kg), key=lambda k: tuned[k][2])
        _, wr_v, _ = run(val, rep, "wret", tuned[rep][1])
        cens = {th: float(np.mean([r == CAP for _, r in recovery_times(wr_v, swv, th)])) for th in (.9, .8, .7)}
        theta = next((th for th in (.9, .8, .7) if cens[th] <= .25), .7)
        print(rho, "rep", rep, "strongest", strongest, "val censored by theta", cens, "theta", theta, flush=True)
        # residual-credit-only ablation at fixed other components (parameters of the better ucb variant)
        ub = min(("ucb_resid", "ucb_target"), key=lambda k: tuned[k][2])
        fixed = dict(tuned[ub][1])
        abl = {"ucb_resid_fixed": dict(fixed, explore="ucb_resid"), "ucb_target_fixed": dict(fixed, explore="ucb_target")}
        test, swt = streams(TEST, rho)
        n = len(TEST); out, mse, rm = {}, {}, {}
        items = [(k, v[0], v[1]) for k, v in tuned.items()] + [(k, "wret", q) for k, q in abl.items()]
        for name, arm, q in items:
            m, wr, sq = run(test, name, arm, q)
            rt = recovery_times(wr, swt, theta)
            r = per_seed_rmst(rt, n)
            mse[name], rm[name] = m, r
            out[name] = dict(params=q, test_mse=float(m.mean()), test_ci=GC.boot(m), rmst=float(np.nanmean(r)),
                             censored=float(np.mean([x == CAP for _, x in rt])), n_switches_scored=len(rt))
            print(rho, name, "mse", round(float(m.mean()), 4), "rmst", round(float(np.nanmean(r)), 1), "cens", round(out[name]["censored"], 3), flush=True)
        ok = ~np.isnan(rm[rep]) & ~np.isnan(rm[strongest])
        v = dict(rep=rep, strongest=strongest, theta=theta, val_censored=cens,
                 mse_improvement=float(1 - mse[rep].mean() / mse[strongest].mean()),
                 mse_improvement_ci=GC.boot_ratio(mse[rep], mse[strongest]),
                 five_pct_margin_ci=GC.boot(0.95 * mse[strongest] - mse[rep]),
                 rmst_diff_base_minus_new=float((rm[strongest][ok] - rm[rep][ok]).mean()),
                 rmst_diff_ci=GC.boot(rm[strongest][ok] - rm[rep][ok]), n_seeds_rmst=int(ok.sum()),
                 rel_degradation_ci=GC.boot((mse[rep] - mse[strongest]) / mse[strongest]),
                 resid_minus_target_fixed=dict(mse=GC.boot(mse["ucb_resid_fixed"] - mse["ucb_target_fixed"]),
                                               mean=float((mse["ucb_resid_fixed"] - mse["ucb_target_fixed"]).mean())))
        res[str(rho)] = dict(arms=out, verdict=v, per_seed=dict(mse={k: x.tolist() for k, x in mse.items()}))
        print("VERDICT", rho, json.dumps(v), flush=True)
    v5, v0 = res["0.5"]["verdict"], res["0.0"]["verdict"]
    res["gate"] = dict(mse_pass=v5["five_pct_margin_ci"][0] > 0, recovery_pass=v5["rmst_diff_ci"][0] > 0,
                       control_pass=v0["rel_degradation_ci"][1] <= .02)
    res["gate"]["pass_"] = all(res["gate"].values())
    print("GATE", res["gate"], flush=True)
    Path(__file__).with_name("gate33d.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
