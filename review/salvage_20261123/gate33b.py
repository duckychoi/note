"""#33 gate 2 (PREREG_33.md "관문 2"): kappa extension, one-factor ablations, weighted recovery metric,
Kale 2017 faithful-estimator baseline, confirmation on fresh test seeds 4000-4095.
Reuses gate33 constants/stream rule (gate33.py is not modified).  numpy only, single process.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import gate33 as G

D, K, S, TAU, T, NOISE = G.D, G.K, G.S, G.TAU, G.T, G.NOISE
VAL = G.VAL
TEST = list(range(4000, 4096))
LRS = G.LRS
KAPPAS = (.1, .3, 1.0, 2.0, 4.0, 8.0)
K0, KSUP, SOLVE_EVERY = 8, 4, 32


def streams(seeds, rho):
    """Identical draws to gate33.streams, but also returns |beta| per round."""
    xs, ys, ab = [], [], []
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
        xs.append(x); ys.append((x * W).sum(1) + NOISE * rng.normal(size=T)); ab.append(np.abs(W))
    return np.stack(xs), np.stack(ys), np.stack(ab)


def topk_excl(score, excl, k):
    return np.argsort(-np.where(excl, -np.inf, score), axis=1)[:, :k]


def simulate(data, arm, p):
    """arm in random/kale_static/kale_forget/kale_restart/signed_corr/resid (resid takes credit, retain, kappa)."""
    xs, ys, ab = data
    n = len(xs); row = np.arange(n)[:, None]
    rng = np.random.default_rng(777)
    w = np.zeros((n, D)); cred = np.zeros((n, D)); cnt = np.zeros((n, D))
    Sb = np.zeros((n, D)); Nb = np.zeros((n, 1))
    tie = rng.uniform(0, 1e-9, (n, D))
    sq = np.empty((n, T)); wr = np.empty((n, T))
    pin = G.N_EXPLORE / (D - G.N_EXPLOIT)
    for t in range(T):
        u = rng.random((n, D))
        if arm == "random":
            C = np.argsort(u, axis=1)[:, :K]
        elif arm.startswith("kale"):
            if arm == "kale_restart" and t % p["R"] == 0:
                Sb[:] = 0; Nb[:] = 0
            E = np.argsort(-(np.abs(Sb / np.maximum(Nb, 1e-12)) + tie), axis=1)[:, :G.N_EXPLOIT]
            inE = np.zeros((n, D), bool); inE[row, E] = True
            C = np.concatenate((E, topk_excl(u, inE, G.N_EXPLORE)), 1)
        elif arm == "signed_corr":
            E = np.argsort(-(np.abs(cred) + tie), axis=1)[:, :K - 2]
            inE = np.zeros((n, D), bool); inE[row, E] = True
            C = np.concatenate((E, topk_excl(u, inE, 2)), 1)
        else:  # resid family
            keep = np.abs(w) if p["retain"] == "w" else np.abs(cred)
            E = np.argsort(-(keep + tie), axis=1)[:, :G.N_EXPLOIT]
            inE = np.zeros((n, D), bool); inE[row, E] = True
            ucb = np.abs(cred) + p["kappa"] * np.sqrt(np.log(t + 2) / (cnt + 1))
            X1 = topk_excl(ucb + tie, inE, 2); inE[row, X1] = True
            C = np.concatenate((E, X1, topk_excl(u, inE, 2)), 1)
        xv = xs[row, t, C]
        err = ys[:, t] - (xv * w[row, C]).sum(1)
        sq[:, t] = err ** 2
        wr[:, t] = ab[:, t][row, C].sum(1) / ab[:, t].sum(1)
        w[row, C] += p["lr"] / np.sqrt(K) * err[:, None] * xv
        if arm.startswith("kale"):
            pj = np.full((n, K), pin); pj[:, :G.N_EXPLOIT] = 1.0
            g = p.get("g", 1.0); Sb *= g; Nb *= g; Nb += 1
            Sb[row, C] += xv * ys[:, t, None] / pj
        elif arm == "signed_corr":
            cred *= .995; cred[row, C] += .2 * (xv * ys[:, t, None] - cred[row, C])
        elif arm == "resid":
            sig = err if p["credit"] == "resid" else ys[:, t]
            cred *= .995; cred[row, C] += .2 * (xv * sig[:, None] - cred[row, C]); cnt[row, C] += 1
    return sq.mean(1), wr


def kale_faithful(data, p):
    """Kale et al. 2017 realizable-setting estimator: k0 uniform reads -> unbiased b_hat, Sigma_hat (exact pair IPW),
    sparse estimate (Lasso on the estimates instead of the paper's Dantzig LP), top-KSUP support, predict with w_hat.
    Variants: p['mode'] in static / forget (g) / restart (R)."""
    xs, ys, ab = data
    n = len(xs); row = np.arange(n)[:, None]
    rng = np.random.default_rng(778)
    c1, c2 = D / K0, D * (D - 1) / (K0 * (K0 - 1))
    Sxx = np.zeros((n, D, D)); Sxy = np.zeros((n, D)); N = 0.0
    g = p.get("g", 1.0)
    w = np.zeros((n, D)); supp = np.zeros((n, KSUP), int); has = False
    sq = np.empty((n, T)); wr = np.empty((n, T))
    eye = np.eye(D)
    off = ~np.eye(K0, dtype=bool)
    for t in range(T):
        if p["mode"] == "restart" and t % p["R"] == 0:
            Sxx[:] = 0; Sxy[:] = 0; N = 0.0
        if t > 0 and t % SOLVE_EVERY == 0 and N > 0:
            Sig = Sxx / N; b = Sxy / N
            Sig = .5 * (Sig + Sig.transpose(0, 2, 1))
            ev, V = np.linalg.eigh(Sig)
            Sig = (V * np.clip(ev, 0, None)[:, None, :]) @ V.transpose(0, 2, 1) + p["mu"] * eye
            L = np.clip(ev, 0, None).max(1) + p["mu"]
            wh = np.zeros((n, D))
            for _ in range(100):
                z = wh - (np.einsum("nij,nj->ni", Sig, wh) - b) / L[:, None]
                wh = np.sign(z) * np.maximum(np.abs(z) - p["lam"] / L[:, None], 0)
            supp = np.argsort(-np.abs(wh), axis=1)[:, :KSUP]
            w = np.zeros((n, D)); w[row, supp] = wh[row, supp]; has = True
        u = rng.random((n, D))
        insup = np.zeros((n, D), bool)
        if has:
            insup[row, supp] = True
        U = topk_excl(u, insup, K0)                         # uniform k0 reads (estimator uses only these)
        filler = topk_excl(u, _mask(U, n), KSUP) if not has else supp
        C = np.concatenate((U, filler), 1)
        xv = xs[row, t, C]
        pred = (xs[row, t, supp] * w[row, supp]).sum(1) if has else np.zeros(n)
        err = ys[:, t] - pred
        sq[:, t] = err ** 2
        wr[:, t] = ab[:, t][row, C].sum(1) / ab[:, t].sum(1)
        # unbiased statistics from the k0 uniform reads only
        if p["mode"] == "forget":
            Sxx *= g; Sxy *= g; N = g * N
        xu = xs[row, t, U]
        out = c2 * xu[:, :, None] * xu[:, None, :]
        di = np.arange(K0)
        out[:, di, di] = c1 * xu ** 2
        Sxx[row[:, :, None], U[:, :, None], U[:, None, :]] += out
        Sxy[row, U] += c1 * xu * ys[:, t, None]
        N += 1
    return sq.mean(1), wr


def _mask(idx, n):
    m = np.zeros((n, D), bool); m[np.arange(n)[:, None], idx] = True; return m


def recovery(wr, theta):
    rec = []
    for b in range(1, T // TAU):
        seg = wr[:, b * TAU:(b + 1) * TAU]
        cs = np.cumsum(np.pad(seg, ((0, 0), (1, 0))), 1)
        ok = (cs[:, 20:] - cs[:, :-20]) / 20 >= theta
        rec.append(np.where(ok.any(1), ok.argmax(1), TAU))
    return np.stack(rec, 1)


boot = G.boot


def tune(arm, grid, val, runner):
    vs = [float(runner(val, arm, q)[0].mean()) for q in grid]
    return grid[int(np.argmin(vs))], min(vs)


def run_kale_f(data, arm, q):
    return kale_faithful(data, q)


def main():
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    res = {}
    t0 = time.time()
    for rho in (0.5, 0.0):
        val = streams(VAL, rho)
        chk = G.streams(VAL[:2], rho)
        assert np.allclose(chk[0], val[0][:2]) and np.allclose(chk[1], val[1][:2])
        arms = {}
        # winner with extended kappa
        arms["resid_ucb"] = tune("resid", [dict(lr=a, kappa=k, credit="resid", retain="w") for a in LRS for k in KAPPAS], val, simulate)
        kap = arms["resid_ucb"][0]["kappa"]
        arms["A1_credit_target"] = tune("resid", [dict(lr=a, kappa=kap, credit="target", retain="w") for a in LRS], val, simulate)
        arms["A2_retain_credit"] = tune("resid", [dict(lr=a, kappa=kap, credit="resid", retain="credit") for a in LRS], val, simulate)
        arms["A3_no_ucb"] = tune("resid", [dict(lr=a, kappa=0.0, credit="resid", retain="w") for a in LRS], val, simulate)
        for b in ("random", "kale_static", "kale_forget", "kale_restart", "signed_corr"):
            arms[b] = tune(b, G.GRIDS[b], val, simulate)
        print(rho, "simple arms tuned", round(time.time() - t0), "s", flush=True)
        kg = [dict(mode="static", lam=l, mu=m) for l in (.02, .05, .1, .2, .5, 1.0, 2.0) for m in (.1, .5, 2.0)]
        kg_f = [dict(mode="forget", lam=l, mu=m, g=g) for l in (.02, .05, .1, .2, .5, 1.0, 2.0) for m in (.1, .5, 2.0) for g in (.99, .995)]
        kg_r = [dict(mode="restart", lam=l, mu=m, R=r) for l in (.02, .05, .1, .2, .5, 1.0, 2.0) for m in (.1, .5, 2.0) for r in (240, 480)]
        for name, grid in (("kaleF_static", kg), ("kaleF_forget", kg_f), ("kaleF_restart", kg_r)):
            arms[name] = tune(name, grid, val, run_kale_f)
            print(rho, name, arms[name], round(time.time() - t0), "s", flush=True)
        # recovery threshold on validation (winner)
        _, wr_v = simulate(val, "resid", arms["resid_ucb"][0])
        theta = next((th for th in (0.9, 0.8, 0.7) if (recovery(wr_v, th) == TAU).mean() <= .25), 0.7)
        print(rho, "theta fixed on validation:", theta, flush=True)
        # test once
        test = streams(TEST, rho)
        out, mse = {}, {}
        for name, (q, vmse) in arms.items():
            m, wr = kale_faithful(test, q) if name.startswith("kaleF") else simulate(test, "resid" if name.startswith(("resid", "A")) else name, q)
            rc = recovery(wr, theta)
            mse[name] = m
            out[name] = dict(params=q, val_mse=vmse, test_mse=float(m.mean()), test_ci=boot(m),
                             wrecall=float(wr.mean()), recovery_median=float(np.median(rc)), unrecovered=float((rc == TAU).mean()))
            print(rho, name, q, "val", round(vmse, 4), "test", round(float(m.mean()), 4), "rec_med", out[name]["recovery_median"],
                  "unrec", round(out[name]["unrecovered"], 3), flush=True)
        pool = ("random", "kale_static", "kale_forget", "kale_restart", "signed_corr", "kaleF_static", "kaleF_forget", "kaleF_restart")
        strongest = min(pool, key=lambda a: out[a]["val_mse"])
        new, base = mse["resid_ucb"], mse[strongest]
        v = dict(strongest=strongest, ratio=float(new.mean() / base.mean()), diff_base_minus_new=float((base - new).mean()),
                 diff_ci=boot(base - new), rel_ci=boot((new - base) / base), theta=theta)
        abl = {}
        for a in ("A1_credit_target", "A2_retain_credit", "A3_no_ucb"):
            d = mse[a] - new; ci = boot(d)
            abl[a] = dict(mean=float(d.mean()), ci=ci, call="기여함" if ci[0] > 0 else ("해로움" if ci[1] < 0 else "구분 불가"))
        v["ablation"] = abl
        res[str(rho)] = dict(arms=out, verdict=v)
        print("VERDICT", rho, json.dumps(v, ensure_ascii=False), flush=True)
    v5, v0 = res["0.5"]["verdict"], res["0.0"]["verdict"]
    res["gate"] = dict(main_pass=v5["ratio"] <= .95 and v5["diff_ci"][0] > 0, control_pass=v0["rel_ci"][1] <= .02)
    res["gate"]["pass_"] = res["gate"]["main_pass"] and res["gate"]["control_pass"]
    print("GATE", res["gate"], flush=True)
    Path(__file__).with_name("gate33b.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
