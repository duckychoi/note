"""#33 gate 3 (PREREG_33.md "관문 3"): codex33 fixes. Fresh test seeds 5000-5095.
New family = |w|-retention + 4 exploration styles (representative picked on validation).
Baseline pool = random-LMS, tuned signed_corr, simple kale_*, IPW-Lasso Kale variant with corrected sampling.
Order in every round: choose C -> read x[C] -> predict -> reveal y -> update.   numpy only, single process.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import gate33 as G
import gate33b as GB

D, K, S, TAU, T = G.D, G.K, G.S, G.TAU, G.T
VAL, TEST = G.VAL, list(range(5000, 5096))
LRS = (.01, .025, .05, .1, .2)
NE, NX = 8, 4
K0, KSUP, EVERY = 8, 4, 32
topk_excl = GB.topk_excl


def mask(idx, n):
    m = np.zeros((n, D), bool); m[np.arange(n)[:, None], idx] = True; return m


def simulate(data, arm, p):
    xs, ys, ab = data
    n = len(xs); row = np.arange(n)[:, None]
    rng = np.random.default_rng(777)
    w = np.zeros((n, D)); cred = np.zeros((n, D)); cnt = np.zeros((n, D))
    Sb = np.zeros((n, D)); Nb = np.zeros((n, 1))
    tie = rng.uniform(0, 1e-9, (n, D))
    sq = np.empty((n, T)); wr = np.empty((n, T))
    pin = NX / (D - NE)
    for t in range(T):
        u = rng.random((n, D))
        if arm == "random":
            C = np.argsort(u, axis=1)[:, :K]
        elif arm.startswith("kale"):
            if arm == "kale_restart" and t % p["R"] == 0:
                Sb[:] = 0; Nb[:] = 0
            E = np.argsort(-(np.abs(Sb / np.maximum(Nb, 1e-12)) + tie), axis=1)[:, :NE]
            C = np.concatenate((E, topk_excl(u, mask(E, n), NX)), 1)
        elif arm == "signed_corr":
            E = np.argsort(-(np.abs(cred) + tie), axis=1)[:, :K - 2]
            C = np.concatenate((E, topk_excl(u, mask(E, n), 2)), 1)
        else:  # wret family
            E = np.argsort(-(np.abs(w) + tie), axis=1)[:, :NE]
            inE = mask(E, n)
            ex = p["explore"]
            if ex == "rand4":
                C = np.concatenate((E, topk_excl(u, inE, 4)), 1)
            else:
                if ex == "count2rand2":
                    sc = -cnt + u * 1e-3
                else:
                    sc = np.abs(cred) + p["kappa"] * np.sqrt(np.log(t + 2) / (cnt + 1)) + tie
                X1 = topk_excl(sc, inE, 2); inE[row, X1] = True
                C = np.concatenate((E, X1, topk_excl(u, inE, 2)), 1)
        xv = xs[row, t, C]                       # read
        pred = (xv * w[row, C]).sum(1)           # predict from read coords only
        y = ys[:, t]                             # reveal
        err = y - pred
        sq[:, t] = err ** 2
        wr[:, t] = ab[:, t][row, C].sum(1) / ab[:, t].sum(1)
        w[row, C] += p["lr"] / np.sqrt(K) * err[:, None] * xv
        cnt[row, C] += 1
        if arm.startswith("kale"):
            pj = np.full((n, K), pin); pj[:, :NE] = 1.0
            g = p.get("g", 1.0); Sb *= g; Nb *= g; Nb += 1
            Sb[row, C] += xv * y[:, None] / pj
        elif arm == "signed_corr":
            cred *= p["decay"]; cred[row, C] += p["eta"] * (xv * y[:, None] - cred[row, C])
        elif arm == "wret" and p["explore"].startswith("ucb"):
            sig = err if p["explore"] == "ucb_resid" else y
            cred *= .995; cred[row, C] += .2 * (xv * sig[:, None] - cred[row, C])
    return sq.mean(1), wr, sq


def kale_ipw_lasso(data, p):
    """Corrected sampling: 8 uniform from all 200 first; union with current support; fill to 12 at random.
    Statistics use only the 8 uniform draws (exact pair IPW). Restart clears stats, weights and support."""
    xs, ys, ab = data
    n = len(xs); row = np.arange(n)[:, None]
    rng = np.random.default_rng(778)
    c1, c2 = D / K0, D * (D - 1) / (K0 * (K0 - 1))
    Sxx = np.zeros((n, D, D)); Sxy = np.zeros((n, D)); N = 0.0
    g = p.get("g", 1.0); eye = np.eye(D); di = np.arange(K0)
    w = np.zeros((n, D)); supp = None
    sq = np.empty((n, T)); wr = np.empty((n, T))
    for t in range(T):
        if p["mode"] == "restart" and t % p["R"] == 0:
            Sxx[:] = 0; Sxy[:] = 0; N = 0.0; w[:] = 0; supp = None
        if t % EVERY == 0 and N > 0:
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
            w = np.zeros((n, D)); w[row, supp] = wh[row, supp]
        u = rng.random((n, D))
        U = np.argsort(u, axis=1)[:, :K0]                       # uniform 8 from all 200
        sel = mask(U, n)
        if supp is not None:
            sel[row, supp] = True
        need = K - sel.sum(1)
        fill_score = np.where(sel, -np.inf, rng.random((n, D)))
        order = np.argsort(-fill_score, axis=1)
        for i in range(n):
            sel[i, order[i, :need[i]]] = True
        C = np.argsort(~sel, axis=1, kind="stable")[:, :K]      # the 12 read coords
        xv = xs[row, t, C]
        pred = (xv * w[row, C]).sum(1)
        y = ys[:, t]
        err = y - pred
        sq[:, t] = err ** 2
        wr[:, t] = ab[:, t][row, C].sum(1) / ab[:, t].sum(1)
        if p["mode"] == "forget":
            Sxx *= g; Sxy *= g; N = g * N
        posU = np.stack([np.searchsorted(C[i], U[i]) for i in range(n)])   # U is a subset of sorted C
        xu = np.take_along_axis(xv, posU, 1)                               # reuse the values already read
        out = c2 * xu[:, :, None] * xu[:, None, :]
        out[:, di, di] = c1 * xu ** 2
        Sxx[row[:, :, None], U[:, :, None], U[:, None, :]] += out
        Sxy[row, U] += c1 * xu * y[:, None]
        N += 1
    return sq.mean(1), wr, sq


def recov(wr, th):
    return GB.recovery(wr, th)


def post_switch_mse(sq):
    return np.mean([sq[:, b * TAU:b * TAU + 100].mean(1) for b in range(1, T // TAU)], 0)


def boot(x, seed=98):
    return G.boot(x, seed)


def boot_ratio(new, base, seed=99):
    r = np.random.default_rng(seed); idx = r.integers(0, len(new), (5000, len(new)))
    q = 1 - new[idx].mean(1) / base[idx].mean(1)
    return [float(np.quantile(q, .025)), float(np.quantile(q, .975))]


def main():
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    t0 = time.time(); res = {}
    fam_grid = {
        "ucb_resid": [dict(lr=a, explore="ucb_resid", kappa=k) for a in LRS for k in (1, 2, 4, 8, 16, 32)],
        "ucb_target": [dict(lr=a, explore="ucb_target", kappa=k) for a in LRS for k in (1, 2, 4, 8, 16, 32)],
        "count2rand2": [dict(lr=a, explore="count2rand2") for a in LRS],
        "rand4": [dict(lr=a, explore="rand4") for a in LRS],
    }
    base_grid = {
        "random": [dict(lr=a) for a in (0.0, .001, .003, .01, .025, .05)],
        "signed_corr": [dict(lr=a, decay=dc, eta=e) for a in LRS for dc in (.98, .99, .995, .999) for e in (.05, .2, .5)],
        "kale_static": G.GRIDS["kale_static"], "kale_forget": G.GRIDS["kale_forget"], "kale_restart": G.GRIDS["kale_restart"],
    }
    kl = [dict(lam=l, mu=m) for l in (.2, .5, 1.0, 2.0) for m in (.5, 2.0)]
    kgrid = {"kaleIPWLasso_static": [dict(mode="static", **q) for q in kl],
             "kaleIPWLasso_forget": [dict(mode="forget", g=.995, **q) for q in kl],
             "kaleIPWLasso_restart": [dict(mode="restart", R=480, **q) for q in kl]}
    for rho in (0.5, 0.0):
        val = GB.streams(VAL, rho)
        tuned = {}
        for name, grid in list(fam_grid.items()) + list(base_grid.items()):
            arm = "wret" if name in fam_grid else name
            vs = [float(simulate(val, arm, q)[0].mean()) for q in grid]
            tuned[name] = (arm, grid[int(np.argmin(vs))], min(vs))
        for name, grid in kgrid.items():
            vs = [float(kale_ipw_lasso(val, q)[0].mean()) for q in grid]
            tuned[name] = ("kipw", grid[int(np.argmin(vs))], min(vs))
        for k, v in tuned.items():
            print(rho, "val", k, v[1], round(v[2], 4), flush=True)
        rep = min(fam_grid, key=lambda k: tuned[k][2])
        pool = list(base_grid) + list(kgrid)
        strongest = min(pool, key=lambda k: tuned[k][2])
        _, wr_v, _ = simulate(val, "wret", tuned[rep][1])
        theta_unrec = {str(th): float((recov(wr_v, th) == TAU).mean()) for th in (.9, .8, .7)}
        print(rho, "representative", rep, "strongest", strongest, "val unrecovered by theta", theta_unrec, round(time.time() - t0), "s", flush=True)
        test = GB.streams(TEST, rho)
        out, mse = {}, {}
        for name, (arm, q, vm) in tuned.items():
            m, wr, sq = kale_ipw_lasso(test, q) if arm == "kipw" else simulate(test, arm, q)
            mse[name] = m
            rc = recov(wr, .8)
            out[name] = dict(params=q, val_mse=vm, test_mse=float(m.mean()), test_ci=boot(m), wrecall=float(wr.mean()),
                             post_switch100_mse=float(post_switch_mse(sq).mean()),
                             recovery08_median=float(np.median(rc)), unrecovered08=float((rc == TAU).mean()),
                             per_seed_mse=m.tolist())
            print(rho, name, "test", round(float(m.mean()), 4), "post100", round(out[name]["post_switch100_mse"], 3), flush=True)
        new, base = mse[rep], mse[strongest]
        v = dict(representative=rep, strongest=strongest, val_unrecovered_by_theta=theta_unrec,
                 five_pct_margin_ci=boot(0.95 * base - new), improvement=float(1 - new.mean() / base.mean()),
                 improvement_ci=boot_ratio(new, base), rel_degradation_ci=boot((new - base) / base))
        res[str(rho)] = dict(arms=out, verdict=v)
        print("VERDICT", rho, json.dumps(v), flush=True)
    v5, v0 = res["0.5"]["verdict"], res["0.0"]["verdict"]
    res["gate"] = dict(main_pass=v5["five_pct_margin_ci"][0] > 0, control_pass=v0["rel_degradation_ci"][1] <= .02)
    res["gate"]["pass_"] = res["gate"]["main_pass"] and res["gate"]["control_pass"]
    print("GATE", res["gate"], flush=True)
    Path(__file__).with_name("gate33c.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
