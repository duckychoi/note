"""
#35 generative version - can theory rescue it?  (exact energy given; best case for an energy-parameterised model)

Theory (standard SMC / importance-sampling argument):
  per-step incremental log-weight = log [p_{t'}(x') L(x|x')] - log [p_t(x) q(x'|x)].
  If the kernel means have local error e (per dim) and kernel std g*sqrt(h), then Var(log w_step) ~ d * e^2 / (g^2 h).
  Euler pair: e = O(h^2)  -> total Var(log w) ~ d * h^2 / g^2 * N h ... = O(d h^2 / g^2)   (N = 1/h steps)
  2nd-order pair: e = O(h^3) -> O(d h^4 / g^2).
  SMC needs K ~ exp(Var log w) particles for a fixed accuracy (Bengtsson-Bickel-Li / Agapiou et al.).
  => (i) at fixed NFE SMC is CONSISTENT in K (error -> 0 as K -> inf) while an ODE solver is biased: there is a crossover K*;
     (ii) K* grows exponentially in d * h^{2p} / g^2: higher-order kernels (p=2) shrink the exponent.
Tests: (A) K-scaling at fixed NFE (Euler-pair SMC vs Heun), (B) 2nd-order kernel pair (Heun mean forward & backward),
       (C) measured Var(log w) vs d and order.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gmm_flow import GMM

W = np.array([0.6, 0.3, 0.1])


def make(d):
    mu = np.zeros((3, d)); mu[0, 0] = -2.5; mu[1, 0] = 2.5; mu[2, 1] = 3.2
    return GMM(W, mu, 0.5)


def f_fw(G, X, t, g):   # reverse-SDE (generation) drift
    u = G.velocity(X, t); s = -(X - t * u) / (1 - t); return u + 0.5 * g**2 * s, u, s


def lnorm(D, var):
    return -0.5 * (D**2).sum(-1) / var


def heun(G, nfe, rng, d, K, t_end=0.999):
    N = nfe // 2; ts = np.linspace(0, t_end, N + 1); X = rng.standard_normal((K, d))
    for k in range(N):
        h = ts[k + 1] - ts[k]; v0 = G.velocity(X, ts[k]); Xe = X + h * v0
        X = X + 0.5 * h * (v0 + G.velocity(Xe, ts[k + 1]))
    return X, {}


def smc(G, nfe, rng, d, K, g=0.5, order=1, t_end=0.999):
    """order 1: Euler forward / Euler time-reversal backward (1 NFE/step).
       order 2: Heun forward mean (2 NFE) + Heun backward mean (1 extra NFE)  -> 3 NFE/step."""
    per = 1 if order == 1 else 3
    N = max(1, (nfe - 1) // per); ts = np.linspace(0, t_end, N + 1); X = rng.standard_normal((K, d))
    lw = np.zeros(K); ess = []; varlw = 0.0
    for k in range(N):
        t0, t1 = ts[k], ts[k + 1]; h = t1 - t0; var = g**2 * h
        f0, _, _ = f_fw(G, X, t0, g)
        if order == 1:
            mf = X + h * f0
        else:
            fp, _, _ = f_fw(G, X + h * f0, t1, g); mf = X + 0.5 * h * (f0 + fp)
        X1 = mf + np.sqrt(var) * rng.standard_normal((K, d))
        u1 = G.velocity(X1, t1); s1 = -(X1 - t1 * u1) / (1 - t1); fb1 = u1 - 0.5 * g**2 * s1   # forward-noising drift (time-reversed)
        if order == 1:
            mb = X1 - h * fb1
        else:
            Xb = X1 - h * fb1; ub = G.velocity(Xb, t0); sb = -(Xb - t0 * ub) / (1 - t0) if t0 < 1 else 0 * ub
            mb = X1 - 0.5 * h * (fb1 + (ub - 0.5 * g**2 * sb))
        inc = G.logpdf_t(X1, t1) + lnorm(X - mb, var) - G.logpdf_t(X, t0) - lnorm(X1 - mf, var)
        varlw += inc.var()
        lw = lw + inc; w = np.exp(lw - lw.max()); w /= w.sum(); e = 1 / (w**2).sum(); ess.append(e / K)
        if e < K / 2:
            X1 = X1[rng.choice(K, K, p=w)]; lw = np.zeros(K)
        X = X1
    w = np.exp(lw - lw.max()); w /= w.sum(); X = X[rng.choice(K, K, p=w)]
    return X, dict(ess=float(np.mean(ess)), varlogw=float(varlw), nfe=N * per + 1)


def metrics(G, X):
    D = ((X[:, None, :] - G.mu[None]) ** 2).sum(-1); lab = D.argmin(1)
    mm = np.bincount(lab, minlength=3) / len(X)
    return float(np.abs(mm - W).sum())


def run(name, fn, G, nfe, d, K, seeds=5):
    ml, info = [], []
    for sd in range(seeds):
        X, inf = fn(G, nfe, np.random.default_rng(7 * d + sd + 1000 * K), d, K); ml.append(metrics(G, X)); info.append(inf)
    r = dict(modeL1=float(np.mean(ml)), sd=float(np.std(ml)))
    if info[0]:
        r.update(ess=float(np.mean([i["ess"] for i in info])), varlogw=float(np.mean([i["varlogw"] for i in info])), nfe=info[0]["nfe"])
    return r


def main():
    out = {}
    print("(A) K-scaling at NFE=10 (mode-mass L1, lower better)")
    for d in [2, 8, 32]:
        G = make(d)
        for K in [500, 2000, 8000]:
            a = run("heun", heun, G, 10, d, K); b = run("smc1", smc, G, 10, d, K)
            out[f"A/d{d}/K{K}"] = dict(heun=a, smc1=b)
            print(f" d={d:3d} K={K:5d}  heun {a['modeL1']:.3f}±{a['sd']:.3f}  smc1 {b['modeL1']:.3f}±{b['sd']:.3f} ESS {b['ess']:.2f} Var(logw) {b['varlogw']:.2f}", flush=True)
    print("(B,C) order & noise at fixed NFE=19/20, K=2000")
    for d in [2, 8, 32, 128]:
        G = make(d); a = run("heun", heun, G, 20, d, 2000)
        line = f" d={d:3d} heun(20) {a['modeL1']:.3f}"
        out[f"B/d{d}/heun"] = a
        for g in [0.25, 0.5, 1.0]:
            for order in [1, 2]:
                b = run("smc", lambda G, n, r, d, K: smc(G, n, r, d, K, g=g, order=order), G, 19 if order == 2 else 20, d, 2000)
                out[f"B/d{d}/g{g}/o{order}"] = b
                line += f" | g{g} o{order}: {b['modeL1']:.3f} ESS{b['ess']:.2f} V{b['varlogw']:.1f}"
        print(line, flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e35_theory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
