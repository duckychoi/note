"""
#35 pivot gate: even with an EXACT energy (oracle log p_t = best case for an energy-parameterised diffusion model),
does SMC keep a usable ESS and beat Heun at equal NFE as the dimension grows?

Target: GMM in R^d, 3 modes at (-2.5,0,..),(2.5,0,..),(0,3.2,..), weights [.6,.3,.1], std 0.5. Linear interpolant.
SMC = standard SMC sampler on the path p_{t_k}:
  forward kernel  q(x'|x)  = N(x + h[v + g^2/2 s](x,t), g^2 h)     (reverse-SDE Euler-Maruyama, marginal-preserving in the limit)
  backward kernel L(x|x')  = N(x' - h[v - g^2/2 s](x',t'), g^2 h)  (time-reversal EM step)
  weight          w = p_{t'}(x') L(x|x') / (p_t(x) q(x'|x))         (oracle log p_t;  Z_t cancels -> energy suffices)
Score from velocity for the linear interpolant: s = -(x - t u)/(1-t)  (no extra NFE). v(x',t') is reused as the next
forward drift, so SMC costs 1 NFE/step (+1 at the end), same as Euler.  Resample when ESS < K/2.
Metrics: mode-mass L1 error (exact weights known), per-dim variance error, mean ESS/K, 5 seeds.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gmm_flow import GMM

K = 2000; W = np.array([0.6, 0.3, 0.1])


def make(d):
    mu = np.zeros((3, d)); mu[0, 0] = -2.5; mu[1, 0] = 2.5; mu[2, 1] = 3.2
    return GMM(W, mu, 0.5)


def vs(G, X, t):
    u = G.velocity(X, t); return u, -(X - t * u) / (1 - t)


def heun(G, nfe, rng, d, t_end=0.999):
    N = nfe // 2; ts = np.linspace(0, t_end, N + 1); X = rng.standard_normal((K, d))
    for k in range(N):
        h = ts[k + 1] - ts[k]; v0 = G.velocity(X, ts[k]); Xe = X + h * v0
        X = X + 0.5 * h * (v0 + G.velocity(Xe, ts[k + 1]))
    return X, None


def euler(G, nfe, rng, d, t_end=0.999):
    ts = np.linspace(0, t_end, nfe + 1); X = rng.standard_normal((K, d))
    for k in range(nfe):
        X = X + (ts[k + 1] - ts[k]) * G.velocity(X, ts[k])
    return X, None


def lnorm(D, var):
    return -0.5 * (D**2).sum(-1) / var - 0.5 * D.shape[-1] * np.log(2 * np.pi * var)


def smc(G, nfe, rng, d, g=0.5, t_end=0.999, weighted=True):
    N = nfe - 1; ts = np.linspace(0, t_end, N + 1); X = rng.standard_normal((K, d)); ess = []
    u, s = vs(G, X, ts[0]); lw = np.zeros(K)
    for k in range(N):
        t0, t1 = ts[k], ts[k + 1]; h = t1 - t0; var = g**2 * h
        mf = X + h * (u + 0.5 * g**2 * s)
        X1 = mf + np.sqrt(var) * rng.standard_normal((K, d))
        u1, s1 = vs(G, X1, t1)
        mb = X1 - h * (u1 - 0.5 * g**2 * s1)
        if weighted:
            lw = lw + G.logpdf_t(X1, t1) + lnorm(X - mb, var) - G.logpdf_t(X, t0) - lnorm(X1 - mf, var)
            w = np.exp(lw - lw.max()); w /= w.sum(); e = 1 / (w**2).sum(); ess.append(e / K)
            if e < K / 2:
                idx = rng.choice(K, K, p=w); X1, u1, s1 = X1[idx], u1[idx], s1[idx]; lw = np.zeros(K)
        X, u, s = X1, u1, s1
    if weighted:  # final weighted -> resample once for equal-weight output
        w = np.exp(lw - lw.max()); w /= w.sum(); X = X[rng.choice(K, K, p=w)]
    return X, (float(np.mean(ess)) if ess else None, float(np.min(ess)) if ess else None)


def metrics(G, X):
    D = ((X[:, None, :] - G.mu[None]) ** 2).sum(-1); lab = D.argmin(1)
    mm = np.bincount(lab, minlength=3) / len(X)
    var_err = 0.0
    if X.shape[1] > 2:
        var_err = float(np.abs(X[:, 2:].var(0) - 0.25).mean() / 0.25)   # exact per-dim var off-mode-plane = 0.25
    return float(np.abs(mm - W).sum()), var_err


def main():
    out = {}
    print(f"{'d':>4s} {'NFE':>4s} {'method':10s} {'modeL1':>14s} {'varRelErr':>14s} {'ESS/K mean,min':>16s}")
    for d in [2, 8, 32, 128]:
        G = make(d)
        for nfe in [6, 10, 20]:
            for name, fn in [("euler", euler), ("heun", heun), ("sde_noW", lambda G, n, r, d: smc(G, n, r, d, weighted=False)),
                             ("smc", smc)]:
                ml, ve, es = [], [], []
                for sd in range(5):
                    X, e = fn(G, nfe, np.random.default_rng(100 * d + sd), d)
                    a, b = metrics(G, X); ml.append(a); ve.append(b); es.append(e)
                ess_s = ""
                if es[0] is not None and es[0][0] is not None:
                    ess_s = f"{np.mean([x[0] for x in es]):.2f},{np.min([x[1] for x in es]):.3f}"
                out[f"d{d}/nfe{nfe}/{name}"] = dict(modeL1=float(np.mean(ml)), modeL1_sd=float(np.std(ml)),
                                                  varRelErr=float(np.mean(ve)), ess=ess_s)
                print(f"{d:4d} {nfe:4d} {name:10s} {np.mean(ml):7.3f}±{np.std(ml):.3f} {np.mean(ve):7.3f}±{np.std(ve):.3f} {ess_s:>16s}", flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e35_dim.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
