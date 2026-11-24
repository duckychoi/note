"""
#35 audit - does SMC ("PF flow") beat ODE solvers at equal NFE in the original toy, and WHY?

Same target as the original toy (note:pf-generative-flow src/toy_pf_vs_euler.py): 2-D GMM, weights [.6,.3,.1],
means (-2.5,0),(2.5,0),(0,3.2), std 0.5, linear interpolant from N(0,I). Exact velocity/score/density from gmm_flow.

Changes vs the original toy (audit points):
  A. NFE is counted honestly: every velocity OR score evaluation per particle = 1 NFE (rejuvenation included).
     Density evaluations p_t(x) used by the SMC weights are an ORACLE (a trained flow has no cheap p_t); counted 0
     here, i.e. SMC is given an advantage.
  B. Stronger baselines at equal NFE: Heun (2nd order), predictor-corrector (Euler + Langevin corrector, Song 2021).
  C. Oracle-dependence test: SMC with the oracle weights but WITHOUT the flow velocity (pure random-walk proposal).
     If it is still good, the gain comes from the oracle density, not from "replacing the ODE".
  D. 10 seeds, mean +- sd.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gmm_flow import GMM

G = GMM([0.6, 0.3, 0.1], [[-2.5, 0.0], [2.5, 0.0], [0.0, 3.2]], 0.5)
K = 1500


def euler(nfe, rng):
    N = nfe; ts = np.linspace(0, 1, N + 1); X = rng.standard_normal((K, 2))
    for k in range(N):
        X = X + (ts[k + 1] - ts[k]) * G.velocity(X, ts[k])
    return X, nfe


def heun(nfe, rng):
    N = nfe // 2; ts = np.linspace(0, 1, N + 1); X = rng.standard_normal((K, 2))
    for k in range(N):
        h = ts[k + 1] - ts[k]; v0 = G.velocity(X, ts[k]); Xe = X + h * v0
        X = X + 0.5 * h * (v0 + G.velocity(Xe, ts[k + 1]))
    return X, 2 * N


def pc(nfe, rng, L=1, r=0.1):
    """Euler predictor + L Langevin corrector steps on p_{t_{k+1}} (score), step eps = r * min_k V_k(t)."""
    N = nfe // (1 + L); ts = np.linspace(0, 1, N + 1); X = rng.standard_normal((K, 2))
    for k in range(N):
        t1 = ts[k + 1]; X = X + (t1 - ts[k]) * G.velocity(X, ts[k])
        eps = r * float(np.min(t1**2 * G.s**2 + (1 - t1) ** 2))
        for _ in range(L):
            X = X + eps * G.score(X, t1) + np.sqrt(2 * eps) * rng.standard_normal(X.shape)
    return X, N * (1 + L)


def smc(N, rng, tau=0.15, rejuv=0, eps=0.012, use_velocity=True):
    """Original toy's SMC: bootstrap weights p_{t1}(x)/q(x) with ORACLE density, ESS<K/2 resample, ULA rejuvenation."""
    ts = np.linspace(0, 1, N + 1); h = 1.0 / N; X = rng.standard_normal((K, 2)); nfe = 0
    for k in range(N):
        t0, t1 = ts[k], ts[k + 1]
        drift = G.velocity(X, t0) if use_velocity else np.zeros_like(X)
        nfe += 1 if use_velocity else 0
        mean = X + h * drift
        X = mean + tau * np.sqrt(h) * rng.standard_normal((K, 2))
        var = tau**2 * h
        d2 = ((X[:, None, :] - mean[None, :, :]) ** 2).sum(-1)
        lq = np.log(np.exp(-0.5 * d2 / var).mean(1) + 1e-300) - np.log(2 * np.pi * var)
        lw = G.logpdf_t(X, t1) - lq; w = np.exp(lw - lw.max()); w /= w.sum()
        if 1.0 / (w**2).sum() < 0.5 * K:
            X = X[rng.choice(K, K, p=w)]
            for _ in range(rejuv):
                X = X + eps * G.score(X, t1) + np.sqrt(2 * eps) * rng.standard_normal((K, 2)); nfe += 1
    return X, nfe


def energy(A, B):
    def pd(X, Y):
        return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) + 1e-12).mean()
    return 2 * pd(A, B) - pd(A, A) - pd(B, B)


def mode_mass(X):
    m = np.array([(((X - mu) ** 2).sum(1) < 1.2**2).mean() for mu in G.mu])
    return m / max(m.sum(), 1e-12)


def main():
    seeds = range(10); out = {}
    budgets = [4, 6, 8, 12, 20]
    methods = {
        "euler": lambda n, r: euler(n, r),
        "heun": lambda n, r: heun(n, r),
        "pc_L1": lambda n, r: pc(n, r, L=1),
        "smc_oracle": lambda n, r: smc(n, r),
        "smc_oracle+rej4": lambda n, r: smc(n, r, rejuv=4),
        "smc_oracle_NO_velocity": lambda n, r: smc(n, r, use_velocity=False),
    }
    print(f"{'method':24s} " + " ".join(f"{'N=' + str(b):>22s}" for b in budgets))
    print(f"{'':24s} " + " ".join(f"{'ED mean±sd (NFE)':>22s}" for _ in budgets))
    for name, fn in methods.items():
        row = []
        for b in budgets:
            eds, nfes, mms = [], [], []
            for sd in seeds:
                rng = np.random.default_rng(1000 + sd)
                X, nfe = fn(b, rng)
                T = G.sample(K, np.random.default_rng(5000 + sd))
                eds.append(energy(X, T)); nfes.append(nfe); mms.append(mode_mass(X))
            out[f"{name}/N{b}"] = dict(ED_mean=float(np.mean(eds)), ED_sd=float(np.std(eds)), NFE_mean=float(np.mean(nfes)),
                                      mode_mass_mean=np.mean(mms, 0).round(3).tolist())
            row.append(f"{np.mean(eds):.4f}±{np.std(eds):.4f}({np.mean(nfes):4.1f})")
        print(f"{name:24s} " + " ".join(f"{c:>22s}" for c in row), flush=True)
    # reference: exact samples vs exact samples (floor of the metric)
    fl = [energy(G.sample(K, np.random.default_rng(s)), G.sample(K, np.random.default_rng(5000 + s))) for s in seeds]
    out["metric_floor_exact_vs_exact"] = dict(ED_mean=float(np.mean(fl)), ED_sd=float(np.std(fl)))
    print(f"metric floor (exact vs exact samples): {np.mean(fl):.4f}±{np.std(fl):.4f}")
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "e35_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
