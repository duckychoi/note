"""
#35 policy candidate A - GATE experiment (11-22, after codex + GPT review).

Question: with an IMPERFECT critic, at the SAME measured cost per action, does intermediate resampling (FK / TDS)
choose actions with higher TRUE value than terminal selection (SIR / argmax) or guidance alone?

Fairness fixes demanded by the reviews:
  * every method uses the SAME stochastic path (SDE with g^2 = 2(1-t)/t, exact start at T0) -> only the selection differs
  * cost is counted, not nominal: velocity calls (cached: one per particle-step for every method), critic forward
    calls, critic backward calls (gradient = 1 fwd + 1 bwd, bwd counted x2).  cost = v + q_fwd + 2 q_bwd, with the
    critic assumed as expensive as the velocity net (stated assumption)
  * argmax terminal selection is a baseline
  * configuration (steps n, temperature beta, final rule) is tuned on VALIDATION critics/seeds, reported on
    separate EVALUATION critics/seeds (mean over seeds, 95% t-interval)
  * primary metric = mean TRUE Q of the chosen action (control-quality proxy); mode-mass L1 is secondary

Toy: behaviour policy = 3-mode GMM in R^8 (exact flow), true Q(a) = -0.15||a - c||^2 with c at the 10% mode.
Critic Q_hat = Q + error field: 12 Gaussian bumps (width 0.8) centred at behaviour samples + N(0, 1.5^2) jitter,
heights sigma * N(0,1); sigma in {0, 1, 3}.  Gradient of Q_hat is analytic (stands in for backprop);
guidance uses grad Q_hat(a1_hat) with the velocity Jacobian detached (DPS-style approximation, no velocity backward).
Results -> results/v2/e35p_gate.json
"""
import os, sys, json, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
import e35p_twist as E

T0, T_END, d = E.T0, E.T_END, 8
P = E.setup(d, lam=0.15, beta=1.0); G = P["G"]


def make_critic(sigma, seed):
    r = np.random.default_rng(seed); n = 12
    loc = G.sample(n, r) + 1.5 * r.standard_normal((n, d)); h = sigma * r.standard_normal(n); w2 = 0.8 ** 2
    def q(A):
        D2 = ((A[:, None, :] - loc[None]) ** 2).sum(-1)
        return E.Q(P, A) + (h[None] * np.exp(-0.5 * D2 / w2)).sum(1)
    def gq(A):
        diff = A[:, None, :] - loc[None]; e = h[None] * np.exp(-0.5 * (diff ** 2).sum(-1) / w2)
        return E.gradQ(P, A) - (e[..., None] * diff).sum(1) / w2
    return q, gq


def run(method, n, K, beta, final, q, gq, Dn, rng):
    """returns chosen actions (Dn,d) and measured cost per decision"""
    resample = method in ("fk", "tds"); guide = method in ("guid", "tds")
    cv = cq = cb = 0
    A = T0 * G.sample(Dn * K, rng) + (1 - T0) * rng.standard_normal((Dn * K, d)); ts = np.linspace(T0, T_END, n + 1)
    u = G.velocity(A, T0); cv += 1
    if resample:
        phi = q(A + (1 - T0) * u) / beta; cq += 1; lw = phi.reshape(Dn, K).copy()
    qa = None
    for k in range(n):
        t0, t1 = ts[k], ts[k + 1]; h = t1 - t0; var = E.g2(t0) * h
        mp = A + h * (u + 0.5 * E.g2(t0) * (-(A - t0 * u) / (1 - t0)))
        mq = mp
        if guide:
            mq = mp + var * gq(A + (1 - t0) * u) / beta; cq += 1; cb += 1
        A1 = mq + np.sqrt(var) * rng.standard_normal(A.shape)
        last = k == n - 1
        if not last:
            u1 = G.velocity(A1, t1); cv += 1
        if resample:
            if last:
                qa = q(A1); phi1 = qa / beta; cq += 1
            else:
                phi1 = q(A1 + (1 - t1) * u1) / beta; cq += 1
            inc = phi1 - phi
            if guide:
                inc = inc - 0.5 * ((A1 - mp) ** 2).sum(-1) / var + 0.5 * ((A1 - mq) ** 2).sum(-1) / var
            lw = lw + inc.reshape(Dn, K); phi = phi1
        A = A1
        if not last:
            u = u1
        if resample and K > 1 and not last:
            w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True); rs = 1 / (w ** 2).sum(1) < K / 2
            if rs.any():
                A3, U3, P3 = A.reshape(Dn, K, d), u.reshape(Dn, K, d), phi.reshape(Dn, K)
                for i in np.where(rs)[0]:
                    j = rng.choice(K, K, p=w[i]); A3[i], U3[i], P3[i] = A3[i][j], U3[i][j], P3[i][j]; lw[i] = 0
                A, u, phi = A3.reshape(-1, d), U3.reshape(-1, d), P3.reshape(-1)
    A = A.reshape(Dn, K, d)
    if K == 1:
        return A[:, 0], (cv + cq + 2 * cb) * K
    if qa is None:
        qa = q(A.reshape(-1, d)); cq += 1
    qa = qa.reshape(Dn, K)
    if final == "argmax":
        idx = qa.argmax(1)
    else:
        lg = lw if resample else qa / beta
        w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        idx = (w.cumsum(1) < rng.random((Dn, 1))).sum(1).clip(max=K - 1)
    return A[np.arange(Dn), idx], (cv + cq + 2 * cb) * K


def per_particle(method, n):
    resample = method in ("fk", "tds"); guide = method in ("guid", "tds")
    return n + (n + 1 if resample else 1) + (3 * n if guide else 0)    # v + q_fwd (+ q_fwd + 2 q_bwd per guided step)


METHODS = ["end", "guid", "fk", "tds"]; NS = [4, 8, 16, 32]; BETAS = [0.25, 1.0, 4.0]; FINALS = ["sir", "argmax"]


def configs(method, C):
    for n in NS:
        K = C // per_particle(method, n)
        if K < (2 if method in ("fk", "tds", "end") else 1):
            continue
        for beta, final in itertools.product(BETAS, FINALS):
            if method == "end" and final == "argmax" and beta != BETAS[0]:
                continue                                           # beta irrelevant
            yield n, int(K), beta, final


def score(X):
    return float(E.Q(P, X).mean())


def main():
    out = {}; ref = score(G.sample(200_000, np.random.default_rng(0)))
    tgt = E.evaluate(P, G.sample(10, np.random.default_rng(0)))     # warm
    print(f"behaviour policy true E[Q] = {ref:.3f}; exact tilted target (beta=1, true Q) masses {np.round(P['tw'],3)}", flush=True)
    for sigma in [0.0, 1.0, 3.0]:
        for C in [64, 256, 1024]:
            line = f"sigma={sigma:3.1f} cost={C:5d}"
            for m in METHODS:
                best = None
                for cfg in configs(m, C):           # validation: critics 100,101, 600 decisions each
                    v = np.mean([score(run(m, *cfg, *make_critic(sigma, 100 + s), 600, np.random.default_rng(10_000 + s))[0]) for s in range(2)])
                    if best is None or v > best[0]:
                        best = (v, cfg)
                vals, costs = [], []
                for s in range(5):                  # evaluation: critics 0..4, fresh sampling seeds
                    X, c = run(m, *best[1], *make_critic(sigma, s), 1000, np.random.default_rng(s)); vals.append(score(X)); costs.append(c)
                mu = float(np.mean(vals)); ci = float(stats.t.ppf(0.975, 4) * np.std(vals, ddof=1) / np.sqrt(5))
                out[f"s{sigma}/C{C}/{m}"] = dict(trueQ=mu, ci95=ci, cfg=dict(zip(["n", "K", "beta", "final"], best[1])), cost=float(np.mean(costs)), vals=vals)
                line += f" | {m} {mu:6.3f}±{ci:.3f} ({best[1][0]}x{best[1][1]},{best[1][3]},b{best[1][2]},c{int(np.mean(costs))})"
            print(line, flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e35p_gate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
