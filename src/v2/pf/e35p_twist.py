"""
#35 policy version, candidate A - toy gate (no robot): sample a FM policy tilted by a value, pi*(a) ∝ pi_FM(a) exp(Q(a)/beta).

Behaviour policy pi_FM: GMM in R^d (weights [.6,.3,.1], modes (-2.5,0..),(2.5,0..),(0,3.2,..), std .5), exact flow (gmm_flow).
Q(a) = -lam ||a - c||^2 with c at the MINORITY mode -> target shifts mass to the rare mode (the hard case for best-of-N).
Everything is closed form: exact target pi* (GMM), exact twist Q_t(a_t) = beta log E[exp(Q(a1)/beta) | a_t].

Methods (budget B = network calls per chosen action; value-gradient counted as 1 extra call):
  bon        : K = B/n ODE (Euler) samples, pick one by softmax(Q/beta)        (IDQL / V-GPS style, terminal potential)
  guid_heur  : 1 particle, SDE drift + g^2 grad Q(a1_hat)/beta  (DPS/QPILOTS-like guidance, no weights), n = B/2
  fk_heur    : SMC, potential Q(a1_hat(a_t)) differences (FK steering), K = B/n
  tds_heur   : SMC + guided proposal + heuristic twist (TDS), exact terminal correction, K = B/(2n)
  guid_exact / tds_exact : same with the exact twist Q_t (upper bound of "learned Q_t", i.e. candidate A's ceiling)
Metrics over D independent decisions: mode-mass L1 vs exact target masses, |E[Q] - E*[Q]|.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gmm_flow import GMM

D = 1500; G_NOISE = 1.0; T_END = 0.999


def setup(d, lam, beta):
    mu = np.zeros((3, d)); mu[0, 0] = -2.5; mu[1, 0] = 2.5; mu[2, 1] = 3.2
    G = GMM([0.6, 0.3, 0.1], mu, 0.5); c = mu[2].copy(); kap = lam / beta; s2 = 0.25
    # exact tilted target: component-wise product with exp(-kap||a-c||^2)
    prec = 1 / s2 + 2 * kap; m = (mu / s2 + 2 * kap * c) / prec
    lw = np.log(G.w) - 0.5 * d * np.log(1 + 2 * kap * s2) - kap * ((mu - c) ** 2).sum(1) / (1 + 2 * kap * s2)
    wt = np.exp(lw - lw.max()); wt /= wt.sum()
    return dict(G=G, c=c, kap=kap, lam=lam, beta=beta, tw=wt, tm=m, ts=np.sqrt(1 / prec), d=d)


def Q(P, A):
    q = -P["lam"] * ((A - P["c"]) ** 2).sum(-1)
    for (loc, H, w) in P.get("bumps", []):        # spurious OOD over-estimation (candidate B); empty -> true Q
        q = q + H * np.exp(-0.5 * ((A - loc) ** 2).sum(-1) / w**2)
    return q


def gradQ(P, A):
    return -2 * P["lam"] * (A - P["c"])


def Qt_exact(P, A, t):
    """beta * log E[exp(Q(a1)/beta) | a_t] for the GMM (closed form)."""
    G = P["G"]; kap = P["kap"]; d = P["d"]; s2 = G.s ** 2
    if t <= 0:
        t = 1e-6
    V = t**2 * s2 + (1 - t) ** 2
    lr = G._resp(A, t)[0]
    lg = np.log(np.clip(lr, 1e-300, None))                         # (n,K) posterior component probs
    mk = G.mu[None] + (t * s2 / V)[None, :, None] * (A[:, None, :] - t * G.mu[None])   # (n,K,d)
    ck = s2 * (1 - t) ** 2 / V                                       # (K,)
    term = -0.5 * d * np.log(1 + 2 * kap * ck)[None] - kap * ((mk - P["c"]) ** 2).sum(-1) / (1 + 2 * kap * ck)[None]
    z = lg + term; mx = z.max(1, keepdims=True)
    return P["beta"] * (mx[:, 0] + np.log(np.exp(z - mx).sum(1)))


def grad_num(f, A, eps=1e-4):
    g = np.zeros_like(A)
    for j in range(A.shape[1]):
        E = np.zeros(A.shape[1]); E[j] = eps
        g[:, j] = (f(A + E) - f(A - E)) / (2 * eps)
    return g


def a1hat(P, A, t):
    return A + (1 - t) * P["G"].velocity(A, t)


def twist(P, A, t, kind):
    if kind == "exact":
        return Qt_exact(P, A, t)
    return Q(P, a1hat(P, A, t))                                     # Tweedie heuristic (FK steering / DPS)


def drift(P, A, t):
    u = P["G"].velocity(A, t); s = -(A - t * u) / (1 - t)
    return u + 0.5 * g2(t) * s


def g2(t):
    """noise schedule whose two-time law matches the linear interpolant (x_t|x_1 = N(t x_1,(1-t)^2)):
    g^2 = 2 sigma^2 (alpha'/alpha - sigma'/sigma) = 2(1-t)/t  -> E[.|a_t] under the SDE = interpolant posterior"""
    return 2 * (1 - t) / t


T0 = 0.02


def pick(lw, K, rng):
    """lw (D,K) -> index per decision sampled from softmax"""
    w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    u = rng.random((w.shape[0], 1)); return (w.cumsum(1) < u).sum(1).clip(max=K - 1)


def bon(P, B, n, rng):
    K = max(1, B // n); d = P["d"]; A = rng.standard_normal((D * K, d)); ts = np.linspace(0, T_END, n + 1)
    for k in range(n):
        A = A + (ts[k + 1] - ts[k]) * P["G"].velocity(A, ts[k])
    A = A.reshape(D, K, d); idx = pick(Q(P, A) / P["beta"], K, rng)
    return A[np.arange(D), idx]


def smc(P, B, n, rng, kind="heur", guide=False, weights=True):
    per = 2 if guide else 1; K = max(1, B // (n * per)); d = P["d"]; beta = P["beta"]
    A = T0 * P["G"].sample(D * K, rng) + (1 - T0) * rng.standard_normal((D * K, d)); ts = np.linspace(T0, T_END, n + 1)
    phi = twist(P, A, T0, kind) / beta; lw = phi.copy().reshape(D, K)
    for k in range(n):
        t0, t1 = ts[k], ts[k + 1]; h = t1 - t0; var = g2(t0) * h
        mp = A + h * drift(P, A, t0)
        if guide:
            gq = grad_num(lambda X: twist(P, X, t0, kind), A) / beta
            mq = mp + var * gq
        else:
            mq = mp
        A1 = mq + np.sqrt(var) * rng.standard_normal(A.shape)
        phi1 = (Q(P, A1) if k == n - 1 else twist(P, A1, t1, kind)) / beta
        inc = phi1 - phi
        if guide:
            inc = inc - 0.5 * ((A1 - mp) ** 2).sum(-1) / var + 0.5 * ((A1 - mq) ** 2).sum(-1) / var
        lw = lw + inc.reshape(D, K); A, phi = A1, phi1
        if weights and K > 1:
            w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True); ess = 1 / (w**2).sum(1)
            rs = ess < K / 2
            if rs.any():
                A3 = A.reshape(D, K, d); P3 = phi.reshape(D, K)
                for i in np.where(rs)[0]:
                    j = rng.choice(K, K, p=w[i]); A3[i] = A3[i][j]; P3[i] = P3[i][j]; lw[i] = 0
                A = A3.reshape(D * K, d); phi = P3.reshape(-1)
    A = A.reshape(D, K, d)
    idx = pick(lw, K, rng) if (weights and K > 1) else np.zeros(D, int)
    return A[np.arange(D), idx]


def evaluate(P, X):
    lab = ((X[:, None, :] - P["tm"][None]) ** 2).sum(-1).argmin(1)
    mm = np.bincount(lab, minlength=3) / len(X)
    rng = np.random.default_rng(0); comp = rng.choice(3, 200_000, p=P["tw"])
    Ts = P["tm"][comp] + P["ts"] * rng.standard_normal((200_000, P["d"]))
    return float(np.abs(mm - P["tw"]).sum()), float(abs(Q(P, X).mean() - Q(P, Ts).mean()))


def main():
    out = {}; n = 8
    for d in [2, 8, 32]:
        P = setup(d, lam=0.15, beta=1.0)
        print(f"\nd={d}: behaviour masses [.6,.3,.1] -> exact tilted target masses {np.round(P['tw'], 3)}")
        for B in [16, 32, 64, 128]:
            M = {"bon": lambda r: bon(P, B, n, r),
                 "guid_heur": lambda r: smc(P, B, B // 2, r, "heur", guide=True, weights=False),
                 "fk_heur": lambda r: smc(P, B, n, r, "heur"),
                 "tds_heur": lambda r: smc(P, B, n, r, "heur", guide=True),
                 "guid_exact": lambda r: smc(P, B, B // 2, r, "exact", guide=True, weights=False),
                 "tds_exact": lambda r: smc(P, B, n, r, "exact", guide=True)}
            line = f" B={B:4d}"
            for name, fn in M.items():
                if name.startswith("tds") and B < 2 * n * 2:
                    continue
                res = [evaluate(P, fn(np.random.default_rng(100 * d + B + s))) for s in range(3)]
                ml = np.mean([r[0] for r in res]); qe = np.mean([r[1] for r in res])
                out[f"d{d}/B{B}/{name}"] = dict(modeL1=float(ml), Qerr=float(qe))
                line += f" | {name} {ml:.3f}/{qe:.2f}"
            print(line, flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e35p_twist.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
