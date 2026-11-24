"""#14 beamspace sanity: the data family is Hamming taper x steering(theta0) -> a 1-parameter manifold.
Baseline that the original toy did not run: parametric (manifold) fit of theta0 and complex gain
from the surviving elements, by grid search + least squares.  Same generator/PSL as
origin/spectral-diffusion-beamspace:src/toy_generative_beam_prior.py.  numpy only.
"""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(0)
N = 16
th = np.linspace(-np.pi / 2, np.pi / 2, 361)


def steer(t):
    return np.exp(1j * np.pi * np.arange(N) * np.sin(t))


A = np.stack([steer(t) for t in th])
ham = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
good = lambda t: ham * steer(t)


def psl(w, t0, guard=0.18):
    P = np.abs(A @ np.conj(w)) ** 2; P /= P.max()
    return 10 * np.log10(P[np.abs(th - t0) >= guard].max() + 1e-12)


Wd = np.stack([good(t) for t in rng.uniform(-0.7, 0.7, 600)])
mu = Wd.mean(0); Xc = Wd - mu
Sig = Xc.conj().T @ Xc / len(Wd) + 1e-3 * np.eye(N)
grid = np.linspace(-0.75, 0.75, 3001)
G = np.stack([good(t) for t in grid])          # manifold atlas


def inpaint_gauss(wo, obs):
    mis = np.setdiff1d(np.arange(N), obs)
    So = Sig[np.ix_(obs, obs)] + 1e-6 * np.eye(len(obs))
    w = mu.astype(complex).copy(); w[obs] = wo
    w[mis] = mu[mis] + Sig[np.ix_(mis, obs)] @ np.linalg.solve(So, wo - mu[obs]); return w


def inpaint_manifold(wo, obs):
    Go = G[:, obs]                                # (grid, |obs|)
    c = (Go.conj() * wo).sum(1) / (np.abs(Go) ** 2).sum(1)     # LS complex gain per theta
    r = np.linalg.norm(wo[None] - c[:, None] * Go, axis=1)
    j = int(np.argmin(r)); return c[j] * G[j], grid[j]


out = {}
for k in (2, 4, 6, 8, 10, 12):
    z, g, m, t, terr = [], [], [], [], []
    for _ in range(200):
        t0 = rng.uniform(-0.7, 0.7); wt = good(t0)
        brk = rng.choice(N, k, replace=False); obs = np.setdiff1d(np.arange(N), brk)
        wz = wt.copy(); wz[brk] = 0
        wm, th_hat = inpaint_manifold(wt[obs], obs)
        z.append(psl(wz, t0)); g.append(psl(inpaint_gauss(wt[obs], obs), t0)); m.append(psl(wm, t0)); t.append(psl(wt, t0)); terr.append(abs(th_hat - t0))
    out[k] = dict(zero=np.mean(z), gauss=np.mean(g), manifold=np.mean(m), truth=np.mean(t), theta_abs_err=float(np.mean(terr)))
    print(k, {kk: round(v, 2) if kk != "theta_abs_err" else f"{v:.1e}" for kk, v in out[k].items()})
Path(__file__).with_name("beam_sanity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
