"""#36 gate 7: multi-throw. Same ball & floor (one unseen material per object); after n in {1,2,4,8} fully observed throws, forecast
the next throw from its 25-frame prefix.  carry: material posterior pi_n(c) ∝ Π_j p̂(y_j | c) over the 144-class grid (each past
throw's marginal likelihood multiplied ONCE, uniform grid prior once), then IS for the next throw with prior pi_n.  scratch: uniform
material prior.  oracle: true continuous material.  See PREREG.md (관문 7).  Usage: python pf36_g7.py run"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402

import pf36_g3 as g  # noqa: E402
import pf36_g4b as f  # noqa: E402
import pf36_g6a as a6  # noqa: E402

torch.set_num_threads(16)
HERE = Path(__file__).resolve().parent
ALPHA = .1
LOGV_Z = float(np.sum(np.log(g.HI-g.LO)))
NTHROW = 8


def objects(seed, n):
    """n objects (unseen material bands), each with NTHROW+1 throws. Past throws: 40% drop over all 50 frames; next throw: prefix only."""
    rng = np.random.default_rng(seed)
    mu = np.where(rng.uniform(size=n) < .5, rng.uniform(.08, .22, n), rng.uniform(.9, 1.2, n))
    d = np.where(rng.uniform(size=n) < .5, rng.uniform(.07, .17, n), rng.uniform(.55, .8, n))
    models = [g.mk(a, b) for a, b in zip(mu, d)]
    throws = []
    for j in range(NTHROW+1):
        pos, bnc = g.simulate(models, g.init_states(rng, n), rng, g.T)
        obs = pos + rng.normal(0, g.SIG, pos.shape)
        mask = (rng.uniform(size=(n, g.T)) > .4).astype(float)
        if j == NTHROW:
            mask[:, g.CUT:] = 0
        throws.append(dict(pos=pos, bnc=bnc, obs=obs, mask=mask))
    return dict(mu=mu, d=d, models=models, throws=throws)


@torch.no_grad()
def flow_given_class(model, x, cls, gen):
    """Sample z ~ Flow(z | x, cls) and return z and log Flow(z|x,cls) (class term removed)."""
    h = model.enc(x)
    c = model.ctx(h, cls)
    v = torch.randn(len(x), 5, generator=gen)
    for l in reversed(model.layers):
        v = l.inverse(v, c)
    z = v*torch.tensor(f.HW, dtype=torch.float32)+torch.tensor(f.CEN, dtype=torch.float32)
    return z


@torch.no_grad()
def log_flow(model, x, cls, z):
    lc = torch.log_softmax(model.cat(model.enc(x)), -1).gather(-1, cls[:, None])[:, 0]
    return model.log_prob(x, cls, z) - lc


def propose(model, obs, mask, cls, rng, gen):
    """Per row: z ~ (1-ALPHA) Flow(z|y,cls) + ALPHA prior; returns z (rows,5) and log q(z|y,cls) (exact)."""
    x = torch.tensor(f.features(obs, mask))
    cls_t = torch.tensor(cls)
    z = flow_given_class(model, x, cls_t, gen).numpy().astype(float)
    pri = rng.uniform(size=len(z)) < ALPHA
    z[pri] = g.LO + (g.HI-g.LO)*rng.uniform(size=(int(pri.sum()), 5))
    lf = log_flow(model, x, cls_t, torch.tensor(z, dtype=torch.float32)).numpy().astype(float)
    inside = np.all((z >= g.LO) & (z <= g.HI), -1)
    lq = np.logaddexp(np.log(1-ALPHA)+lf, np.log(ALPHA)-LOGV_Z)
    lp = np.where(inside, -LOGV_Z, -np.inf)
    return z, lp, lq


def simulate_ll(models_rows, z, obs_rows, mask_rows, rng, upto, keep_future=False):
    st = np.zeros((len(z), 7))
    st[:, [1, 2, 4, 5, 6]] = z
    ll = -.5*np.square(obs_rows[:, 0]-st[:, 1:3]).sum(-1)*mask_rows[:, 0]/g.SIG**2
    fut = []
    for t in range(1, g.T if keep_future else upto):
        st, b = g.step_frame(models_rows, st, rng)
        if t < upto:
            ll += -.5*np.square(obs_rows[:, t]-st[:, 1:3]).sum(-1)*mask_rows[:, t]/g.SIG**2
        else:
            fut.append(st[:, 1:3].copy())
    return ll, st, (np.stack(fut, 1) if fut else None)


def class_loglik(model, thr, K, rng, gen):
    """log p̂(y | c) for every object and class (n,144) by IS over launch state with prior kicks, full 50-frame likelihood."""
    n = len(thr["obs"])
    cls = np.tile(np.repeat(np.arange(144), K), n)
    obj = np.repeat(np.arange(n), 144*K)
    z, lp, lq = propose(model, thr["obs"][obj], thr["mask"][obj], cls, rng, gen)
    ll, _, _ = simulate_ll([g.POOL[c//12][c % 12] for c in cls], z, thr["obs"][obj], thr["mask"][obj], rng, g.T)
    lw = (lp + ll - lq).reshape(n, 144, K)
    return logsumexp(lw, -1) - np.log(K)


def next_throw(model, thr, logprior, N, rng, gen, true_models=None):
    """IS for the next throw's prefix with material prior exp(logprior) (n,144); forecast frames 25-49 after final resampling."""
    n = len(thr["obs"])
    obj = np.repeat(np.arange(n), N)
    x = torch.tensor(f.features(thr["obs"], thr["mask"]))
    with torch.no_grad():
        cat = torch.softmax(model.cat(model.enc(x)), -1).numpy().astype(float)
    pri = np.exp(logprior - logsumexp(logprior, 1, keepdims=True))
    qcls = .5*cat + .5*pri
    cls = np.concatenate([rng.choice(144, N, p=q/q.sum()) for q in qcls])
    if true_models is not None:                 # oracle: true continuous material; flow conditioned on nearest class
        cls = np.repeat(np.array([np.abs(np.log(g.PF_MU)-np.log(m.geom_friction[1, 0])).argmin()*12 +
                                  np.abs(g.PF_D-m.geom_solref[1, 1]).argmin() for m in true_models]), N)
    z, lp, lq = propose(model, thr["obs"][obj], thr["mask"][obj], cls, rng, gen)
    rows_models = [true_models[o] for o in obj] if true_models is not None else [g.POOL[c//12][c % 12] for c in cls]
    ll, _, fut = simulate_ll(rows_models, z, thr["obs"][obj], thr["mask"][obj], rng, g.CUT, keep_future=True)
    if true_models is not None:
        lw = lp + ll - lq
    else:
        lw = np.log(pri[obj, cls]+1e-300) + lp + ll - (np.log(qcls[obj, cls]) + lq)
    lw = lw.reshape(n, N)
    fut = fut.reshape(n, N, g.T-g.CUT, 2)
    # final resampling (no jitter) then score
    w = softmax(lw, 1)
    case = dict(pos=thr["pos"], bnc=thr["bnc"])
    dr = np.stack([np.minimum(np.searchsorted(ww.cumsum(), rng.uniform(size=128)), N-1) for ww in w])
    smp = np.take_along_axis(fut, dr[..., None, None], 1)
    truth = thr["pos"][:, g.CUT:]
    dist = lambda a, b: np.sqrt(np.square(a-b).mean((-1, -2)))
    energy = dist(smp, truth[:, None]).mean(1) - .5*dist(smp[:, :64], smp[:, 64:]).mean(1)
    st = a6.interval_stats(a6.intervals(fut, lw), thr["pos"])
    return dict(energy=energy, ess=1/np.square(w).sum(1), **{f"{k}_{q}": np.full(n, st[k][q]) for k in st for q in ("cover", "width", "iscore")})


def run(log):
    models = a6.load_models()
    model = models[8100]
    t_all = time.time()
    rows = []
    for block in range(4):
        ob = objects(57000+block, 48)
        rng = np.random.default_rng(57100+block)
        gen = torch.Generator().manual_seed(57100+block)
        # material posterior after n throws (marginal likelihood of each past throw multiplied once)
        t0 = time.time()
        cll = np.stack([class_loglik(model, ob["throws"][j], 64, rng, gen) for j in range(NTHROW)], 0)   # (8,n,144)
        t_cll = (time.time()-t0)/NTHROW
        cum = np.cumsum(cll, 0)
        lmu, dd = np.log(g.PF_MU)[np.arange(144)//12], g.PF_D[np.arange(144) % 12]
        nxt = ob["throws"][NTHROW]
        for n_thr in (0, 1, 2, 4, 8):
            logprior = np.zeros((48, 144)) if n_thr == 0 else cum[n_thr-1]
            post = softmax(logprior, 1)
            mu_hat, d_hat = post@lmu, post@dd
            for N in ((256,) if n_thr else (256, 1024, 4096)):
                t1 = time.time()
                r = next_throw(model, nxt, logprior, N, rng, gen)
                sec = time.time()-t1 + (t_cll*n_thr)
                rows.append(dict(block=block, arm="carry" if n_thr else "scratch", throws=n_thr, N=N, sec=sec,
                                 **{k: float(np.mean(v)) for k, v in r.items()},
                                 logmu_abs_err=float(np.mean(abs(mu_hat-np.log(ob["mu"])))), d_abs_err=float(np.mean(abs(d_hat-ob["d"]))),
                                 logmu_post_sd=float(np.mean(np.sqrt(np.maximum(post@lmu**2-mu_hat**2, 0)))),
                                 d_post_sd=float(np.mean(np.sqrt(np.maximum(post@dd**2-d_hat**2, 0))))))
                log(block, rows[-1]["arm"], n_thr, N, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
        t1 = time.time()
        r = next_throw(model, nxt, np.zeros((48, 144)), 256, rng, gen, true_models=ob["models"])
        rows.append(dict(block=block, arm="oracle", throws=-1, N=256, sec=time.time()-t1, **{k: float(np.mean(v)) for k, v in r.items()}))
        log(block, "oracle", {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
        (HERE/"gate7.json").write_text(json.dumps(dict(rows=rows, total_sec=time.time()-t_all), indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    dict(run=run)[sys.argv[1]](log)
