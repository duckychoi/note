"""#36(+#35) sequential particle-filter gate. See PREREG.md. Imports verification_20261123 helpers only."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"verification_20261123"))
from contact36 import BASE, DT, SIGMA, TIME, model, trajectories  # noqa: E402
from bounce36 import event_bank, prediction  # noqa: E402

G, FLOOR, CUT = 9.81, .06, 12
LO, HI = np.array([.32, -1.15, 0.]), np.array([1.72, .5, .9])
T_ALL = DT*TIME


def traj(theta, idx=None):
    """Analytic bounce trajectory, identical formula to bounce36.event_bank. theta (...,3) -> (...,len(idx))."""
    t = T_ALL if idx is None else T_ALL[idx]
    base = BASE if idx is None else BASE[idx]
    h, v, e = theta[..., 0:1], theta[..., 1:2], theta[..., 2:3]
    s0 = np.sqrt(v**2 + 2*G*(h-FLOOR))
    first = (v+s0)/G
    y = h + v*t + base
    touched = t >= first
    elapsed = np.clip(t-first, 0, None)
    es = np.clip(e, 1e-9, .999)
    stop = (2/G)*s0*es/(1-es)
    flying = touched & (elapsed < stop) & (e > 1e-9)
    ratio = np.clip(elapsed/np.where(stop > 0, stop, 1), 0, 1-1e-12)
    bounce = np.floor(np.log1p(-ratio)/np.log(es))
    since = elapsed - stop*(1-es**bounce)
    speed = s0*es**(bounce+1)
    yb = FLOOR + speed*since - .5*G*since**2
    return np.where(flying, yb, np.where(touched, FLOOR, y))


def systematic(w, rng):
    """w (C,N) normalized -> indices (C,N)."""
    C, N = w.shape
    u = (rng.uniform(size=(C, 1)) + np.arange(N))/N
    return np.minimum(np.array([np.searchsorted(c, uu) for c, uu in zip(w.cumsum(1), u)]), N-1)


def loglik(theta, obs, mask, temp, upto):
    """Tempered log-likelihood of frames < upto that are observed. theta (C,N,3), obs/mask (C,T)."""
    idx = np.arange(upto)
    pred = traj(theta, idx)
    r = (obs[:, None, :upto]-pred)**2*mask[:, None, :upto]
    return -.5*r.sum(-1)/(SIGMA**2*temp)


def pf_continuous(obs, mask, N, temp, rng, upto=CUT, K=3):
    C = len(obs)
    theta = LO + (HI-LO)*rng.uniform(size=(C, N, 3))
    logw = np.zeros((C, N))
    calls = np.full(C, N, float)
    for t in range(upto):
        if mask[:, t].any():
            pred = traj(theta, [t])[..., 0]
            logw += -.5*(obs[:, None, t]-pred)**2*mask[:, None, t]/(SIGMA**2*temp)
        w = softmax(logw, 1)
        ess = 1/np.square(w).sum(1)
        need = ess < N/2
        if not need.any():
            continue
        sel = np.where(need)[0]
        ix = systematic(w[sel], rng)
        th = np.take_along_axis(theta[sel], ix[..., None], 1)
        cur = loglik(th, obs[sel], mask[sel], temp, t+1)
        for _ in range(K):
            mu = th.mean(1, keepdims=True)
            cov = np.einsum("cni,cnj->cij", th-mu, th-mu)/N + 1e-10*np.eye(3)
            L = np.linalg.cholesky(cov*2.38**2/3)
            prop = th + np.einsum("cij,cnj->cni", L, rng.standard_normal(th.shape))
            inside = np.all((prop >= LO) & (prop <= HI), -1)
            new = np.where(inside, loglik(np.where(inside[..., None], prop, th), obs[sel], mask[sel], temp, t+1), -np.inf)
            acc = np.log(rng.uniform(size=new.shape)) < new-cur
            th = np.where(acc[..., None], prop, th)
            cur = np.where(acc, new, cur)
            calls[sel] += N
        theta[sel] = th
        logw[sel] = 0.
    return theta, logw, calls


def pf_grid(obs, bank, shape, N, rng, K=3, step=2):
    """Discrete-grid PF (validation): particles = grid indices, symmetric neighbour random-walk MH."""
    C, T = obs.shape
    dims = np.array(shape)
    ijk = np.stack([rng.integers(0, d, size=(C, N)) for d in dims], -1)
    flat = lambda a: np.ravel_multi_index(tuple(np.moveaxis(a, -1, 0)), shape)
    logw = np.zeros((C, N))
    for t in range(T):
        logw += -.5*(obs[:, None, t]-bank[flat(ijk), t])**2/SIGMA**2
        w = softmax(logw, 1)
        if (1/np.square(w).sum(1) < N/2).any():
            sel = np.where(1/np.square(w).sum(1) < N/2)[0]
            ix = systematic(w[sel], rng)
            a = np.take_along_axis(ijk[sel], ix[..., None], 1)
            ll = lambda b: (-.5*(obs[sel, None, :t+1]-bank[flat(b), :t+1])**2/SIGMA**2).sum(-1)
            cur = ll(a)
            for _ in range(K):
                prop = a + rng.integers(-step, step+1, size=a.shape)
                inside = np.all((prop >= 0) & (prop < dims), -1)
                prop = np.where(inside[..., None], prop, a)
                new = np.where(inside, ll(prop), -np.inf)
                acc = np.log(rng.uniform(size=new.shape)) < new-cur
                a = np.where(acc[..., None], prop, a)
                cur = np.where(acc, new, cur)
            ijk[sel] = a
            logw[sel] = 0.
    w = softmax(logw, 1)
    return np.einsum("cn,cnt->ct", w, bank[flat(ijk)]), w, ijk


def score(futures, logw, y, rng, shared=False, S=128):
    """futures (C,M,12) or shared (M,12); logw (C,M). Energy score, Brier(final>.2), 80% coverage of final height."""
    w = softmax(logw, 1)
    fut = y[:, CUT:]
    cdf = w.cumsum(1)
    draws = np.stack([np.minimum(np.searchsorted(c, rng.uniform(size=S)), w.shape[1]-1) for c in cdf])
    samples = futures[draws] if shared else np.take_along_axis(futures, draws[..., None], 1)
    dist = lambda a, b: np.sqrt(np.square(a-b).mean(-1))
    energy = dist(samples, fut[:, None]).mean(1) - .5*dist(samples[:, :S//2], samples[:, S//2:]).mean(1)
    final = futures[..., -1] if not shared else np.broadcast_to(futures[:, -1], w.shape)
    pred = (w*(final > .20)).sum(1)
    event = fut[:, -1] > .20
    order = np.argsort(final, 1)
    fs, cw = np.take_along_axis(final, order, 1), np.take_along_axis(w, order, 1).cumsum(1)
    lo = fs[np.arange(len(fs)), np.minimum((cw < .1).sum(1), fs.shape[1]-1)]
    hi = fs[np.arange(len(fs)), np.minimum((cw < .9).sum(1), fs.shape[1]-1)]
    cover = (fut[:, -1] >= lo-1e-9) & (fut[:, -1] <= hi+1e-9)
    return dict(energy=energy, brier=np.square(pred-event), cover=cover.astype(float))


# ---------------------------------------------------------------- gate 1
def gate1():
    bank = event_bank()
    shape = (71, 34, 10)
    grid_theta = np.stack(np.meshgrid(np.linspace(.32, 1.72, 71), np.linspace(-1.15, .5, 34),
                                      np.linspace(0, .9, 10), indexing="ij"), -1).reshape(-1, 3)
    assert np.max(abs(traj(grid_theta)-bank)) < 1e-9, "analytic traj != event_bank"
    models = [model(d) for d in (.18, .4, .75)]
    rows = []
    for seed in range(31200, 31204):
        y, obs, _ = trajectories(models, seed)
        exact_mean, _ = prediction(bank, obs, 1.0)
        lp = (obs@bank.T - .5*(bank**2).sum(1))/SIGMA**2
        wx = softmax(lp, 1)
        e_exact = wx@grid_theta[:, 2]
        pf_mean, w, ijk = pf_grid(obs, bank, shape, 1024, np.random.default_rng(seed))
        e_pf = (w*grid_theta[np.ravel_multi_index(tuple(np.moveaxis(ijk, -1, 0)), shape), 2]).sum(1)
        rows.append(dict(seed=seed,
                         traj_rmse_mm=(1000*np.sqrt(np.square(pf_mean-exact_mean).mean(1))).tolist(),
                         e_diff=np.abs(e_pf-e_exact).tolist(),
                         exact_vs_truth_mm=(1000*np.sqrt(np.square(exact_mean-y).mean(1))).tolist()))
    tr = np.concatenate([r["traj_rmse_mm"] for r in rows]); ed = np.concatenate([r["e_diff"] for r in rows])
    summary = dict(n_cases=len(tr), traj_rmse_mm_median=float(np.median(tr)), traj_rmse_mm_p90=float(np.percentile(tr, 90)),
                   e_diff_median=float(np.median(ed)), e_diff_p90=float(np.percentile(ed, 90)),
                   frac_traj_rmse_le_3mm=float(np.mean(tr <= 3)),
                   passed=bool(np.median(tr) <= 3 and np.median(ed) <= .05))
    return dict(summary=summary, rows=rows)


# ---------------------------------------------------------------- gate 2 / 2b
MATS = {"within_materials": (.18, .4, .75), "new_materials": (.12, .55, .95)}


def make_mask(y, first, rng, intermittent):
    mask = np.zeros(y.shape)
    mask[:, :CUT] = 1
    if intermittent:
        mask[:, :CUT] *= rng.uniform(size=(len(y), CUT)) > .4
        for i, f in enumerate(first):
            if 0 <= f < CUT:
                mask[i, max(f-2, 0):f+3] = 0
        mask[:, CUT:] = 0
    return mask


def libraries():
    lib = {"retrieval_train_materials": trajectories([model(d) for d in MATS["within_materials"]], 41001, n=2048)[0],
           "retrieval_prior_materials": trajectories([model(d) for d in np.linspace(.1, 1., 64)], 41002, n=2048)[0]}
    rng = np.random.default_rng(41000)
    full = event_bank()
    lib["analytic_grid_full"] = full
    lib["analytic_random_2048"] = full[rng.choice(len(full), 2048, replace=False)]
    return lib


def lib_logw(lib, obs, mask, temp):
    o, m = obs*mask, mask
    # -0.5*sum m (o-b)^2 = o.b - .5 m.b^2 (+const)
    return ((o[:, :CUT]@lib[:, :CUT].T) - .5*(m[:, :CUT]@np.square(lib[:, :CUT]).T))/(SIGMA**2*temp)


def job(args):
    kind, name, seed, mats, temp, intermittent, N = args
    lib = LIB
    y, obs, first = trajectories([model(d) for d in MATS[mats]], seed)
    mask = make_mask(y, first, np.random.default_rng(seed+7), intermittent)
    rng = np.random.default_rng(seed+1000)
    t0 = time.time()
    if kind == "lib":
        L = lib[name]
        lw = lib_logw(L, obs, mask, temp)
        res = score(L[:, CUT:], lw, y, rng, shared=True)
        if name == "retrieval_prior_materials":  # single best physics fit = MAP of the MuJoCo prior library
            onehot = np.where(lw == lw.max(1, keepdims=True), 0., -np.inf)
            res_map = score(L[:, CUT:], onehot, y, np.random.default_rng(seed+1000), shared=True)
            res = dict(res, map_energy=res_map["energy"], map_brier=res_map["brier"])
        calls = np.zeros(len(y))
    else:
        chunk = max(1, 2_000_000//N)
        outs, calls = [], []
        for s in range(0, len(y), chunk):
            sl = slice(s, s+chunk)
            th, lw, c = pf_continuous(obs[sl], mask[sl], N, temp, np.random.default_rng(seed*7+s+N))
            outs.append(score(traj(th)[..., CUT:], lw, y[sl], np.random.default_rng(seed+1000+s)))
            calls.append(c)
        res = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        calls = np.concatenate(calls)
    return dict(kind=kind, name=name, N=N, seed=seed, materials=mats, temp=temp, intermittent=intermittent,
                metrics={k: float(np.mean(v)) for k, v in res.items()}, calls_per_case=float(calls.mean()),
                seconds=time.time()-t0)


LIB = None


def init_pool():
    global LIB
    LIB = libraries()


def methods():
    return [("lib", n, None) for n in ("retrieval_train_materials", "retrieval_prior_materials",
                                        "analytic_grid_full", "analytic_random_2048")] + \
           [("pf", "pf_analytic", N) for N in (64, 256, 1024, 8192)]


def gate2(intermittent, pool):
    temps = (.5, 1., 2., 4.)
    val = [(k, n, s, "within_materials", t, intermittent, N if N else 0)
           for k, n, N in methods() if not (k == "pf" and N != 1024) for s in (41100, 41101) for t in temps]
    vres = pool.map(job, val)
    chosen = {}
    for k, n, N in methods():
        cand = {t: np.mean([r["metrics"]["energy"] for r in vres if r["name"] == n and r["temp"] == t]) for t in temps}
        chosen[n] = min(cand, key=cand.get)
    mats = ("new_materials",) if intermittent else ("within_materials", "new_materials")
    test = [(k, n, s, m, chosen[n], intermittent, N if N else 0) for k, n, N in methods()
            for m in mats for s in range(41200, 41212)]
    tres = pool.map(job, test)
    table = {}
    for m in mats:
        for k, n, N in methods():
            rs = [r for r in tres if r["materials"] == m and r["name"] == n and r["N"] == (N if N else 0)]
            key = f"{m}|{n}" + (f"|N={N}" if N else "")
            per_seed = np.array([r["metrics"]["energy"] for r in rs])
            table[key] = dict({mk: float(np.mean([r["metrics"][mk] for r in rs])) for mk in rs[0]["metrics"]},
                              energy_seed_sd=float(per_seed.std(ddof=1)), calls_per_case=float(np.mean([r["calls_per_case"] for r in rs])),
                              temp=chosen[n], seconds=float(sum(r["seconds"] for r in rs)), per_seed_energy=per_seed.tolist())
    return dict(temperatures=chosen, table=table, val_raw=vres, test_raw=tres)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "gate1":
        out = gate1()
        print(json.dumps(out["summary"], indent=1))
        (HERE/"gate1.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    else:
        with Pool(22, initializer=init_pool) as pool:
            out = gate2(step == "gate2b", pool)
        for k, v in out["table"].items():
            print(k, {kk: round(vv, 4) for kk, vv in v.items() if not isinstance(vv, list)})
        (HERE/f"{step}.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
