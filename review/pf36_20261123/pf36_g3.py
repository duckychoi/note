"""#36 gate 3: 2-D spinning ball with friction, stochastic per-bounce restitution, intermittent obs. See PREREG.md (gate 3)."""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
T, CUT, SUB, SIG, R = 50, 25, 10, .015, .06
NOISE = .15
LW_H = .2  # Liu-West bandwidth
import os
NTH = int(os.environ.get('PF_NTH', '8'))
ENG = rollout.Rollout(nthread=NTH)
_CACHE = {}


def mk(mu, d):
    key = (round(float(mu), 6), round(float(d), 6))
    if key not in _CACHE:
        g = f'friction="{mu} .005 .0001" solref=".015 {d}" solimp=".95 .99 .001"'
        _CACHE[key] = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep=".002"/><worldbody>
<geom type="plane" size="100 100 .1" {g}/><body><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 0 1"/>
<joint type="hinge" axis="0 1 0"/><geom type="sphere" size="{R}" mass=".25" {g}/></body></worldbody></mujoco>""")
    return _CACHE[key]


BASE_M = mk(.5, .3)
DATAS = [mujoco.MjData(BASE_M) for _ in range(NTH)]
PF_MU, PF_D = np.geomspace(.06, 1.3, 12), np.linspace(.06, .9, 12)
POOL = [[mk(m, d) for d in PF_D] for m in PF_MU]
TR_MU, TR_D = np.linspace(.3, .8, 8), np.linspace(.2, .5, 8)
TRPOOL = [[mk(m, d) for d in TR_D] for m in TR_MU]
LO = np.array([-.2, .35, -.5, -1., -25.])
HI = np.array([.2, 1., 1.5, .5, 25.])


def init_states(rng, n):
    s = np.zeros((n, 7))
    u = LO + (HI-LO)*rng.uniform(size=(n, 5))
    s[:, 1], s[:, 2], s[:, 4], s[:, 5], s[:, 6] = u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4]
    return s


def step_frame(models, states, rng, noise=True, chunk=200_000):
    """Advance one frame; bounce (vz - -> +) gets multiplicative lognormal noise. Returns new states, bounce flags."""
    out = np.empty_like(states)
    for s in range(0, len(states), chunk):
        st, _ = ENG.rollout(models[s:s+chunk], DATAS, states[s:s+chunk], nstep=SUB)
        out[s:s+chunk] = st[:, -1]
    out[:, 0] = 0.
    bounce = (states[:, 5] < -.2) & (out[:, 5] > .1)  # real bounce, not resting chatter
    if noise:
        out[:, 5] = np.where(bounce, out[:, 5]*np.exp(NOISE*rng.standard_normal(len(out))), out[:, 5])
    return out, bounce


def simulate(models, states, rng, nframes, noise=True):
    """Return positions (n, nframes, 2) incl. initial frame, bounce flags (n, nframes)."""
    pos = np.empty((len(states), nframes, 2))
    bnc = np.zeros((len(states), nframes), bool)
    pos[:, 0] = states[:, 1:3]
    for t in range(1, nframes):
        states, b = step_frame(models, states, rng, noise)
        pos[:, t], bnc[:, t] = states[:, 1:3], b
    return pos, bnc


def make_cases(seed, n, test):
    rng = np.random.default_rng(seed)
    if test:
        mu = np.where(rng.uniform(size=n) < .5, rng.uniform(.08, .22, n), rng.uniform(.9, 1.2, n))
        d = np.where(rng.uniform(size=n) < .5, rng.uniform(.07, .17, n), rng.uniform(.55, .8, n))
    else:
        mu, d = rng.uniform(.3, .8, n), rng.uniform(.2, .5, n)
    models = [mk(a, b) for a, b in zip(mu, d)]
    pos, bnc = simulate(models, init_states(rng, n), rng, T)
    obs = pos + rng.normal(0, SIG, pos.shape)
    mask = np.zeros((n, T))
    mask[:, :CUT] = rng.uniform(size=(n, CUT)) > .4
    return dict(pos=pos, bnc=bnc, obs=obs, mask=mask, mu=mu, d=d)


def library(pool, mu_n, d_n, n, seed):
    rng = np.random.default_rng(seed)
    im, idd = rng.integers(0, mu_n, n), rng.integers(0, d_n, n)
    st0 = init_states(rng, n)
    pos, bnc = simulate([pool[a][b] for a, b in zip(im, idd)], st0, rng, T)
    return pos, bnc, st0, im, idd


# ------------------------------------------------------------------ scoring
def events(pos, bnc, xthr):
    return (bnc[..., CUT:].sum(-1) >= 2), (pos[..., -1, 0]-pos[..., CUT-1, 0] > xthr)


def score_weighted(fut, fb, fx, logw, case, xthr, rng, S=128):
    """fut (C,M,25,2) futures, fb/fx (C,M) event bools, logw (C,M)."""
    w = softmax(logw, 1)
    tf = case["pos"][:, CUT:]
    e1, e2 = events(case["pos"], case["bnc"], xthr)
    draws = np.stack([np.minimum(np.searchsorted(c, rng.uniform(size=S)), w.shape[1]-1) for c in w.cumsum(1)])
    smp = np.take_along_axis(fut, draws[..., None, None], 1)
    dist = lambda a, b: np.sqrt(np.square(a-b).mean((-1, -2)))
    energy = dist(smp, tf[:, None]).mean(1) - .5*dist(smp[:, :S//2], smp[:, S//2:]).mean(1)
    out = dict(energy=energy, brier_bounce2=np.square((w*fb).sum(1)-e1), brier_xpast=np.square((w*fx).sum(1)-e2))
    for k, j in (("cover_x", 0), ("cover_z", 1)):
        fin = fut[:, :, -1, j]
        o = np.argsort(fin, 1)
        fs, cw = np.take_along_axis(fin, o, 1), np.take_along_axis(w, o, 1).cumsum(1)
        lo = fs[np.arange(len(fs)), np.minimum((cw < .1).sum(1), fs.shape[1]-1)]
        hi = fs[np.arange(len(fs)), np.minimum((cw < .9).sum(1), fs.shape[1]-1)]
        out[k] = ((tf[:, -1, j] >= lo-1e-9) & (tf[:, -1, j] <= hi+1e-9)).astype(float)
    return out


def score_point(fut, b1, b2, case, xthr):
    tf = case["pos"][:, CUT:]
    e1, e2 = events(case["pos"], case["bnc"], xthr)
    return dict(energy=np.sqrt(np.square(fut-tf).mean((-1, -2))), brier_bounce2=np.square(b1.astype(float)-e1),
                brier_xpast=np.square(b2.astype(float)-e2))


# ------------------------------------------------------------------ methods
def lib_logw(lib_pos, case, temp, chunk=20000):
    m = np.repeat(case["mask"][:, :CUT, None], 2, -1).reshape(len(case["mask"]), -1)
    o = (case["obs"][:, :CUT]).reshape(len(m), -1)*m
    L = lib_pos[:, :CUT].reshape(len(lib_pos), -1)
    out = np.empty((len(m), len(L)))
    for s in range(0, len(L), chunk):
        Ls = L[s:s+chunk]
        out[:, s:s+chunk] = (o@Ls.T - .5*(m@np.square(Ls).T))/(SIG**2*temp)
    return out


def run_retrieval(lib, case, temp, xthr, seed, topk=4096):
    pos, bnc = lib[0], lib[1]
    lw = lib_logw(pos, case, temp)
    # keep top-k per case (weights outside are negligible; checked via mass reported)
    idx = np.argpartition(-lw, min(topk, lw.shape[1]-1), 1)[:, :topk] if lw.shape[1] > topk else np.tile(np.arange(lw.shape[1]), (len(lw), 1))
    full_w = softmax(lw, 1)
    kept_mass = np.take_along_axis(full_w, idx, 1).sum(1)
    sub = np.take_along_axis(lw, idx, 1)
    fut = pos[idx][:, :, CUT:]
    b1, b2 = events(pos[idx], bnc[idx], xthr)
    res = score_weighted(fut, b1, b2, sub, case, xthr, np.random.default_rng(seed))
    res["kept_mass"] = kept_mass
    return res


def run_pf_full(case, N, temp, xthr, seed, init=None, resample_after=0):
    """Same as run_pf but also tracks x at frame CUT-1 for E2."""
    rng = np.random.default_rng(seed)
    C = len(case["obs"])
    if init is None:
        st = init_states(rng, C*N)
        im, idd = rng.integers(0, 12, C*N), rng.integers(0, 12, C*N)
        logw = np.zeros((C, N))
    else:  # proposal draws with log(prior/proposal) initial weights
        st, im, idd, logw = init(rng)
        logw = logw.copy()
    obs, mask = case["obs"], case["mask"]
    fut = np.empty((C*N, T-CUT, 2))
    nb = np.zeros(C*N, int)
    x24 = None
    for t in range(T):
        if t > 0:
            st, b = step_frame([POOL[a][c] for a, c in zip(im, idd)], st, rng)
            if t >= CUT:
                nb += b
        if t >= CUT:
            fut[:, t-CUT] = st[:, 1:3]
            continue
        p = st[:, 1:3].reshape(C, N, 2)
        logw += -.5*np.square(obs[:, None, t]-p).sum(-1)*mask[:, t, None]/(SIG**2*temp)
        w = softmax(logw, 1)
        ess = 1/np.square(w).sum(1)
        res = np.where(ess < N/2)[0] if t >= resample_after else np.array([], int)
        if len(res):
            u = (rng.uniform(size=(len(res), 1))+np.arange(N))/N
            ix = np.stack([np.minimum(np.searchsorted(w[c].cumsum(), uu), N-1) for c, uu in zip(res, u)]) + res[:, None]*N
            sl = (res[:, None]*N + np.arange(N)).ravel()
            ix = ix.ravel()
            # Liu-West kernel jitter on continuous state (x,z,vx,vz,w): shrink to weighted mean, add h^2 * weighted cov
            D = [1, 2, 4, 5, 6]
            X = st[:, D].reshape(C, N, 5)[res]
            wr = w[res]
            mean = np.einsum("cn,cnd->cd", wr, X)
            dev = X-mean[:, None]
            cov = np.einsum("cn,cnd,cne->cde", wr, dev, dev) + 1e-10*np.eye(5)
            Lc = np.linalg.cholesky(cov)
            st[sl], im[sl], idd[sl] = st[ix], im[ix], idd[ix]
            n = len(sl)
            Xr = st[sl][:, D].reshape(len(res), N, 5)
            a = np.sqrt(1-LW_H**2)
            Xr = a*Xr + (1-a)*mean[:, None] + LW_H*np.einsum("cde,cne->cnd", Lc, rng.standard_normal(Xr.shape))
            tmp = st[sl]
            tmp[:, D] = Xr.reshape(-1, 5)
            st[sl] = tmp
            st[sl, 2] = np.maximum(st[sl, 2], R)
            im[sl] = np.clip(im[sl] + (rng.uniform(size=n) < .15)*rng.choice([-1, 1], n), 0, 11)
            idd[sl] = np.clip(idd[sl] + (rng.uniform(size=n) < .15)*rng.choice([-1, 1], n), 0, 11)
            logw[res] = 0.
        if t == CUT-1:
            x24 = st[:, 1].copy()
    fut = fut.reshape(C, N, T-CUT, 2)
    fb = (nb >= 2).reshape(C, N)
    fx = (fut[:, :, -1, 0] - x24.reshape(C, N)) > xthr
    return score_weighted(fut, fb, fx, logw, case, xthr, np.random.default_rng(seed+1))


def run_cem(case, xthr, seed, pop=64, gens=16):
    """Point MAP fit over (x0,z0,vx0,vz0,w0, mu, d) without process noise; 1024 rollouts per case."""
    rng = np.random.default_rng(seed)
    C = len(case["obs"])
    lo = np.r_[LO, np.log(.06), .06]
    hi = np.r_[HI, np.log(1.3), .9]
    mu_, sd_ = np.tile((lo+hi)/2, (C, 1)), np.tile((hi-lo)/4, (C, 1))
    best_p, best_s = None, np.full(C, np.inf)
    best_pos, best_b = np.zeros((C, T, 2)), np.zeros((C, T), bool)
    m = case["mask"][:, :CUT]
    for g in range(gens):
        P = np.clip(mu_[:, None]+sd_[:, None]*rng.standard_normal((C, pop, 7)), lo, hi).reshape(-1, 7)
        st = np.zeros((len(P), 7))
        st[:, 1], st[:, 2], st[:, 4], st[:, 5], st[:, 6] = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
        mi = np.abs(np.log(PF_MU)[None]-P[:, 5:6]).argmin(1)
        di = np.abs(PF_D[None]-P[:, 6:7]).argmin(1)
        pos, bnc = simulate([POOL[a][b] for a, b in zip(mi, di)], st, rng, T, noise=False)
        pos, bnc = pos.reshape(C, pop, T, 2), bnc.reshape(C, pop, T)
        sse = (np.square(pos[:, :, :CUT]-case["obs"][:, None, :CUT]).sum(-1)*m[:, None]).sum(-1)
        Pc = P.reshape(C, pop, 7)
        el = np.argsort(sse, 1)[:, :8]
        e = np.take_along_axis(Pc, el[..., None], 1)
        mu_, sd_ = e.mean(1), np.maximum(e.std(1), (hi-lo)*.01)
        k = sse.argmin(1)
        better = sse[np.arange(C), k] < best_s
        best_s = np.where(better, sse[np.arange(C), k], best_s)
        best_pos[better] = pos[np.arange(C), k][better]
        best_b[better] = bnc[np.arange(C), k][better]
    b1, b2 = events(best_pos, best_b, xthr)
    return score_point(best_pos[:, CUT:], b1, b2, case, xthr)


def run_extrap(case, xthr):
    C = len(case["obs"])
    dt = .02
    fut = np.zeros((C, T-CUT, 2))
    b1, b2 = np.zeros(C, bool), np.zeros(C, bool)
    for i in range(C):
        idx = np.where(case["mask"][i, :CUT] > 0)[0]
        tx = idx[-8:]*dt
        px = np.polyfit(tx, case["obs"][i, idx[-8:], 0], 1) if len(tx) >= 2 else np.array([0., case["obs"][i, idx[-1], 0]])
        tz = idx[-6:]*dt
        zz = case["obs"][i, idx[-6:], 1] + 4.905*tz**2
        pz = np.polyfit(tz, zz, 1) if len(tz) >= 2 else np.array([0., zz[-1]])
        # integrate z from last observed frame with reflection
        t0 = idx[-1]*dt
        z, vz = np.polyval(pz, t0)-4.905*t0**2, pz[0]-9.81*t0
        nb = 0
        traj = []
        tt = t0
        for f in range(idx[-1]+1, T):
            for _ in range(20):
                vz -= 9.81*dt/20
                z += vz*dt/20
                if z < R and vz < 0:
                    z, vz = R, -.6*vz
                    if f >= CUT:
                        nb += 1
            if f >= CUT:
                traj.append(z)
        fut[i, :, 1] = traj
        fut[i, :, 0] = np.polyval(px, np.arange(CUT, T)*dt)
        b1[i] = nb >= 2
        b2[i] = fut[i, -1, 0]-np.polyval(px, (CUT-1)*dt) > xthr
    return score_point(fut, b1, b2, case, xthr)


# ------------------------------------------------------------------ main
def summarize(res):
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    val = make_cases(51000, 48, test=False)
    tests = [make_cases(s, 48, test=True) for s in range(51200, 51204)]
    log("cases ready; test contacts mean bounces", float(np.mean([c["bnc"].sum(1).mean() for c in tests])))
    libs = {"retrieval_train_196k": library(TRPOOL, 8, 8, 196_608, 52001),
            "retrieval_train_1024": library(TRPOOL, 8, 8, 1024, 52002),
            "retrieval_broad_196k": library(POOL, 12, 12, 196_608, 52003)}
    trp, trb = libs["retrieval_train_196k"]
    xthr = float(np.median(trp[:, -1, 0]-trp[:, CUT-1, 0]))
    log("libraries ready; xthr", xthr)
    temps = {}
    for name, lib in libs.items():
        cand = {t: float(np.mean(run_retrieval(lib, val, t, xthr, 1)["energy"])) for t in (.5, 1, 2, 4, 8)}
        temps[name] = min(cand, key=cand.get)
        log(name, "val", cand)
    cand = {t: float(np.mean(run_pf_full(val, 1024, t, xthr, 3)["energy"])) for t in (1, 2, 4)}
    temps["pf"] = min(cand, key=cand.get)
    log("pf val", cand)
    rows = {}
    def add(key, per_seed, calls_case, calls_total):
        rows[key] = dict({k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]},
                         energy_per_seed=[p["energy"] for p in per_seed], calls_per_case=calls_case, calls_total=calls_total)
        log(key, {k: round(v, 4) for k, v in rows[key].items() if not isinstance(v, list)})
    nc = sum(len(c["obs"]) for c in tests)
    for name, lib in libs.items():
        add(name, [summarize(run_retrieval(lib, c, temps[name], xthr, 7+i)) for i, c in enumerate(tests)], 0, len(lib[0]))
    add("cem_map_1024", [summarize(run_cem(c, xthr, 11+i)) for i, c in enumerate(tests)], 1024, 1024*nc)
    add("extrapolation", [summarize(run_extrap(c, xthr)) for c in tests], 0, 0)
    for N in (64, 256, 1024, 8192):
        per = []
        for i, c in enumerate(tests):
            if N == 8192:  # chunk cases to bound memory
                parts = [run_pf_full({k: v[j:j+12] for k, v in c.items()}, N, temps["pf"], xthr, 100+N+i*10+j) for j in range(0, 48, 12)]
                per.append({k: float(np.mean(np.concatenate([p[k] for p in parts]))) for k in parts[0]})
            else:
                per.append(summarize(run_pf_full(c, N, temps["pf"], xthr, 100+N+i)))
        add(f"pf_N{N}", per, N, N*nc)
    # verdict
    ref = rows["pf_N8192"]["energy"]
    ret = min(rows["retrieval_train_196k"]["energy"], rows["retrieval_broad_196k"]["energy"])
    small = {N: rows[f"pf_N{N}"]["energy"]/ref-1 for N in (64, 256)}
    no_room = small[256] < .05 and abs(rows["pf_N256"]["energy"]/ret-1) < .05
    room = max(small.values()) >= .05 and ref <= .95*ret
    verdict = "no_headroom" if no_room else ("headroom" if room else "intermediate")
    log("verdict", verdict, small, "ref", ref, "best retrieval", ret)
    out = dict(config=dict(T=T, CUT=CUT, sigma=SIG, bounce_noise=NOISE, drop=.4, n_test=nc, xthr=xthr, temps=temps,
                           test_bounce_rate=float(np.mean([c["bnc"][:, CUT:].sum(1).mean() for c in tests])),
                           event_rates=dict(bounce2=float(np.mean([events(c["pos"], c["bnc"], xthr)[0].mean() for c in tests])),
                                            xpast=float(np.mean([events(c["pos"], c["bnc"], xthr)[1].mean() for c in tests])))),
               table=rows, verdict=dict(label=verdict, pf_small_vs_ref=small, ref_energy=ref, best_retrieval_energy=ret),
               seconds=time.time()-t0)
    (HERE/"gate3.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
