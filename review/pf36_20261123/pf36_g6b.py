"""#36 gate 6B (v2, after audit): init {prior, prefix-flow} x kick proposal {engine prior, selective learned (1-D mixture, exact),
always-on interval proposal (FPPF noise-space adaptation, conditional RealNVP, exact)}; equal N, same resampling, temperature 1, no jitter.
S1: observe frames 0-24, forecast 25-49.  S2 (online): observe 0-39 (observations exist in 25-39), score filtering at observed frames
25-39 in three bins, forecast 40-49.  See PREREG.md (관문 6B v2).
Usage: python pf36_g6b.py train | rule | run"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from scipy.special import softmax, logsumexp  # noqa: E402

import pf36_g3 as g  # noqa: E402
import pf36_g4b as f  # noqa: E402
import pf36_g6a as a6  # noqa: E402  (installs the mj_step counter)

torch.set_num_threads(16)
HERE = Path(__file__).resolve().parent
KMAX = 8
M_BANK = 147_456
BETA = .1
LOG2PI = np.log(2*np.pi)


# ---------------------------------------------------------------- cases with observations over the whole trajectory
def cases_full(seed, n, kind):
    c = a6.cases_custom(seed, n, kind)                     # same prefix mask/obs as before
    r = np.random.default_rng(seed + 1_000_003)           # separate RNG for the online mask
    c["mask"][:, g.CUT:] = r.uniform(size=(n, g.T-g.CUT)) > .4
    return c


# ---------------------------------------------------------------- engine step split into pre-kick rollout + kick
def step_pre(models, states):
    out = np.empty_like(states)
    for s in range(0, len(states), 200_000):
        st, _ = g.ENG.rollout(models[s:s+200_000], g.DATAS, states[s:s+200_000], nstep=g.SUB)
        out[s:s+200_000] = st[:, -1]
    out[:, 0] = 0.
    a6.COUNT["rows"] += len(states)
    return out, (states[:, 5] < -.2) & (out[:, 5] > .1)


# ---------------------------------------------------------------- learned proposals
def kcond(state5, mu_idx, d_idx, j, obs):
    oh = np.zeros((len(j), KMAX), np.float32)
    oh[np.arange(len(j)), np.asarray(j)-1] = 1
    return np.concatenate([state5, np.log(g.PF_MU[mu_idx])[:, None], g.PF_D[d_idx][:, None], oh, obs], 1).astype(np.float32)


class KickMix(nn.Module):
    """q(u | pre-kick state, material, frames-to-observation, observation) = K-component Gaussian mixture (monotone CDF, exact)."""
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(17, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3*K))

    def params(self, c):
        h = self.net(c)
        return torch.log_softmax(h[:, :self.K], -1), h[:, self.K:2*self.K], -2.5+3.5*torch.sigmoid(h[:, 2*self.K:])

    def log_prob(self, c, u):
        lw, mu, ls = self.params(c)
        return torch.logsumexp(lw - .5*((u[:, None]-mu)/torch.exp(ls))**2 - ls - .5*LOG2PI, -1)

    @torch.no_grad()
    def sample(self, c, gen):
        lw, mu, ls = self.params(c)
        k = torch.multinomial(torch.exp(lw), 1, generator=gen)[:, 0]
        r = torch.arange(len(c))
        return mu[r, k] + torch.exp(ls[r, k])*torch.randn(len(c), generator=gen)


class CondCoupling(nn.Module):
    def __init__(self, keep, ctx):
        super().__init__()
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(KMAX+ctx, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 2*KMAX))

    def st(self, u, c):
        h = self.net(torch.cat([u*self.keep, c], -1))
        return 2*torch.tanh(h[:, :KMAX]/2)*(1-self.keep), h[:, KMAX:]*(1-self.keep)

    def forward(self, u, c):       # data -> base, positive scales exp(-s): invertible by construction
        s, t = self.st(u, c)
        return (u-t)*torch.exp(-s), -s.sum(-1)

    def inverse(self, v, c):
        s, t = self.st(v, c)
        return v*torch.exp(s)+t


class GapFlow(nn.Module):
    """Always-on interval proposal q(u_{s+1..s+8} | state at s, material, gap k, observation at s+k): conditional RealNVP (exact)."""
    def __init__(self):
        super().__init__()
        masks = [[1, 0]*4, [0, 1]*4, [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]]*2
        self.layers = nn.ModuleList([CondCoupling(m, 17) for m in masks])

    def log_prob(self, u, c):
        ld = 0.
        for l in self.layers:
            u, d = l(u, c)
            ld = ld + d
        return -.5*(u**2).sum(-1) - KMAX/2*LOG2PI + ld

    @torch.no_grad()
    def sample(self, c, gen):
        v = torch.randn(len(c), KMAX, generator=gen)
        for l in reversed(self.layers):
            v = l.inverse(v, c)
        return v


def logged_bank(n, seed):
    """Re-simulate the broad bank (same RNG stream as g.library(POOL,...,196608,52003)) logging pre-kick states, kicks, bounces."""
    rng = np.random.default_rng(seed)
    N0 = 196_608
    im, idd = rng.integers(0, 12, N0), rng.integers(0, 12, N0)
    st = g.init_states(rng, N0)
    models = [g.POOL[a][b] for a, b in zip(im, idd)]
    S = np.empty((n, g.T, 5), np.float32)       # post-kick state per frame
    P = np.empty((n, g.T, 5), np.float32)       # pre-kick state per frame
    E = np.zeros((n, g.T), np.float32)
    B = np.zeros((n, g.T), bool)
    S[:, 0] = P[:, 0] = st[:n][:, [1, 2, 4, 5, 6]]
    for t in range(1, g.T):
        out, bounce = step_pre(models, st)
        P[:, t] = out[:n][:, [1, 2, 4, 5, 6]]
        eps = rng.standard_normal(N0)
        out[:, 5] = np.where(bounce, out[:, 5]*np.exp(g.NOISE*eps), out[:, 5])
        st = out
        S[:, t] = st[:n][:, [1, 2, 4, 5, 6]]
        E[:, t], B[:, t] = eps[:n], bounce[:n]
    return S, P, E, B, im[:n], idd[:n]


def fit(model, data, epochs, lr, seed, log, name):
    C, U = data
    n = len(C)
    nv = n//10
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state = np.inf, None
    for ep in range(epochs):
        perm = torch.randperm(n-nv)+nv
        for b in range(0, len(perm), 2048):
            j = perm[b:b+2048]
            loss = -model.log_prob(*((C[j], U[j]) if isinstance(model, KickMix) else (U[j], C[j]))).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        with torch.no_grad():
            vl = float(-model.log_prob(*((C[:nv], U[:nv]) if isinstance(model, KickMix) else (U[:nv], C[:nv]))).mean())
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    log(name, "val nll", round(best, 4))
    return best


def train(log):
    t0 = time.time()
    c0 = a6.COUNT["rows"]
    S, P, E, B, im, idd = logged_bank(M_BANK, 52003)
    t_bank = time.time()-t0
    replay = (a6.COUNT["rows"]-c0)*g.SUB/a6.TRAJ_STEPS
    log("logged bank", t_bank, "replay traj-equiv", replay)
    rng = np.random.default_rng(6200)
    # selective kick examples: every bounce frame f, observation j frames later (geometric like the 40% drop), j<=8
    ii, ff = np.where(B)
    reps = 3
    ii, ff = np.repeat(ii, reps), np.repeat(ff, reps)
    j = np.minimum(rng.geometric(.6, len(ii)), KMAX)
    keep = ff+j <= g.T-1
    ii, ff, j = ii[keep], ff[keep], j[keep]
    obs = S[ii, ff+j, :2] + rng.normal(0, g.SIG, (len(ii), 2))
    Ck = torch.tensor(kcond(P[ii, ff], im[ii], idd[ii], j, obs))
    Uk = torch.tensor(E[ii, ff])
    perm = torch.randperm(len(Ck), generator=torch.Generator().manual_seed(1))
    Ck, Uk = Ck[perm], Uk[perm]
    info = dict(bank_sec=t_bank, bank_replay_traj=replay, kick_examples=len(Ck))
    for K in (1, 3, 8):
        t1 = time.time()
        m = KickMix(K)
        info[f"kick{K}_val_nll"] = fit(m, (Ck, Uk), 15, 1e-3, 6200+K, log, f"kick K={K}")
        info[f"kick{K}_sec"] = time.time()-t1
        torch.save(m.state_dict(), HERE/f"gate6b_kick{K}.pt")
    info["prior_val_nll"] = float(.5*(Uk[:len(Uk)//10]**2).mean() + .5*LOG2PI)
    # always-on interval examples
    n_ex = 400_000
    i = rng.integers(0, M_BANK, n_ex)
    k = np.minimum(rng.geometric(.6, n_ex), KMAX)
    s = np.minimum(rng.integers(0, g.T-1, n_ex), g.T-1-k)
    obs = S[i, s+k, :2] + rng.normal(0, g.SIG, (n_ex, 2))
    Cg = torch.tensor(kcond(S[i, s], im[i], idd[i], k, obs))
    idx = s[:, None]+1+np.arange(KMAX)[None]
    Ug = torch.tensor(np.where(idx < g.T, E[i[:, None], np.minimum(idx, g.T-1)], rng.standard_normal((n_ex, KMAX))).astype(np.float32))
    t1 = time.time()
    gm = GapFlow()
    info["gap_val_nll"] = fit(gm, (Cg, Ug), 12, 1e-3, 6300, log, "gap flow")
    info["gap_prior_val_nll"] = float((.5*(Ug[:n_ex//10]**2).sum(-1)).mean() + KMAX/2*LOG2PI)
    info["gap_sec"] = time.time()-t1
    torch.save(gm.state_dict(), HERE/"gate6b_gapflow.pt")
    (HERE/"gate6b_train.json").write_text(json.dumps(info, indent=1)+"\n", encoding="utf-8")
    log(info)


def load_props():
    out = {}
    for K in (1, 3, 8):
        m = KickMix(K); m.load_state_dict(torch.load(HERE/f"gate6b_kick{K}.pt")); m.eval(); out[f"kick{K}"] = m
    gm = GapFlow(); gm.load_state_dict(torch.load(HERE/"gate6b_gapflow.pt")); gm.eval(); out["gap"] = gm
    return out


# ---------------------------------------------------------------- filter
def run_filter(case, init, kick, N, obs_end, seed, ours_model, props, move="none", resample_from=0, temp=1.):
    """init: prior|ours.  kick: prior | kick1|kick3|kick8 (selective: only particles that collide before the next observation)
    | kick{K}on (same 1-D mixture drawn for EVERY particle at every in-interval frame, all draws weighted, noise-space style)
    | gap (always-on 8-dim interval flow).
    Causality (v3 fix): proposal weights p/q that were conditioned on the NEXT observation are held in `pend` (carried with
    ancestors) and enter the weights only at that next observed frame, so estimate/score/resample at frame t use photos <= t only."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    C = len(case["obs"])
    obs, mask = case["obs"], case["mask"].copy()
    mask[:, obs_end:] = 0
    if init == "ours":
        st, im, idd, lpq0 = f.proposal_init(ours_model, case, N, 0)(rng)
        lpq0 = lpq0.copy()
    else:
        st = g.init_states(rng, C*N)
        im, idd = rng.integers(0, 12, C*N), rng.integers(0, 12, C*N)
        lpq0 = np.zeros((C, N))
    logw = lpq0.copy()
    comp = dict(static=lpq0.copy(), kick=np.zeros((C, N)), lik=np.zeros((C, N)))
    cumll = np.zeros((C, N))
    pend = np.zeros((C, N))                  # proposal weights waiting for their next observation
    cov_rows = []                            # (t, case idx, x lo/hi/truth, z lo/hi/truth, after_collision)
    nxt = np.full((C, g.T), 10**6)          # next observed frame strictly after f
    for c in range(C):
        o = np.where(mask[c] > 0)[0]
        for fr in range(g.T):
            later = o[o > fr]
            nxt[c, fr] = later[0] if len(later) else 10**6
    gap_eps = None                           # always-on proposal buffer for the current interval (C*N, KMAX), with start frame per case
    gap_start = np.zeros(C, int)
    fut = np.empty((C*N, g.T-obs_end, 2))
    filt, dec_prefix, dec_online, mh_acc = [], None, [], []
    n_prop = 0
    t0 = time.time()
    case_of = np.repeat(np.arange(C), N)

    def vdec():
        """Per-case variance over particles with finite total log-weight (outside-prior particles excluded), plus ESS."""
        fin = np.isfinite(logw)
        out = {}
        for k, v in dict(comp, total=logw).items():
            vv = []
            for c in range(C):
                x = v[c][fin[c]]
                vv.append(np.var(x) if len(x) > 1 else 0.)
            out[k] = float(np.mean(vv))
        w = softmax(logw, 1)
        out["ess"] = float(np.mean(1/np.square(w).sum(1)))
        return out

    def resample(cs, w):
        nonlocal st, im, idd, gap_eps
        u = (rng.uniform(size=(len(cs), 1))+np.arange(N))/N
        ixl = np.stack([np.minimum(np.searchsorted(w[c].cumsum(), uu), N-1) for c, uu in zip(cs, u)])
        ix = (ixl + cs[:, None]*N).ravel()
        sl = (cs[:, None]*N + np.arange(N)).ravel()
        st[sl], im[sl], idd[sl] = st[ix], im[ix], idd[ix]
        cumll[cs] = np.take_along_axis(cumll[cs], ixl, 1)
        lpq0[cs] = np.take_along_axis(lpq0[cs], ixl, 1)
        pend[cs] = np.take_along_axis(pend[cs], ixl, 1)
        if gap_eps is not None:      # unused future innovations: fresh engine-prior draws per descendant
            gap_eps[sl] = rng.standard_normal((len(sl), KMAX))
        logw[cs] = 0.
        for v in comp.values():
            v[cs] = 0.

    def mh_move(t):
        nonlocal st, im, idd
        if init == "ours":
            st2, im2, id2, lpq2 = f.proposal_init(ours_model, case, N, 0)(rng)
        else:
            st2, im2, id2 = g.init_states(rng, C*N), rng.integers(0, 12, C*N), rng.integers(0, 12, C*N)
            lpq2 = np.zeros((C, N))
        ll2 = -.5*np.square(obs[:, None, 0]-st2[:, 1:3].reshape(C, N, 2)).sum(-1)*mask[:, 0, None]/g.SIG**2
        for tt in range(1, t+1):
            o2, b2 = step_pre([g.POOL[a][c] for a, c in zip(im2, id2)], st2)
            o2[:, 5] = np.where(b2, o2[:, 5]*np.exp(g.NOISE*rng.standard_normal(len(o2))), o2[:, 5])
            st2 = o2
            ll2 += -.5*np.square(obs[:, None, tt]-st2[:, 1:3].reshape(C, N, 2)).sum(-1)*mask[:, tt, None]/g.SIG**2
        la = (lpq2 + ll2/temp) - (lpq0 + cumll/temp)
        acc = (np.log(rng.uniform(size=la.shape)) < la).ravel()
        st[acc], im[acc], idd[acc] = st2[acc], im2[acc], id2[acc]
        cumll.ravel()[acc] = ll2.ravel()[acc]
        lpq0.ravel()[acc] = lpq2.ravel()[acc]
        pend.ravel()[acc] = 0.                 # replacement path used engine-prior kicks only
        mh_acc.append(float(acc.mean()))

    for t in range(g.T):
        dnew = 0.
        if t > 0:
            pre, bounce = step_pre([g.POOL[a][c] for a, c in zip(im, idd)], st)
            u = rng.standard_normal(len(pre))                              # fresh engine-prior kicks per particle
            if kick.startswith("kick") and t < obs_end:
                tn = nxt[case_of, t]
                inwin = (tn < obs_end) & (tn-t <= KMAX)
                sel = inwin if kick.endswith("on") else bounce & inwin   # on: every particle; else only collisions before next obs
                if sel.any():
                    pid = np.where(sel)[0]
                    j = tn[pid]-t
                    cnd = torch.tensor(kcond(pre[pid][:, [1, 2, 4, 5, 6]].astype(np.float32), im[pid], idd[pid], j,
                                             obs[case_of[pid], tn[pid]].astype(np.float32)))
                    m = props[kick.removesuffix("on")]
                    x = m.sample(cnd, gen)
                    pri = torch.rand(len(x), generator=gen) < BETA
                    x[pri] = torch.randn(int(pri.sum()), generator=gen)
                    with torch.no_grad():
                        lq = m.log_prob(cnd, x).numpy().astype(float)
                    x = x.numpy().astype(float)
                    lp = -.5*x**2 - .5*LOG2PI
                    lq = np.logaddexp(np.log(1-BETA)+lq, np.log(BETA)+lp)
                    u[pid] = x                                                # used only where bounce (see below)
                    d = np.zeros(C*N); d[pid] = lp-lq
                    dnew = d.reshape(C, N)                                    # joins pend AFTER this frame's flush
                    n_prop += len(pid)
            if kick == "gap" and t < obs_end:
                rows = np.where(bounce)[0]
                if gap_eps is not None:
                    jj = t - gap_start[case_of] - 1
                    ok = (jj >= 0) & (jj < KMAX)
                    u = np.where(ok, gap_eps[np.arange(len(u)), np.clip(jj, 0, KMAX-1)], u)
            pre[:, 5] = np.where(bounce, pre[:, 5]*np.exp(g.NOISE*u), pre[:, 5])
            st = pre
        if t >= obs_end:
            fut[:, t-obs_end] = st[:, 1:3]
            continue
        p = st[:, 1:3].reshape(C, N, 2)
        obs_t = mask[:, t] > 0
        if obs_t.any():                       # the photo this pending proposal looked at has now arrived
            logw[obs_t] += pend[obs_t]; comp["kick"][obs_t] += pend[obs_t]; pend[obs_t] = 0.
        pend += dnew
        ll = -.5*np.square(obs[:, None, t]-p).sum(-1)*mask[:, t, None]/g.SIG**2
        cumll += ll
        logw += ll/temp
        comp["lik"] += ll/temp
        if t >= g.CUT and obs_t.any():
            w = softmax(logw, 1)
            cs = np.where(obs_t)[0]
            truth = case["pos"][cs, t]
            mp = np.einsum("cn,cnd->cd", w[cs], p[cs])
            dr = np.stack([np.minimum(np.searchsorted(ww.cumsum(), rng.uniform(size=64)), N-1) for ww in w[cs]])
            smp = np.take_along_axis(p[cs], dr[..., None], 1)
            dist = lambda a, b: np.sqrt(np.square(a-b).sum(-1))
            filt.append((t, np.sqrt(np.square(mp-truth).sum(-1)), dist(smp, truth[:, None]).mean(1)-.5*dist(smp[:, :32], smp[:, 32:]).mean(1)))
            for k, j in (("x", 0), ("z", 1)):
                o = np.argsort(p[cs, :, j], 1)
                fs, cw = np.take_along_axis(p[cs, :, j], o, 1), np.take_along_axis(w[cs], o, 1).cumsum(1)
                lo = fs[np.arange(len(cs)), np.minimum((cw < .1).sum(1), N-1)]
                hi = fs[np.arange(len(cs)), np.minimum((cw < .9).sum(1), N-1)]
                cov_rows.append((k, t, lo, hi, truth[:, j], case["bnc"][cs, max(0, t-3):t+1].any(1)))
            dec_online.append(vdec())
        if t == g.CUT-1:
            dec_prefix = vdec()
        if t >= resample_from:
            w = softmax(logw, 1)
            ess = 1/np.square(w).sum(1)
            res = np.where((ess < N/2) & (obs_t | (t == obs_end-1)))[0]   # v3: unobserved resample only at forecast start
            if len(res):
                resample(res, w)
        if move == "mh" and t == g.CUT-1:
            mh_move(t)
        if kick == "gap" and t < obs_end-1:
            # always-on: at every observed frame (and frame 0) propose innovations for the interval up to the next observation
            anchors = np.where(obs_t | (t == 0))[0]
            if len(anchors):
                if gap_eps is None:
                    gap_eps = rng.standard_normal((C*N, KMAX))
                rows_c, ks, ob = [], [], []
                for c in anchors:
                    tn = nxt[c, t]
                    if tn < obs_end and tn-t <= KMAX:
                        rows_c.append(c); ks.append(tn-t); ob.append(obs[c, tn])
                for c in anchors:
                    gap_start[c] = t
                    gap_eps[c*N:(c+1)*N] = rng.standard_normal((N, KMAX))   # fresh per descendant
                if rows_c:
                    rows_c = np.array(rows_c)
                    pid = (rows_c[:, None]*N + np.arange(N)).ravel()
                    cnd = torch.tensor(kcond(st[pid][:, [1, 2, 4, 5, 6]].astype(np.float32), im[pid], idd[pid],
                                             np.repeat(ks, N), np.repeat(np.array(ob), N, 0).astype(np.float32)))
                    x = props["gap"].sample(cnd, gen)
                    pri = torch.rand(len(x), generator=gen) < BETA
                    x[pri] = torch.randn(int(pri.sum()), KMAX, generator=gen)
                    with torch.no_grad():
                        lq = props["gap"].log_prob(x, cnd).numpy().astype(float)
                    x = x.numpy().astype(float)
                    lp = -.5*(x**2).sum(-1) - KMAX/2*LOG2PI
                    lq = np.logaddexp(np.log(1-BETA)+lq, np.log(BETA)+lp)
                    gap_eps[pid] = x
                    d = (lp-lq).reshape(len(rows_c), N)
                    pend[rows_c] += d
                    n_prop += len(pid)
    assert not np.any(pend), "pending proposal weights left after the last observation"
    sec = time.time()-t0
    global LAST_FILT
    LAST_FILT = filt
    fut = fut.reshape(C, N, g.T-obs_end, 2)
    w = softmax(logw, 1)
    truth = case["pos"][:, obs_end:]
    dr = np.stack([np.minimum(np.searchsorted(ww.cumsum(), rng.uniform(size=128)), N-1) for ww in w])
    smp = np.take_along_axis(fut, dr[..., None, None], 1)
    dist = lambda a, b: np.sqrt(np.square(a-b).mean((-1, -2)))
    out = dict(forecast_energy=dist(smp, truth[:, None]).mean(1)-.5*dist(smp[:, :64], smp[:, 64:]).mean(1), sec=np.full(C, sec/C))
    st_iv = a6.interval_stats(a6.intervals(fut, logw), case["pos"])
    for k in ("x", "z"):
        for q in ("cover", "width", "iscore"):
            out[f"{k}_{q}"] = np.full(1, st_iv[k][q])
    for lo_t in (25, 30, 35):
        sel = [x for x in filt if lo_t <= x[0] < lo_t+5]
        if sel:
            out[f"filt_rmse_{lo_t}"] = np.concatenate([x[1] for x in sel])
            out[f"filt_energy_{lo_t}"] = np.concatenate([x[2] for x in sel])
    if filt:
        out["filter_energy"] = np.concatenate([x[2] for x in filt])
    for name, dd in (("prefix", dec_prefix), ("online", None)):
        if name == "prefix" and dd:
            for k, v in dd.items():
                out[f"var_prefix_{k}"] = np.full(1, v)
    if dec_online:
        for k in dec_online[0]:
            out[f"var_online_{k}"] = np.full(1, np.mean([d[k] for d in dec_online]))
    out["proposals_per_case"] = np.full(1, n_prop/C)
    for k in ("x", "z"):                      # online 80% interval coverage: all observed frames >= 25 / within 3 frames after a true collision
        rr = [r for r in cov_rows if r[0] == k]
        if rr:
            lo, hi, y, ac = (np.concatenate([r[i] for r in rr]) for i in (2, 3, 4, 5))
            cv = (y >= lo-1e-9) & (y <= hi+1e-9)
            out[f"track_{k}_cover"], out[f"track_{k}_width"] = np.full(1, cv.mean()), np.full(1, (hi-lo).mean())
            if ac.any():
                out[f"aftercol_{k}_cover"], out[f"aftercol_{k}_width"] = np.full(1, cv[ac].mean()), np.full(1, (hi-lo)[ac].mean())
                out[f"aftercol_n"] = np.full(1, ac.sum())
    if mh_acc:
        out["mh_accept"] = np.full(1, np.mean(mh_acc))
    return out


def causal_check(log, kb="kick3"):
    """Future-photo invariance (astra recheck design + a full-mask variant): changing only photos at frames >= 30 must leave every
    online score at frames < 30 bit-identical, for every arm.  Returns dict; caller aborts on any failure."""
    props = load_props()
    model = a6.load_models()[8100]
    base = cases_full(62681, 48, "shift")
    sparse = dict(base, mask=base["mask"].copy()); sparse["mask"][:, 25:] = 0; sparse["mask"][:, [25, 30]] = 1
    res, ok = {}, True
    for name, case in (("astra_25_30", sparse), ("full_mask", base)):
        changed = dict(case, obs=case["obs"].copy()); changed["obs"][:, 30:, 1] += .10
        for kick, move in (("prior", "none"), (kb, "none"), (kb+"on", "none"), ("gap", "none"), (kb, "mh")):
            got = []
            for c in (case, changed):
                run_filter(c, "ours", kick, 256, 40, 62900, model, props, move=move, resample_from=24)
                got.append([(t, a_, b_) for t, a_, b_ in LAST_FILT if t < 30])
            n = sum(len(x[1]) for x in got[0])
            bad = sum(int(np.sum((x[1] != y[1]) | (x[2] != y[2]))) for x, y in zip(got[0], got[1]))
            same_frames = [x[0] for x in got[0]] == [y[0] for y in got[1]]
            res[f"{name}|{kick}|{move}"] = dict(scores=n, changed=bad, same_frames=same_frames)
            ok &= (bad == 0) and same_frames and n > 0
            log("causal", name, kick, move, res[f"{name}|{kick}|{move}"])
    res["pass"] = bool(ok)
    return res


def mean(d):
    return {k: float(np.mean(v)) for k, v in d.items()}


def pick_rule(log):
    """Choose ONE resampling start frame for all arms on dev (prior & ours init, engine-prior kicks), S1 forecast energy."""
    props = load_props()
    models = a6.load_models()
    dev = cases_full(55000, 48, "shift")
    res = {}
    for rf in (0, g.CUT-1):
        e = [mean(run_filter(dev, "prior", "prior", 256, g.CUT, 11, None, props, resample_from=rf))["forecast_energy"]]
        e += [mean(run_filter(dev, "ours", "prior", 256, g.CUT, 11, models[8100], props, resample_from=rf))["forecast_energy"]]
        res[rf] = dict(prior=e[0], ours=e[1], mean=float(np.mean(e)))
        log("rule", rf, res[rf])
    kick = {}
    for K in (1, 3, 8):
        kick[K] = mean(run_filter(dev, "ours", f"kick{K}", 256, 40, 11, models[8100], props,
                                  resample_from=min(res, key=lambda r: res[r]["mean"])))["filter_energy"]
        log("kick dev", K, kick[K])
    out = dict(resample_from=int(min(res, key=lambda r: res[r]["mean"])), rule_dev=res, kick_dev=kick, kick_best=int(min(kick, key=kick.get)))
    (HERE/"gate6b_rule.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    log(out)


def run(log):
    props = load_props()
    models = a6.load_models()
    rule = json.loads((HERE/"gate6b_rule.json").read_text(encoding="utf-8"))
    rf, kb = rule["resample_from"], f"kick{rule['kick_best']}"
    chk = causal_check(log, kb)
    if not chk["pass"]:
        raise SystemExit("future-photo invariance FAILED: " + json.dumps(chk))
    tests = [cases_full(s, 48, "shift") for s in range(56000, 56004)]
    plan = [("prior", "prior", None, "none"), ("prior", kb, None, "none"), ("prior", "gap", None, "none")]
    for s in (8100, 8200, 8300):
        plan += [("ours", "prior", s, "none"), ("ours", kb, s, "none"), ("ours", kb+"on", s, "none"), ("ours", "gap", s, "none"),
                 ("ours", kb, s, "mh")]
    rows = {}
    for setting, obs_end in (("S1", g.CUT), ("S2", 40)):
        for init, kick, s, move in plan:
            N = 172 if move == "mh" else 256
            key = f"{setting}|{init}|{kick}|{move}|N{N}|{s}"
            c0 = a6.COUNT["rows"]; t0 = time.time()
            per = [mean(run_filter(c, init, kick, N, obs_end, 600+i, models.get(s), props, move=move, resample_from=rf)) for i, c in enumerate(tests)]
            r = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            r["per_seed"] = {k: [p[k] for p in per] for k in per[0] if k in ("forecast_energy", "filter_energy", "track_x_cover", "track_z_cover")}
            r["test_traj_actual"] = (a6.COUNT["rows"]-c0)*g.SUB/a6.TRAJ_STEPS
            r["test_wall_sec"] = time.time()-t0
            rows[key] = r
            log(key, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
            (HERE/"gate6b.json").write_text(json.dumps(dict(rule=rule, causal_check=chk, rows=rows), indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    dict(train=train, rule=pick_rule, run=run, check=lambda lg: lg(json.dumps(causal_check(lg))))[sys.argv[1]](log)
