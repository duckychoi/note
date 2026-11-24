"""#38 stage 0 (PREREG_38s0.md): is there headroom for posterior (Bayes) planning over MAP-material planning
in the occluded 2-D contact world of pf36_g3?  Compute only, no model training.
Run with the fm34 venv python (MuJoCo), PF_NTH=8.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

os.environ.setdefault("PF_NTH", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pf36_20261123"))
import pf36_g3 as P  # noqa: E402

T, CUT, SIG = P.T, P.CUT, P.SIG
ACTIONS = np.array([-1.5, -.75, 0., .75, 1.5])
WIN, N_PF, M, N_TRUE, TEMP = .10, 2048, 512, 64, 4.0
SEEDS = (38000, 38001, 38002, 38003)
D_LW = [1, 2, 4, 5, 6]


def make_cases(seed, n):
    rng = np.random.default_rng(seed)
    mu = np.where(rng.uniform(size=n) < .5, rng.uniform(.08, .22, n), rng.uniform(.9, 1.2, n))
    d = np.where(rng.uniform(size=n) < .5, rng.uniform(.07, .17, n), rng.uniform(.55, .8, n))
    models = [P.mk(a, b) for a, b in zip(mu, d)]
    st = P.init_states(rng, n)
    pos = np.empty((n, CUT, 2)); pos[:, 0] = st[:, 1:3]
    for t in range(1, CUT):
        st, _ = P.step_frame(models, st, rng)
        pos[:, t] = st[:, 1:3]
    obs = pos + rng.normal(0, SIG, pos.shape)
    mask = (rng.uniform(size=(n, CUT)) > .4).astype(float)
    a_r = rng.integers(0, len(ACTIONS), n)
    final = rollout_final([models], st.copy(), a_r, rng)
    return dict(mu=mu, d=d, models=models, state=st, obs=obs, mask=mask, target=final, a_rand=a_r)


def rollout_final(models_list, st, a_idx, rng):
    """Apply kick ACTIONS[a_idx] at frame 25 to states, roll 25 frames with bounce noise, return final x."""
    models = models_list[0] if len(models_list) == 1 else models_list
    st = st.copy(); st[:, 4] += ACTIONS[a_idx]
    for _ in range(T - CUT):
        st, _ = P.step_frame(models, st, rng)
    return st[:, 1]


def pf_to_cut(case, rng, mask=None):
    """pf36_g3.run_pf_full filtering part, stopped at frame CUT-1. Returns states, material idx, normalized weights."""
    C, N = len(case["obs"]), N_PF
    st = P.init_states(rng, C * N)
    im, idd = rng.integers(0, 12, C * N), rng.integers(0, 12, C * N)
    logw = np.zeros((C, N))
    obs = case["obs"]; mask = case["mask"] if mask is None else mask
    for t in range(CUT):
        if t > 0:
            st, _ = P.step_frame([P.POOL[a][c] for a, c in zip(im, idd)], st, rng)
        p = st[:, 1:3].reshape(C, N, 2)
        logw += -.5 * np.square(obs[:, None, t] - p).sum(-1) * mask[:, t, None] / (SIG**2 * TEMP)
        if t == CUT - 1:
            break
        w = softmax(logw, 1); ess = 1 / np.square(w).sum(1)
        res = np.where(ess < N / 2)[0]
        if len(res):
            u = (rng.uniform(size=(len(res), 1)) + np.arange(N)) / N
            ix = np.stack([np.minimum(np.searchsorted(w[c].cumsum(), uu), N - 1) for c, uu in zip(res, u)]) + res[:, None] * N
            sl = (res[:, None] * N + np.arange(N)).ravel(); ix = ix.ravel()
            X = st[:, D_LW].reshape(C, N, 5)[res]; wr = w[res]
            mean = np.einsum("cn,cnd->cd", wr, X); dev = X - mean[:, None]
            Lc = np.linalg.cholesky(np.einsum("cn,cnd,cne->cde", wr, dev, dev) + 1e-10 * np.eye(5))
            st[sl], im[sl], idd[sl] = st[ix], im[ix], idd[ix]
            Xr = st[sl][:, D_LW].reshape(len(res), N, 5)
            a = np.sqrt(1 - P.LW_H**2)
            Xr = a * Xr + (1 - a) * mean[:, None] + P.LW_H * np.einsum("cde,cne->cnd", Lc, rng.standard_normal(Xr.shape))
            tmp = st[sl]; tmp[:, D_LW] = Xr.reshape(-1, 5); st[sl] = tmp
            st[sl, 2] = np.maximum(st[sl, 2], P.R)
            n = len(sl)
            im[sl] = np.clip(im[sl] + (rng.uniform(size=n) < .15) * rng.choice([-1, 1], n), 0, 11)
            idd[sl] = np.clip(idd[sl] + (rng.uniform(size=n) < .15) * rng.choice([-1, 1], n), 0, 11)
            logw[res] = 0.
    return st.reshape(C, N, 7), im.reshape(C, N), idd.reshape(C, N), softmax(logw, 1)


def success_probs(states, models, target, rng):
    """states (C,M,7), models list per particle (C*M) -> (C, A) success fraction."""
    C, Mm = states.shape[:2]
    out = np.empty((C, len(ACTIONS)))
    flat = states.reshape(-1, 7)
    tgt = np.repeat(target, Mm)
    for k in range(len(ACTIONS)):
        fx = rollout_final(models if isinstance(models, list) and len(models) == C * Mm else [models], flat, np.full(len(flat), k), rng)
        out[:, k] = (np.abs(fx - tgt) <= WIN).reshape(C, Mm).mean(1)
    return out


def evaluate(case, rng, mask=None):
    C = len(case["obs"])
    st, im, idd, w = pf_to_cut(case, rng, mask)
    # resample M particles per case
    u = (rng.uniform(size=(C, 1)) + np.arange(M)) / M
    ix = np.stack([np.minimum(np.searchsorted(w[c].cumsum(), u[c]), N_PF - 1) for c in range(C)])
    S_ = np.take_along_axis(st, ix[..., None], 1); I_ = np.take_along_axis(im, ix, 1); J_ = np.take_along_axis(idd, ix, 1)
    cell = im * 12 + idd
    mass = np.stack([np.bincount(cell[c], weights=w[c], minlength=144) for c in range(C)])
    mapc = mass.argmax(1); mapmass = mass.max(1)
    pb = success_probs(S_, [P.POOL[a][b] for a, b in zip(I_.ravel(), J_.ravel())], case["target"], rng)
    pm = success_probs(S_, [P.POOL[c // 12][c % 12] for c in np.repeat(mapc, M)], case["target"], rng)
    pk = success_probs(S_, [m for m in case["models"] for _ in range(M)], case["target"], rng)
    true_st = np.repeat(case["state"][:, None], N_TRUE, 1)
    ps = success_probs(true_st, [m for m in case["models"] for _ in range(N_TRUE)], case["target"], rng)
    pick = lambda p: p.argmax(1)
    val = lambda a: ps[np.arange(C), a]
    return dict(bayes=val(pick(pb)), map=val(pick(pm)), known=val(pick(pk)), full=ps.max(1),
                differ=(pick(pb) != pick(pm)).astype(float), mapmass=mapmass)


def boot(x, seed=5):
    r = np.random.default_rng(seed).choice(x, (5000, len(x))).mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def main():
    t0 = time.time(); rows = []; rows_v = []
    for s in SEEDS:
        case = make_cases(s, 48)
        rows.append(evaluate(case, np.random.default_rng(s + 1)))
        rows_v.append(evaluate(case, np.random.default_rng(s + 2), mask=np.ones_like(case["mask"])))
        print(s, "done", round(time.time() - t0), "s", flush=True)
    cat = lambda rs, k: np.concatenate([r[k] for r in rs])
    amb = cat(rows, "mapmass") < .5
    out = dict(n=int(len(amb)), n_ambiguous=int(amb.sum()))
    for name, sel in (("all", np.ones_like(amb)), ("ambiguous", amb)):
        d = {k: float(cat(rows, k)[sel].mean()) for k in ("bayes", "map", "known", "full", "differ")}
        bm = (cat(rows, "bayes") - cat(rows, "map"))[sel]
        d["bayes_minus_map"] = float(bm.mean()); d["bayes_minus_map_ci"] = boot(bm) if sel.sum() > 1 else None
        d["full_minus_bayes"] = d["full"] - d["bayes"]
        voi = (cat(rows_v, "bayes") - cat(rows, "bayes"))[sel]
        d["voi_unoccluded_bayes"] = float(voi.mean()); d["voi_ci"] = boot(voi) if sel.sum() > 1 else None
        out[name] = d
        print(name, json.dumps(d), flush=True)
    a = out["ambiguous"]
    out["proceed_signal"] = bool(a["bayes_minus_map"] >= .03 and a["bayes_minus_map_ci"] and a["bayes_minus_map_ci"][0] > 0)
    out["seconds"] = time.time() - t0
    print("PROCEED", out["proceed_signal"], flush=True)
    (HERE / "gate38s0.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
