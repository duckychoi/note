"""#38 stage 0b (PREREG_38s0b.md): principled fine action grid, gap split material vs state, probe vs extra observation.
Compute only.  fm34 venv python, PF_NTH=7.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "7")
import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402

import gate38s0 as S0  # noqa: E402  (pf_to_cut, same PF as stage 0)

P = S0.P
T, CUT, SIG = P.T, P.CUT, P.SIG
ACT = np.round(np.linspace(-1.5, 1.5, 31), 6)
WIN, M, N_TRUE, PROBE, PW = .10, 256, 64, .75, 5
SEEDS = (38100, 38101, 38102, 38103)
HERE = Path(__file__).resolve().parent


def roll(models, st, kick, nfr, rng):
    st = st.copy(); st[:, 4] += kick
    for _ in range(nfr):
        st, _ = P.step_frame(models, st, rng)
    return st


def succ(states, models, target, nfr, rng):
    """states (C,m,7), models list len C*m -> (C, 31) success fraction."""
    C, m = states.shape[:2]; flat = states.reshape(-1, 7); tgt = np.repeat(target, m)
    out = np.empty((C, len(ACT)))
    for k, a in enumerate(ACT):
        fx = roll(models, flat, a, nfr, rng)[:, 1]
        out[:, k] = (np.abs(fx - tgt) <= WIN).reshape(C, m).mean(1)
    return out


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
    kick = rng.uniform(-1.5, 1.5, n)
    target = roll(models, st, kick, T - CUT, rng)[:, 1]
    # probe truth: probe kick at frame 24, 5 observed frames, true state at frame 29
    sp = st.copy(); sp[:, 4] += PROBE; ppos = np.empty((n, PW, 2))
    for f in range(PW):
        sp, _ = P.step_frame(models, sp, rng); ppos[:, f] = sp[:, 1:3]
    return dict(models=models, state=st, obs=obs, mask=mask, target=target,
                probe_obs=ppos + rng.normal(0, SIG, ppos.shape), probe_state=sp)


def resample(w, m, rng):
    C, N = w.shape
    u = (rng.uniform(size=(C, 1)) + np.arange(m)) / m
    return np.stack([np.minimum(np.searchsorted(w[c].cumsum(), u[c]), N - 1) for c in range(C)])


def mods(I, J):
    return [P.POOL[a][b] for a, b in zip(I.ravel(), J.ravel())]


def evaluate(case, rng, full_obs=False, probe=True):
    C = len(case["obs"])
    mask = np.ones_like(case["mask"]) if full_obs else None
    st, im, idd, w = S0.pf_to_cut(case, rng, mask)
    true_m = lambda m: [x for x in case["models"] for _ in range(m)]
    ix = resample(w, M, rng)
    Sx = np.take_along_axis(st, ix[..., None], 1); I_ = np.take_along_axis(im, ix, 1); J_ = np.take_along_axis(idd, ix, 1)
    cell = im * 12 + idd
    mass = np.stack([np.bincount(cell[c], weights=w[c], minlength=144) for c in range(C)])
    mapc = mass.argmax(1)
    map_m = [P.POOL[c // 12][c % 12] for c in np.repeat(mapc, M)]
    tgt, nf = case["target"], T - CUT
    ps = succ(np.repeat(case["state"][:, None], N_TRUE, 1), true_m(N_TRUE), tgt, nf, rng)
    val = lambda p: ps[np.arange(C), p.argmax(1)]
    out = dict(mapmass=mass.max(1), full=ps.max(1), bayes=val(succ(Sx, mods(I_, J_), tgt, nf, rng)))
    if full_obs:
        return out
    out["map"] = val(succ(Sx, map_m, tgt, nf, rng))
    out["known"] = val(succ(Sx, true_m(M), tgt, nf, rng))
    mean = np.einsum("cn,cnd->cd", w, st)
    pt = np.repeat(mean[:, None], M, 1)
    out["point_known"] = val(succ(pt, true_m(M), tgt, nf, rng))
    out["point_map"] = val(succ(pt, map_m, tgt, nf, rng))
    if probe:  # probe kick on all N particles, then observe (or not) frames 25-29
        N = st.shape[1]; flat = st.reshape(-1, 7).copy(); flat[:, 4] += PROBE
        mm = mods(im, idd); lw = np.log(np.maximum(w, 1e-300))
        for f in range(PW):
            flat, _ = P.step_frame(mm, flat, rng)
            lw_obs = -.5 * np.square(case["probe_obs"][:, None, f] - flat[:, 1:3].reshape(C, N, 2)).sum(-1) / (SIG**2 * S0.TEMP)
            lw = lw + lw_obs if f == 0 else lw + lw_obs
        st29 = flat.reshape(C, N, 7)
        nf2 = T - CUT - PW
        pp = succ(np.repeat(case["probe_state"][:, None], N_TRUE, 1), true_m(N_TRUE), tgt, nf2, rng)
        vp = lambda p: pp[np.arange(C), p.argmax(1)]
        for name, ww in (("probe_obs", softmax(lw, 1)), ("probe_blind", w)):
            j = resample(ww, M, rng)
            Sj = np.take_along_axis(st29, j[..., None], 1)
            out[name] = vp(succ(Sj, mods(np.take_along_axis(im, j, 1), np.take_along_axis(idd, j, 1)), tgt, nf2, rng))
        out["probe_full"] = pp.max(1)
    return out


def boot(x, seed=5):
    r = np.random.default_rng(seed).choice(x, (5000, len(x))).mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def main():
    t0 = time.time(); rows, rows_v = [], []
    for s in SEEDS:
        case = make_cases(s, 48)
        rows.append(evaluate(case, np.random.default_rng(s + 1)))
        rows_v.append(evaluate(case, np.random.default_rng(s + 2), full_obs=True))
        print(s, "done", round(time.time() - t0), "s", flush=True)
    cat = lambda rs, k: np.concatenate([r[k] for r in rs])
    mm = cat(rows, "mapmass")
    out = dict(n=int(len(mm)), actions=ACT.tolist(), per_case={k: cat(rows, k).tolist() for k in rows[0]},
               per_case_unoccluded={k: cat(rows_v, k).tolist() for k in rows_v[0]})
    keys = ("bayes", "map", "known", "point_known", "point_map", "full", "probe_obs", "probe_blind", "probe_full")
    for name, sel in (("all", mm >= 0), ("ambiguous", mm < .5), ("ambiguous_03", mm < .3)):
        d = dict(n=int(sel.sum()), **{k: float(cat(rows, k)[sel].mean()) for k in keys})
        if sel.sum() < 2:
            out[name] = d; continue
        diff = lambda a, b: (cat(rows, a) - cat(rows, b))[sel]
        d.update(bayes_minus_map=float(diff("bayes", "map").mean()), bayes_minus_map_ci=boot(diff("bayes", "map")),
                 material_part=float(diff("known", "bayes").mean()), material_ci=boot(diff("known", "bayes")),
                 state_part=float(diff("full", "known").mean()), state_ci=boot(diff("full", "known")),
                 known_minus_point_known=float(diff("known", "point_known").mean()), kpk_ci=boot(diff("known", "point_known")),
                 bayes_minus_point_map=float(diff("bayes", "point_map").mean()), bpm_ci=boot(diff("bayes", "point_map")),
                 probe_value=float(diff("probe_obs", "probe_blind").mean()), probe_ci=boot(diff("probe_obs", "probe_blind")))
        vo = (cat(rows_v, "bayes") - cat(rows, "bayes"))[sel]
        d.update(extra_obs_value=float(vo.mean()), extra_obs_ci=boot(vo))
        out[name] = d
        print(name, json.dumps(d), flush=True)
    a = out["ambiguous"]
    out["proceed_signal"] = bool(a.get("bayes_minus_map", 0) >= .03 and a["bayes_minus_map_ci"][0] > 0)
    out["seconds"] = time.time() - t0
    print("PROCEED", out["proceed_signal"], flush=True)
    (HERE / "gate38s0b.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
