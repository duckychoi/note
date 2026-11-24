"""#36 gate 6A: ledger audit, NPE/IS/resample/jitter ablation, coverage diagnostics, split conformal. See PREREG.md (관문 6A).
Usage: python pf36_g6a.py ledger | ablate | diag | conformal"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import softmax  # noqa: E402

import pf36_g3 as g  # noqa: E402
import pf36_g4b as f  # noqa: E402
import pf36_g5 as g5  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = (8100, 8200, 8300)
COUNT = {"rows": 0}
_orig_step = g.step_frame


def _counted(models, states, rng, noise=True, chunk=200_000):
    COUNT["rows"] += len(states)
    return _orig_step(models, states, rng, noise, chunk)


g.step_frame = _counted


def steps():
    return COUNT["rows"]*g.SUB


TRAJ_STEPS = (g.T-1)*g.SUB  # mj_steps in one 50-frame trajectory


def load_models():
    sd = torch.load(HERE/"gate5_primary_models.pt")
    out = {}
    for s in SEEDS:
        m = f.Proposal()
        m.load_state_dict(sd[f"147456_{s}"])
        m.eval()
        out[s] = m
    return out


def cases_custom(seed, n, kind):
    rng = np.random.default_rng(seed)
    if kind == "grid":
        mu, d = g.PF_MU[rng.integers(0, 12, n)], g.PF_D[rng.integers(0, 12, n)]
    elif kind == "shift":
        mu = np.where(rng.uniform(size=n) < .5, rng.uniform(.08, .22, n), rng.uniform(.9, 1.2, n))
        d = np.where(rng.uniform(size=n) < .5, rng.uniform(.07, .17, n), rng.uniform(.55, .8, n))
    else:
        mu, d = rng.uniform(.3, .8, n), rng.uniform(.2, .5, n)
    pos, bnc = g.simulate([g.mk(a, b) for a, b in zip(mu, d)], g.init_states(rng, n), rng, g.T)
    obs = pos + rng.normal(0, g.SIG, pos.shape)
    mask = np.zeros((n, g.T))
    mask[:, :g.CUT] = rng.uniform(size=(n, g.CUT)) > .4
    return dict(pos=pos, bnc=bnc, obs=obs, mask=mask, mu=mu, d=d)


def concat(cs):
    return {k: np.concatenate([c[k] for c in cs]) for k in cs[0]}


def intervals(fut, logw):
    w = softmax(logw, 1)
    out = {}
    for k, j in (("x", 0), ("z", 1)):
        fin = fut[:, :, -1, j]
        o = np.argsort(fin, 1)
        fs, cw = np.take_along_axis(fin, o, 1), np.take_along_axis(w, o, 1).cumsum(1)
        lo = fs[np.arange(len(fs)), np.minimum((cw < .1).sum(1), fs.shape[1]-1)]
        hi = fs[np.arange(len(fs)), np.minimum((cw < .9).sum(1), fs.shape[1]-1)]
        out[k] = (lo, hi)
    return out


def interval_stats(iv, y):
    out = {}
    for k, j in (("x", 0), ("z", 1)):
        lo, hi = iv[k]
        t = y[:, -1, j]
        below, above = t < lo-1e-9, t > hi+1e-9
        out[k] = dict(cover=float(np.mean(~below & ~above)), width=float(np.mean(hi-lo)), miss_low=float(below.mean()),
                      miss_high=float(above.mean()),
                      iscore=float(np.mean(hi-lo + 10*np.clip(lo-t, 0, None) + 10*np.clip(t-hi, 0, None))))
    return out


def run6(case, model, N, temp, mode, seed, alpha=.1, xthr=.18):
    """mode: npe (alpha 0, equal weights) | is (weights, no resample) | resample (final, no jitter) | jitter (current)."""
    old = f.ALPHA
    f.ALPHA = 0. if mode in ("npe", "is_noalpha") else alpha
    try:
        rng = np.random.default_rng(seed)
        C = len(case["obs"])
        st, im, idd, logw = f.proposal_init(model, case, N, 0)(rng)
        logw = np.zeros((C, N)) if mode in ("npe", "npe_mix") else logw.copy()
    finally:
        f.ALPHA = old
    obs, mask = case["obs"], case["mask"]
    fut = np.empty((C*N, g.T-g.CUT, 2))
    nb = np.zeros(C*N, int)
    diag = {}
    for t in range(g.T):
        if t > 0:
            st, b = g.step_frame([g.POOL[a][c] for a, c in zip(im, idd)], st, rng)
            if t >= g.CUT:
                nb += b
        if t >= g.CUT:
            fut[:, t-g.CUT] = st[:, 1:3]
            continue
        if mode not in ("npe", "npe_mix"):
            p = st[:, 1:3].reshape(C, N, 2)
            logw += -.5*np.square(obs[:, None, t]-p).sum(-1)*mask[:, t, None]/(g.SIG**2*temp)
        if t == g.CUT-1:
            w = softmax(logw, 1)
            ess = 1/np.square(w).sum(1)
            cls = (im*12+idd).reshape(C, N)
            ncls = []
            for c in range(C):
                m = np.bincount(cls[c], weights=w[c], minlength=144)
                srt = np.sort(m)[::-1].cumsum()
                ncls.append(int(np.searchsorted(srt, .99)+1))
            diag = dict(ess=ess, ncls=np.array(ncls, float))
            if mode in ("resample", "jitter"):
                res = np.where(ess < N/2)[0]
                if len(res):
                    u = (rng.uniform(size=(len(res), 1))+np.arange(N))/N
                    ix = (np.stack([np.minimum(np.searchsorted(w[c].cumsum(), uu), N-1) for c, uu in zip(res, u)]) + res[:, None]*N).ravel()
                    sl = (res[:, None]*N + np.arange(N)).ravel()
                    D = [1, 2, 4, 5, 6]
                    if mode == "jitter":
                        X = st[:, D].reshape(C, N, 5)[res]
                        wr = w[res]
                        mean = np.einsum("cn,cnd->cd", wr, X)
                        dev = X-mean[:, None]
                        Lc = np.linalg.cholesky(np.einsum("cn,cnd,cne->cde", wr, dev, dev) + 1e-10*np.eye(5))
                    st[sl], im[sl], idd[sl] = st[ix], im[ix], idd[ix]
                    if mode == "jitter":
                        n = len(sl)
                        a = np.sqrt(1-g.LW_H**2)
                        Xr = st[sl][:, D].reshape(len(res), N, 5)
                        Xr = a*Xr + (1-a)*mean[:, None] + g.LW_H*np.einsum("cde,cne->cnd", Lc, rng.standard_normal(Xr.shape))
                        tmp = st[sl]
                        tmp[:, D] = Xr.reshape(-1, 5)
                        st[sl] = tmp
                        st[sl, 2] = np.maximum(st[sl, 2], g.R)
                        im[sl] = np.clip(im[sl] + (rng.uniform(size=n) < .15)*rng.choice([-1, 1], n), 0, 11)
                        idd[sl] = np.clip(idd[sl] + (rng.uniform(size=n) < .15)*rng.choice([-1, 1], n), 0, 11)
                    logw[res] = 0.
            x24 = st[:, 1].copy()
    fut = fut.reshape(C, N, g.T-g.CUT, 2)
    fb = (nb >= 2).reshape(C, N)
    fx = (fut[:, :, -1, 0]-x24.reshape(C, N)) > xthr
    sc = g.score_weighted(fut, fb, fx, logw, case, xthr, np.random.default_rng(seed+1))
    return sc, intervals(fut, logw), diag


def chunked(case, fn, size):
    n = len(case["obs"])
    outs = [fn({k: v[s:s+size] for k, v in case.items()}) for s in range(0, n, size)]
    sc = {k: np.concatenate([o[0][k] for o in outs]) for k in outs[0][0]}
    iv = {k: (np.concatenate([o[1][k][0] for o in outs]), np.concatenate([o[1][k][1] for o in outs])) for k in ("x", "z")}
    dg = {k: np.concatenate([o[2][k] for o in outs]) for k in outs[0][2]} if outs[0][2] else {}
    return sc, iv, dg


def mean(d):
    return {k: float(np.mean(v)) for k, v in d.items()}


def base_setup():
    tests = [g.make_cases(s, 48, test=True) for s in range(51200, 51204)]
    tr = g.library(g.TRPOOL, 8, 8, 196_608, 52001)
    xthr = float(np.median(tr[0][:, -1, 0]-tr[0][:, g.CUT-1, 0]))
    return tests, xthr


# ------------------------------------------------------------------ A1
def ledger(log):
    L = {}
    t = time.time(); c0 = steps()
    val = g.make_cases(51000, 48, test=False)
    tests = [g.make_cases(s, 48, test=True) for s in range(51200, 51204)]
    L["cases(ground truth, not a method cost)"] = dict(steps=steps()-c0, sec=time.time()-t)
    t = time.time(); c0 = steps()
    tr = g.library(g.TRPOOL, 8, 8, 196_608, 52001)
    L["train-range bank 196,608 (xthr only, uncharged)"] = dict(steps=steps()-c0, sec=time.time()-t)
    xthr = float(np.median(tr[0][:, -1, 0]-tr[0][:, g.CUT-1, 0]))
    t = time.time(); c0 = steps()
    broad = g.library(g.POOL, 12, 12, 196_608, 52003)
    L["broad bank 196,608 generated (flow uses first 147,456; retrieval uses all)"] = dict(steps=steps()-c0, sec=time.time()-t)
    t = time.time()
    model, info = g5.train5(broad, 147_456, 8100, g5.chosen_cfg())
    L["flow training (seed 8100)"] = dict(steps=0, sec=time.time()-t)
    t = time.time(); c0 = steps()
    for tp in (1, 2, 4):
        g5.pf_eval(model, val, 256, tp, xthr, 3)
    L["flow-PF val temperature tuning 3x48xN256"] = dict(steps=steps()-c0, sec=time.time()-t)
    t = time.time(); c0 = steps()
    for i, c in enumerate(tests):
        g5.pf_eval(model, c, 256, 1, xthr, 500+256+i)
    L["flow-PF test inference 192xN256"] = dict(steps=steps()-c0, sec=time.time()-t, per_case_sec=(time.time()-t)/192)
    # PF-8192 reference (gate 3): val tuning was done at N=1024 x 3 temps; test at N=8192
    t = time.time(); c0 = steps()
    for tp in (1, 2, 4):
        g.run_pf_full(val, 1024, tp, xthr, 3)
    L["PF-8192 ref: val temperature tuning (3x48xN1024)"] = dict(steps=steps()-c0, sec=time.time()-t)
    t = time.time(); c0 = steps()
    sub = {k: v[:12] for k, v in tests[0].items()}
    for j in range(0, 48, 12):
        g.run_pf_full({k: v[j:j+12] for k, v in tests[0].items()}, 8192, 4, xthr, 100+8192+j)
    sec = time.time()-t
    L["PF-8192 ref: test inference measured on 48 cases (x4 for 192)"] = dict(steps=(steps()-c0)*4, sec=sec*4, per_case_sec=sec/48)
    # retrieval
    import pf36_g4 as h
    t = time.time()
    lw1 = g.lib_logw(broad[0], val, 1.)
    for e in h.ESS_T:
        h.score_from_lw(broad, h.adaptive_lw(lw1, e)[0], val, xthr, 1)
    L["retrieval val tuning (no sims)"] = dict(steps=0, sec=time.time()-t)
    t = time.time()
    for c in tests:
        lw1 = g.lib_logw(broad[0], c, 1.)
        h.score_from_lw(broad, h.adaptive_lw(lw1, 8)[0], c, xthr, 7)
    L["retrieval test inference 192"] = dict(steps=0, sec=time.time()-t, per_case_sec=(time.time()-t)/192)
    for v in L.values():
        v["traj_equiv"] = v["steps"]/TRAJ_STEPS
    b = L["broad bank 196,608 generated (flow uses first 147,456; retrieval uses all)"]
    flow_used_bank = dict(steps=b["steps"]*147_456/196_608, sec=b["sec"]*147_456/196_608)
    flow = dict(
        charged_in_gate5=147_456+256*192,
        actual_min=(flow_used_bank["steps"]+L["flow-PF test inference 192xN256"]["steps"])/TRAJ_STEPS,
        actual_with_tuning=(flow_used_bank["steps"]+L["flow-PF test inference 192xN256"]["steps"]+L["flow-PF val temperature tuning 3x48xN256"]["steps"])/TRAJ_STEPS,
        actual_as_scripted=(b["steps"]+L["flow-PF test inference 192xN256"]["steps"]+L["flow-PF val temperature tuning 3x48xN256"]["steps"])/TRAJ_STEPS,
        cold_sec=flow_used_bank["sec"]+L["flow training (seed 8100)"]["sec"]+L["flow-PF val temperature tuning 3x48xN256"]["sec"]+L["flow-PF test inference 192xN256"]["sec"],
        warm_per_case_sec=L["flow-PF test inference 192xN256"]["per_case_sec"])
    ref = dict(charged_in_gate3=8192*192,
               actual_with_tuning=(L["PF-8192 ref: test inference measured on 48 cases (x4 for 192)"]["steps"]+L["PF-8192 ref: val temperature tuning (3x48xN1024)"]["steps"])/TRAJ_STEPS,
               actual_test_only=L["PF-8192 ref: test inference measured on 48 cases (x4 for 192)"]["steps"]/TRAJ_STEPS,
               cold_sec=L["PF-8192 ref: val temperature tuning (3x48xN1024)"]["sec"]+L["PF-8192 ref: test inference measured on 48 cases (x4 for 192)"]["sec"],
               warm_per_case_sec=L["PF-8192 ref: test inference measured on 48 cases (x4 for 192)"]["per_case_sec"])
    ret = dict(actual=b["traj_equiv"], cold_sec=b["sec"]+L["retrieval val tuning (no sims)"]["sec"]+L["retrieval test inference 192"]["sec"],
               warm_per_case_sec=L["retrieval test inference 192"]["per_case_sec"])
    fixed_flow = flow["cold_sec"]-192*flow["warm_per_case_sec"]
    fixed_ref = ref["cold_sec"]-192*ref["warm_per_case_sec"]
    breakeven = (fixed_flow-fixed_ref)/max(ref["warm_per_case_sec"]-flow["warm_per_case_sec"], 1e-9)
    out = dict(traj_steps=TRAJ_STEPS, stages=L, flow=flow, ref_pf8192=ref, retrieval=ret,
               ratio_sims_ref_over_flow=dict(charged=ref["charged_in_gate3"]/flow["charged_in_gate5"],
                                             actual_min=ref["actual_test_only"]/flow["actual_min"],
                                             actual_with_tuning=ref["actual_with_tuning"]/flow["actual_with_tuning"],
                                             as_scripted=ref["actual_with_tuning"]/flow["actual_as_scripted"]),
               ratio_cold_time_ref_over_flow=ref["cold_sec"]/flow["cold_sec"], breakeven_cases_time=breakeven)
    (HERE/"gate6a_ledger.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    log(json.dumps({k: v for k, v in out.items() if k != "stages"}, indent=1))


# ------------------------------------------------------------------ A2
def train_cut_head(log):
    """Direct cutoff-state + material head: regenerate the first 147,456 broad-bank trajectories (same rng stream) keeping frame-24 state."""
    rng = np.random.default_rng(52003)
    n = 196_608
    im, idd = rng.integers(0, 12, n), rng.integers(0, 12, n)
    st = g.init_states(rng, n)
    models = [g.POOL[a][b] for a, b in zip(im, idd)]
    pos = np.empty((n, g.T, 2))
    pos[:, 0] = st[:, 1:3]
    cut = None
    for t in range(1, g.T):
        st, _ = g.step_frame(models, st, rng)
        pos[:, t] = st[:, 1:3]
        if t == g.CUT-1:
            cut = st[:, [1, 2, 4, 5, 6]].copy()
    M = 147_456
    lo, hi = np.percentile(cut[:M], .5, 0), np.percentile(cut[:M], 99.5, 0)
    old = (f.CEN, f.HW, f.LOG_HW)
    f.CEN, f.HW = (lo+hi)/2, (hi-lo)/2
    f.LOG_HW = float(np.log(f.HW).sum())
    lib = (pos, None, np.concatenate([np.zeros((n, 1)), cut[:, :2], np.zeros((n, 1)), cut[:, 2:]], 1), im, idd)
    t0 = time.time()
    model, info = g5.train5(lib, M, 8100, g5.chosen_cfg())
    log("cut head trained", info["epochs"], info["best_val_nll"], time.time()-t0)
    return model, (f.CEN, f.HW, f.LOG_HW), old


def run_cut(model, norm, case, xthr, seed, N=256):
    old = (f.CEN, f.HW, f.LOG_HW)
    f.CEN, f.HW, f.LOG_HW = norm
    try:
        C = len(case["obs"])
        x = torch.tensor(f.features(case["obs"], case["mask"]))
        cls, z = model.sample(x, N, torch.Generator().manual_seed(seed))
    finally:
        f.CEN, f.HW, f.LOG_HW = old
    cls, z = cls.numpy().ravel(), z.numpy().reshape(-1, 5).astype(float)
    st = np.zeros((C*N, 7))
    st[:, [1, 2, 4, 5, 6]] = z
    st[:, 2] = np.maximum(st[:, 2], g.R)
    rng = np.random.default_rng(seed)
    pos, bnc = g.simulate([g.POOL[a][b] for a, b in zip(cls//12, cls % 12)], st, rng, g.T-g.CUT+1)
    fut = pos[:, 1:].reshape(C, N, g.T-g.CUT, 2)
    fb = (bnc[:, 1:].sum(1) >= 2).reshape(C, N)
    fx = (fut[:, :, -1, 0]-z[:, 0].reshape(C, N)) > xthr
    lw = np.zeros((C, N))
    return g.score_weighted(fut, fb, fx, lw, case, xthr, np.random.default_rng(seed+1)), intervals(fut, lw), {}


def ablate(log):
    tests, xthr = base_setup()
    models = load_models()
    test = concat(tests)
    rows = {}
    for s, m in models.items():
        for mode in ("npe", "is", "resample", "jitter"):
            t = time.time()
            per = []
            for i, c in enumerate(tests):
                sc, iv, dg = run6(c, m, 256, 1, mode, 500+256+i, xthr=xthr)
                per.append(dict(**mean(sc), ess=float(np.mean(dg["ess"])), ncls=float(np.mean(dg["ncls"])),
                                **{f"{k}_{q}": v for k, d in interval_stats(iv, c["pos"]).items() for q, v in d.items()}))
            r = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            r["energy_per_seed"] = [p["energy"] for p in per]
            r["sec"] = time.time()-t
            rows[f"{s}|{mode}"] = r
            log(s, mode, {k: round(v, 4) for k, v in r.items() if not isinstance(v, list)})
    cut_model, norm, old = train_cut_head(log)
    f.CEN, f.HW, f.LOG_HW = old
    per = []
    t = time.time()
    for i, c in enumerate(tests):
        sc, iv, _ = run_cut(cut_model, norm, c, xthr, 900+i)
        per.append(dict(**mean(sc), **{f"{k}_{q}": v for k, d in interval_stats(iv, c["pos"]).items() for q, v in d.items()}))
    r = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    r["energy_per_seed"] = [p["energy"] for p in per]
    r["sec"] = time.time()-t
    rows["8100|cut_head_npe"] = r
    log("cut head", {k: round(v, 4) for k, v in r.items() if not isinstance(v, list)})
    agg = {}
    for mode in ("npe", "is", "resample", "jitter"):
        rs = [rows[f"{s}|{mode}"] for s in SEEDS]
        agg[mode] = {k: [float(np.mean([x[k] for x in rs])), float(np.std([x[k] for x in rs], ddof=1))]
                     for k in ("energy", "brier_bounce2", "brier_xpast", "cover_x", "cover_z", "x_width", "z_width", "x_iscore", "z_iscore", "ess", "ncls", "sec")}
    cur, isr = agg["jitter"], agg["is"]
    stop = bool(abs(isr["energy"][0]/cur["energy"][0]-1) <= .02 and abs(isr["cover_x"][0]-cur["cover_x"][0]) <= .03
                and abs(isr["cover_z"][0]-cur["cover_z"][0]) <= .03 and isr["sec"][0] < cur["sec"][0])
    out = dict(rows=rows, aggregate=agg, stop_rule_triggered=stop)
    (HERE/"gate6a_ablate.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    log("aggregate", json.dumps(agg), "stop", stop)


def ablate2(log):
    """Split defensive prior mixture vs IS weighting (2x2: alpha {0,.1} x weights {equal, IS}); no resampling."""
    tests, xthr = base_setup()
    models = load_models()
    rows = {}
    for s, m in models.items():
        for mode in ("npe", "npe_mix", "is_noalpha", "is"):
            per = []
            for i, c in enumerate(tests):
                sc, iv, dg = run6(c, m, 256, 1, mode, 500+256+i, xthr=xthr)
                per.append(dict(**mean(sc), **{f"{k}_{q}": v for k, d in interval_stats(iv, c["pos"]).items() for q, v in d.items()}))
            rows[f"{s}|{mode}"] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            log(s, mode, {k: round(v, 4) for k, v in rows[f"{s}|{mode}"].items()})
    agg = {mode: {k: [float(np.mean([rows[f"{s}|{mode}"][k] for s in SEEDS])), float(np.std([rows[f"{s}|{mode}"][k] for s in SEEDS], ddof=1))]
                  for k in ("energy", "cover_x", "cover_z", "x_iscore", "z_iscore")} for mode in ("npe", "npe_mix", "is_noalpha", "is")}
    (HERE/"gate6a_ablate2.json").write_text(json.dumps(dict(rows=rows, aggregate=agg), indent=1)+chr(10), encoding="utf-8")
    log(json.dumps(agg))


# ------------------------------------------------------------------ A3
def diag(log):
    _, xthr = base_setup()
    models = load_models()
    dev = {"in": concat([cases_custom(s, 48, "in") for s in (53000, 53001)]),
           "shift": concat([cases_custom(s, 48, "shift") for s in (53100, 53101)])}
    grid = cases_custom(53200, 48, "grid")
    rows = []
    for dname, case in dev.items():
        for s, m in models.items():
            for N in (256, 1024):
                for tp in (1, 2, 4):
                    for mode in ("is", "resample", "jitter"):
                        sc, iv, dg = chunked(case, lambda c: run6(c, m, N, tp, mode, 7000+N+tp, xthr=xthr), 96 if N == 256 else 48)
                        st = interval_stats(iv, case["pos"])
                        r = dict(dev=dname, seed=s, N=N, temp=tp, mode=mode, **mean(sc), ess=float(np.median(dg["ess"])),
                                 ncls=float(np.median(dg["ncls"])), **{f"{k}_{q}": v for k, d in st.items() for q, v in d.items()})
                        rows.append(r)
            log(dname, s, "done")
    ref = []
    for dname, case in (("grid", grid), ("shift", {k: v[:48] for k, v in dev["shift"].items()})):
        for s, m in models.items():
            sc, iv, dg = chunked(case, lambda c: run6(c, m, 16384, 1, "is", 7777, xthr=xthr), 6)
            st = interval_stats(iv, case["pos"])
            ref.append(dict(dev=dname, seed=s, N=16384, temp=1, mode="is", **mean(sc), ess=float(np.median(dg["ess"])),
                            ncls=float(np.median(dg["ncls"])), **{f"{k}_{q}": v for k, d in st.items() for q, v in d.items()}))
            log("ref", dname, s, {k: round(v, 3) for k, v in ref[-1].items() if isinstance(v, float)})
    (HERE/"gate6a_diag.json").write_text(json.dumps(dict(rows=rows, reference=ref), indent=1)+"\n", encoding="utf-8")


# ------------------------------------------------------------------ A4
def conformal(log, mode, N=256, tp=1):
    tests, xthr = base_setup()
    models = load_models()
    cal = concat([cases_custom(s, 48, "shift") for s in range(54000, 54004)])
    test = concat(tests)
    out = {}
    for s, m in models.items():
        _, ivc, _ = chunked(cal, lambda c: run6(c, m, N, tp, mode, 8000, xthr=xthr), 96)
        _, ivt, _ = chunked(test, lambda c: run6(c, m, N, tp, mode, 8100, xthr=xthr), 96)
        raw = interval_stats(ivt, test["pos"])
        adj = {}
        qs = {}
        for k, j in (("x", 0), ("z", 1)):
            lo, hi = ivc[k]
            y = cal["pos"][:, -1, j]
            sc = np.maximum(lo-y, y-hi)
            n = len(sc)
            q = float(np.sort(sc)[min(int(np.ceil((n+1)*.8))-1, n-1)])
            qs[k] = q
            adj[k] = (ivt[k][0]-q, ivt[k][1]+q)
        cal_st = interval_stats(adj, test["pos"])
        out[s] = dict(q=qs, raw=raw, calibrated=cal_st)
        log(s, "q", qs, "raw", raw, "cal", cal_st)
    # reference PF-8192 raw intervals on the same test set (gate 3 config: temp 4, Liu-West)
    ref_iv = {"x": ([], []), "z": ([], [])}
    for i, c in enumerate(tests):
        for j in range(0, 48, 12):
            sub = {k: v[j:j+12] for k, v in c.items()}
            fut_iv = _ref_intervals(sub, xthr, 100+8192+i*10+j)
            for k in ("x", "z"):
                ref_iv[k][0].append(fut_iv[k][0])
                ref_iv[k][1].append(fut_iv[k][1])
    ref_iv = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in ref_iv.items()}
    out["reference_pf8192_raw"] = interval_stats(ref_iv, test["pos"])
    log("ref", out["reference_pf8192_raw"])
    (HERE/"gate6a_conformal.json").write_text(json.dumps(dict(mode=mode, N=N, temp=tp, results=out), indent=1)+"\n", encoding="utf-8")


def _ref_intervals(case, xthr, seed):
    """Re-run gate-3 PF-8192 and capture intervals (copy of run_pf_full's sampler, returning intervals)."""
    cap = {}
    orig = g.score_weighted

    def grab(fut, fb, fx, logw, case_, xthr_, rng, S=128):
        cap["iv"] = intervals(fut, logw)
        return orig(fut, fb, fx, logw, case_, xthr_, rng, S)
    g.score_weighted = grab
    try:
        g.run_pf_full(case, 8192, 4, xthr, seed)
    finally:
        g.score_weighted = orig
    return cap["iv"]


if __name__ == "__main__":
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    cmd = sys.argv[1]
    if cmd == "conformal":
        conformal(log, sys.argv[2])
    else:
        dict(ledger=ledger, ablate=ablate, ablate2=ablate2, diag=diag)[cmd](log)
