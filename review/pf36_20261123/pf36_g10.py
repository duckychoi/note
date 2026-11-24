"""#36 gate 10: planning from the filter belief (occluded 2-D ball task, as #38 stage 0b) and a cheap probe stage.
(a) at frame 24 pick a horizontal kick from a fine grid to land the ball within +-0.10 m of x* at frame 54 (30 frames later).
    Beliefs at EQUAL simulation budget (N particles x 24 frames): flow-proposal IS (ours, ckpt 8100) vs bootstrap PF (#38 0b
    machine: Liu-West + material walk) vs point estimate (posterior-mean state + MAP material).  Temperature of each belief chosen on dev.
(b) probe-then-act vs act-now under the flow belief: gentle nudge at frame 24, observe frames 25-29 (all frames), then kick at frame 29
    with 25 frames left (grid widened by 30/25 so full-information success stays high).  Probe = fixed (0, wait-and-look) or chosen
    from 5 nudges by expected information gain about the particle identity (Monte-Carlo mutual information over belief particles).
Value of a chosen action = TRUE expected success from 64 rollouts at the true state & true material.
See PREREG.md (관문 10).  Usage: python pf36_g10.py dev | run"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "salvage_20261123"))
import gate38s0 as S0  # noqa: E402  (bootstrap PF of #38 stage 0/0b)

torch.set_num_threads(16)
HERE = Path(__file__).resolve().parent
T, CUT, SIG = g.T, g.CUT, g.SIG
H_NOW, PW = 30, 5
H_LATE = H_NOW - PW
ACT_NOW = np.round(np.linspace(-1.5, 1.5, 31), 6)            # 0.1 m/s spacing: shift <= 0.1 x 0.6 s = 0.06 m <= half window 0.10 m
ACT_LATE = np.round(np.linspace(-1.8, 1.8, 37), 6)           # same spacing, range x 30/25
PROBES = np.array([-.5, -.25, 0., .25, .5])
WIN, M, N_TRUE, N_BEL = .10, 256, 64, 2048
D_LW = [1, 2, 4, 5, 6]
PF_ROWS = [0]                                                 # particle-frames simulated (belief only)


def roll(models, st, kick, nfr, rng):
    st = st.copy(); st[:, 4] += kick
    for _ in range(nfr):
        st, _ = g.step_frame(models, st, rng)
    return st


def succ(states, models, target, acts, nfr, rng):
    C, m = states.shape[:2]; flat = states.reshape(-1, 7); tgt = np.repeat(target, m)
    out = np.empty((C, len(acts)))
    for k, a in enumerate(acts):
        fx = roll(models, flat, a, nfr, rng)[:, 1]
        out[:, k] = (np.abs(fx - tgt) <= WIN).reshape(C, m).mean(1)
    return out


def make_cases(seed, n):
    rng = np.random.default_rng(seed)
    mu = np.where(rng.uniform(size=n) < .5, rng.uniform(.08, .22, n), rng.uniform(.9, 1.2, n))
    d = np.where(rng.uniform(size=n) < .5, rng.uniform(.07, .17, n), rng.uniform(.55, .8, n))
    models = [g.mk(a, b) for a, b in zip(mu, d)]
    st = g.init_states(rng, n)
    pos = np.empty((n, CUT, 2)); pos[:, 0] = st[:, 1:3]
    for t in range(1, CUT):
        st, _ = g.step_frame(models, st, rng)
        pos[:, t] = st[:, 1:3]
    obs = pos + rng.normal(0, SIG, pos.shape)
    mask = (rng.uniform(size=(n, CUT)) > .4).astype(float)
    target = roll(models, st, rng.uniform(-1.5, 1.5, n), H_NOW, rng)[:, 1]
    probe = {}
    for p in PROBES:                                         # one realised probe trajectory per nudge (common seed per case block)
        r2 = np.random.default_rng([seed, int(round(p*100)) + 1000])
        sp = st.copy(); sp[:, 4] += p; pp = np.empty((n, PW, 2))
        for k in range(PW):
            sp, _ = g.step_frame(models, sp, r2); pp[:, k] = sp[:, 1:3]
        probe[float(p)] = dict(obs=pp + r2.normal(0, SIG, pp.shape), state=sp)
    return dict(mu=mu, d=d, models=models, state=st, obs=obs, mask=mask, target=target, probe=probe)


# ---------------------------------------------------------------- beliefs at frame 24
def belief_flow(case, model, N, temp, rng):
    C = len(case["obs"])
    st, im, idd, lpq = f.proposal_init(model, dict(obs=case["obs"], mask=case["mask"]), N, 0)(rng)
    mods = [g.POOL[a][b] for a, b in zip(im, idd)]
    ll = -.5*np.square(case["obs"][:, None, 0]-st[:, 1:3].reshape(C, N, 2)).sum(-1)*case["mask"][:, 0, None]/SIG**2
    for t in range(1, CUT):
        st, _ = g.step_frame(mods, st, rng)
        ll += -.5*np.square(case["obs"][:, None, t]-st[:, 1:3].reshape(C, N, 2)).sum(-1)*case["mask"][:, t, None]/SIG**2
    PF_ROWS[0] += C*N*(CUT-1)
    return st.reshape(C, N, 7), im.reshape(C, N), idd.reshape(C, N), lpq + ll/temp


def belief_boot(case, N, temp, rng):
    S0.N_PF, S0.TEMP = N, temp
    st, im, idd, w = S0.pf_to_cut(dict(obs=case["obs"], mask=case["mask"]), rng)
    PF_ROWS[0] += len(case["obs"])*N*(CUT-1)
    with np.errstate(divide="ignore"):
        return st, im, idd, np.log(w)


def resample(st, im, idd, logw, m, rng):
    C, N = logw.shape
    w = softmax(logw, 1)
    u = (rng.uniform(size=(C, 1)) + np.arange(m))/m
    ix = np.stack([np.minimum(np.searchsorted(w[c].cumsum(), u[c]), N-1) for c in range(C)])
    return np.take_along_axis(st, ix[..., None], 1), np.take_along_axis(im, ix, 1), np.take_along_axis(idd, ix, 1)


def point(st, im, idd, logw, m):
    C, N = logw.shape
    w = softmax(logw, 1)
    mean = np.einsum("cn,cnd->cd", w, st)
    cell = im*12 + idd
    mass = np.stack([np.bincount(cell[c], weights=w[c], minlength=144) for c in range(C)])
    mc = mass.argmax(1)
    return np.repeat(mean[:, None], m, 1), np.repeat((mc//12)[:, None], m, 1), np.repeat((mc % 12)[:, None], m, 1)


def plan(S, I, J, target, acts, nfr, rng):
    p = succ(S, [g.POOL[a][b] for a, b in zip(I.ravel(), J.ravel())], target, acts, nfr, rng)
    return p.argmax(1)


def true_value(case, rng):
    C = len(case["obs"])
    tm = lambda m: [mm for mm in case["models"] for _ in range(m)]
    now = succ(np.repeat(case["state"][:, None], N_TRUE, 1), tm(N_TRUE), case["target"], ACT_NOW, H_NOW, rng)
    late = {p: succ(np.repeat(case["probe"][p]["state"][:, None], N_TRUE, 1), tm(N_TRUE), case["target"], ACT_LATE, H_LATE, rng)
            for p in case["probe"]}
    return now, late


def eig(S, I, J, p, rng):
    """MC mutual information between the 5 probe observations and the particle identity (belief particles S: (C,M,7))."""
    C, m = S.shape[:2]
    sp = S.reshape(-1, 7).copy(); sp[:, 4] += p
    mods = [g.POOL[a][b] for a, b in zip(I.ravel(), J.ravel())]
    Y = np.empty((C*m, PW, 2))
    for k in range(PW):
        sp, _ = g.step_frame(mods, sp, rng); Y[:, k] = sp[:, 1:3]
    Y = Y.reshape(C, m, -1)
    y = Y + rng.normal(0, SIG, Y.shape)
    out = np.empty(C)
    for c in range(C):
        d2 = ((y[c][:, None]-Y[c][None])**2).sum(-1)/(2*SIG**2)          # (i, j)
        out[c] = np.mean(-np.diag(d2) - (logsumexp(-d2, 1) - np.log(m)))
    return out


def continue_probe(st, im, idd, logw, case, p, rng):
    C, N = logw.shape
    s = st.reshape(-1, 7).copy(); s[:, 4] += p
    mods = [g.POOL[a][b] for a, b in zip(im.ravel(), idd.ravel())]
    lw = logw.copy()
    po = case["probe"][p]["obs"]
    for k in range(PW):
        s, _ = g.step_frame(mods, s, rng)
        lw += -.5*np.square(po[:, None, k]-s[:, 1:3].reshape(C, N, 2)).sum(-1)/SIG**2
    return s.reshape(C, N, 7), lw


def evaluate(case, model, temps, rng, boot_big=False, probe_stage=True):
    C = len(case["obs"])
    now, late = true_value(case, rng)
    val_now = lambda a: now[np.arange(C), a]
    out = dict(full_now=now.max(1), full_late={str(p): late[p].max(1) for p in late})
    secs, rows = {}, {}
    beliefs = {}
    arms = [("flow", lambda: belief_flow(case, model, N_BEL, temps["flow"], rng)),
            ("boot", lambda: belief_boot(case, N_BEL, temps["boot"], rng))]
    if boot_big:
        arms.append(("boot4x", lambda: belief_boot(case, 4*N_BEL, temps["boot"], rng)))
    for name, fn in arms:
        r0, t0 = PF_ROWS[0], time.time()
        beliefs[name] = fn()
        secs[f"belief_{name}"] = time.time()-t0; rows[f"belief_{name}"] = PF_ROWS[0]-r0
        t0 = time.time()
        S, I, J = resample(*beliefs[name], M, rng)
        out[f"{name}_plan"] = val_now(plan(S, I, J, case["target"], ACT_NOW, H_NOW, rng))
        secs[f"plan_{name}"] = time.time()-t0
        if name in ("flow", "boot"):
            S, I, J = point(*beliefs[name], M)
            out[f"{name}_point"] = val_now(plan(S, I, J, case["target"], ACT_NOW, H_NOW, rng))
    if probe_stage:
        st, im, idd, lw = beliefs["flow"]
        S, I, J = resample(st, im, idd, lw, M, rng)
        t0 = time.time()
        gains = np.stack([eig(S, I, J, p, rng) for p in PROBES], 1)
        secs["eig"] = time.time()-t0
        choice = PROBES[gains.argmax(1)]                                   # secondary: all 5 candidates
        ok = np.abs(PROBES) <= .25 + 1e-9                                  # primary: {0, +-.25} (dev precondition >= 95%)
        choice_p = PROBES[ok][gains[:, ok].argmax(1)]
        out["eig_choice"] = choice; out["eig_choice_primary"] = choice_p; out["eig_gain"] = gains
        val = {}
        for p in sorted(set([0.]) | set(choice.tolist()) | set(choice_p.tolist())):
            s2, lw2 = continue_probe(st, im, idd, lw, case, p, rng)
            S2, I2, J2 = resample(s2, im, idd, lw2, M, rng)
            a = plan(S2, I2, J2, case["target"], ACT_LATE, H_LATE, rng)
            val[p] = late[p][np.arange(C), a]
        out["probe_fixed0"] = val[0.]
        out["probe_eig_all5"] = np.array([val[p][c] for c, p in enumerate(choice)])
        out["full_probe_eig_all5"] = np.array([late[p][c].max() for c, p in enumerate(choice)])
        out["probe_eig"] = np.array([val[p][c] for c, p in enumerate(choice_p)])
        out["full_probe_eig"] = np.array([late[p][c].max() for c, p in enumerate(choice_p)])
    return out, secs, rows


def boot_ci(x, seed=5):
    r = np.random.default_rng(seed).choice(x, (5000, len(x))).mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def dev(log):
    model = a6.load_models()[8100]
    case = make_cases(38200, 48)
    res = {}
    for t in (1., 2., 4.):
        o, s, r = evaluate(case, model, dict(flow=t, boot=t), np.random.default_rng(1), probe_stage=False)
        res[t] = dict(flow=float(o["flow_plan"].mean()), boot=float(o["boot_plan"].mean()))
        log("dev temp", t, res[t])
    o, s, r = evaluate(case, model, dict(flow=1., boot=4.), np.random.default_rng(2), probe_stage=True)
    full_probe = {p: float(np.mean(v)) for p, v in o["full_late"].items()}
    out = dict(temps=res, temp_flow=max(res, key=lambda t: res[t]["flow"]), temp_boot=max(res, key=lambda t: res[t]["boot"]),
               full_now=float(o["full_now"].mean()), full_late=full_probe)
    log(out)
    (HERE/"gate10_dev.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")


def run(log):
    model = a6.load_models()[8100]
    d = json.loads((HERE/"gate10_dev.json").read_text(encoding="utf-8"))
    temps = dict(flow=d["temp_flow"], boot=d["temp_boot"])
    allo, alls, allr = [], [], []
    for s in range(38300, 38304):
        case = make_cases(s, 48)
        o, sec, rows = evaluate(case, model, temps, np.random.default_rng(s+7), boot_big=True)
        allo.append(o); alls.append(sec); allr.append(rows)
        log(s, {k: round(float(np.mean(v)), 4) for k, v in o.items() if isinstance(v, np.ndarray) and v.ndim == 1}, sec)
    cat = lambda k: np.concatenate([o[k] for o in allo])
    keys = [k for k in allo[0] if isinstance(allo[0][k], np.ndarray) and allo[0][k].ndim == 1 and k not in ("eig_choice", "eig_choice_primary")]
    res = dict(temps=temps, n=int(len(cat("flow_plan"))), mean={k: float(cat(k).mean()) for k in keys})
    pairs = {"flow_plan-boot_plan": ("flow_plan", "boot_plan"), "flow_plan-boot4x_plan": ("flow_plan", "boot4x_plan"),
             "flow_plan-flow_point": ("flow_plan", "flow_point"), "boot_plan-boot_point": ("boot_plan", "boot_point"),
             "probe_eig-flow_plan": ("probe_eig", "flow_plan"), "probe_fixed0-flow_plan": ("probe_fixed0", "flow_plan"),
             "probe_eig-probe_fixed0": ("probe_eig", "probe_fixed0"),
             "probe_eig_all5-flow_plan": ("probe_eig_all5", "flow_plan"), "probe_eig_all5-probe_fixed0": ("probe_eig_all5", "probe_fixed0")}
    res["paired"] = {k: dict(mean=float((cat(a)-cat(b)).mean()), ci=boot_ci(cat(a)-cat(b)),
                             blocks=[float((o[a]-o[b]).mean()) for o in allo]) for k, (a, b) in pairs.items()}
    res["full_probe_late_mean"] = {p: float(np.mean(np.concatenate([o["full_late"][p] for o in allo]))) for p in allo[0]["full_late"]}
    res["eig_choice_counts"] = {str(p): int((cat("eig_choice") == p).sum()) for p in PROBES}
    res["eig_choice_primary_counts"] = {str(p): int((cat("eig_choice_primary") == p).sum()) for p in PROBES}
    res["sec"] = {k: float(np.sum([s[k] for s in alls])) for k in alls[0]}
    res["belief_particle_frames"] = {k: int(np.sum([r[k] for r in allr])) for k in allr[0]}
    res["pass_a"] = bool(res["paired"]["flow_plan-boot_plan"]["ci"][0] > 0)
    log(json.dumps(res, indent=1))
    (HERE/"gate10.json").write_text(json.dumps(res, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    dict(dev=dev, run=run)[sys.argv[1]](log)
