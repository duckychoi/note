"""#34 v3 gate 0: build two matched-speed/cadence HalfCheetah gaits and test whether a held-out
contact-aligned cycle MMD separates them beyond a same-gait noise floor. See PREREG.md."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"verification_20261123"))
from coordination34 import features  # noqa: E402  (import only; never run that folder's scripts)

SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
P = 12            # gait period in env steps (0.6 s), fixed for both gaits
H_SEARCH, H_EVAL, TAIL = 192, 384, 256
NTHREAD = 16
BTHIGH, FTHIGH = 0, 3


class Sim:
    def __init__(self):
        self.env = gym.make("HalfCheetah-v5")
        self.m = self.env.unwrapped.model
        self.skip = self.env.unwrapped.frame_skip
        self.dt = self.env.unwrapped.dt
        self.data = [mujoco.MjData(self.m) for _ in range(NTHREAD)]
        self.engine = Rollout(nthread=NTHREAD)
        self.d1 = mujoco.MjData(self.m)
        self.floor = 0
        self.bfoot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "bfoot")
        self.ffoot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ffoot")

    def start(self, seed):
        self.env.reset(seed=seed)
        x = np.zeros(mujoco.mj_stateSize(self.m, SPEC))
        mujoco.mj_getState(self.m, self.env.unwrapped.data, x, SPEC)
        return x

    def actions(self, params, H, phase0, delta):
        """params (n,6,5): c, a1, psi1, a2, psi2 per joint. delta fixes fthigh psi1 = bthigh psi1 + delta."""
        p = params.reshape(-1, 6, 5).copy()
        if delta is not None:
            p[:, FTHIGH, 2] = p[:, BTHIGH, 2] + delta
        phi = 2*np.pi*np.arange(H)/P + phase0
        c, a1, s1, a2, s2 = (p[..., k][:, None, :] for k in range(5))
        u = c + a1*np.cos(phi[None, :, None] + s1) + a2*np.cos(2*phi[None, :, None] + s2)
        return np.tanh(u)

    def run(self, params, seed, H, delta):
        start = self.start(seed)
        phase0 = np.random.default_rng(seed).uniform(0, 2*np.pi)
        act = self.actions(params, H, phase0, delta)
        states, _ = self.engine.rollout(self.m, self.data, start[None], control=np.repeat(act, self.skip, axis=1))
        return start, states   # states (n, H*skip, state)

    def task(self, start, states, vstar, H):
        nq = self.m.nq
        q = states[:, self.skip-1::self.skip, 1:1+nq]
        half = H//2                               # steady-state speed (same definition as the eval tail)
        v = (q[:, -1, 0]-q[:, -half-1, 0])/(half*self.dt)
        up = np.mean(np.cos(q[:, :, 2]) > 0, axis=1)
        return -abs(v-vstar) - 2*np.maximum(.9-up, 0), v, up

    def close(self):
        self.engine.close()
        self.env.close()


def cem(sim, rng, vstar, delta, seed_base, gens=8, pop=64):
    center = np.zeros(30)
    center[1::5] = .6                              # a1 amplitude prior
    scale = np.full(30, .55)
    scale[2::5] = np.pi                            # phases
    scale[4::5] = np.pi
    cands = []
    for g in range(gens):
        params = rng.normal(center, scale, (pop, 30))
        avg = np.mean([sim.task(*sim.run(params, seed_base+2*g+j, H_SEARCH, delta), vstar, H_SEARCH)[0]
                       for j in range(2)], axis=0)
        elite = params[np.argsort(avg)[-10:]]
        center, scale = elite.mean(0), np.maximum(elite.std(0), .05)
        cands += [center.copy(), elite[-1].copy()]
    cands = np.array(cands)
    val = np.mean([sim.task(*sim.run(cands, seed_base+100+j, H_SEARCH, delta), vstar, H_SEARCH)[0]
                   for j in range(4)], axis=0)
    return cands[np.argmax(val)]


def episodes(sim, params, delta, vstar, reset_seeds):
    """Run H_EVAL rollouts; return per-episode dict with substep qpos (joints), contacts, task stats."""
    out = []
    nq, nv = sim.m.nq, sim.m.nv
    for s in reset_seeds:
        start, st = sim.run(params[None], s, H_EVAL, delta)
        st = st[0]
        q, qd = st[:, 1:1+nq], st[:, 1+nq:1+nq+nv]
        frames = q[sim.skip-1::sim.skip]
        tail = frames[-TAIL:]
        v = (tail[-1, 0]-frames[-TAIL-1, 0])/(TAIL*sim.dt)
        up = float(np.mean(np.cos(frames[:, 2]) > 0))
        cb = np.zeros(len(q), bool)
        cf = np.zeros(len(q), bool)
        for t in range(len(q)):
            sim.d1.qpos[:], sim.d1.qvel[:] = q[t], qd[t]
            mujoco.mj_forward(sim.m, sim.d1)
            for k in range(sim.d1.ncon):
                g = {sim.d1.contact[k].geom1, sim.d1.contact[k].geom2}
                if sim.floor in g:
                    cb[t] |= sim.bfoot in g
                    cf[t] |= sim.ffoot in g
        out.append(dict(reset=int(s), v=float(v), upright=up, joints=q[:, 3:9].copy(), cb=cb, cf=cf,
                        frames_joints=frames[:, 3:9].copy(),
                        task_ok=bool(abs(v-vstar) <= .1*vstar and up >= .9)))
    return out


def cycles(ep, sub_per_step):
    """Contact-aligned cycles on the TAIL window at substep resolution."""
    n = TAIL*sub_per_step
    J, cb, cf = ep["joints"][-n:], ep["cb"][-n:], ep["cf"][-n:]
    Ps = P*sub_per_step
    # debounced touchdown: >=3 substeps in contact after >=3 substeps airborne; then >=0.5P apart
    raw = [t for t in range(3, n-3) if cb[t:t+3].all() and not cb[t-3:t].any()]
    td = []
    for t in raw:
        if not td or t-td[-1] >= .5*Ps:
            td.append(t)
    td = np.array(td, int)
    feats, missing = [], 0
    for t0, t1 in zip(td[:-1], td[1:]):
        if not (.5*Ps <= t1-t0 <= 1.5*Ps):
            missing += 1
            continue
        x = t0 + np.arange(32)/32*(t1-t0)
        res = np.stack([np.interp(x, np.arange(n), J[:, k]) for k in range(6)], 1).ravel()
        feats.append(np.r_[res, cb[t0:t1].mean(), cf[t0:t1].mean()])
    return np.array(feats).reshape(-1, 194), missing


def mmd2(X, Y, h):
    def k(a, b):
        d = (a**2).sum(1)[:, None]+(b**2).sum(1)[None]-2*a@b.T
        return np.exp(-np.maximum(d, 0)/(2*h*h))
    m, n = len(X), len(Y)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    return float((kxx.sum()-np.trace(kxx))/(m*(m-1)) + (kyy.sum()-np.trace(kyy))/(n*(n-1)) - 2*kxy.mean())


def auc(far, near):
    r = rankdata(np.r_[far, near])
    return float((r[:len(far)].sum()-len(far)*(len(far)+1)/2)/(len(far)*len(near)))


def build(sim, vstar, nseeds=6):
    gaits = {}
    for name, delta in (("A_inphase", 0.0), ("B_antiphase", np.pi)):
        seeds = []
        for i in range(nseeds):
            rng = np.random.default_rng(7000+100*i+(0 if delta == 0 else 50))
            p = cem(sim, rng, vstar, delta, 120000+1000*i+(0 if delta == 0 else 500))
            eps = episodes(sim, p, delta, vstar, [130000+10*i+j+(0 if delta == 0 else 5000) for j in range(4)])
            seeds.append(dict(params=p, eps=eps))
            print(name, i, [(round(e["v"], 2), e["upright"], e["task_ok"]) for e in eps], flush=True)
        gaits[name] = dict(delta=delta, seeds=seeds)
    return gaits


def evaluate(sim, gaits):
    sub = sim.skip
    names = list(gaits)
    # per seed: list of cycle arrays (task-passing episodes only)
    cyc, miss, npass = {}, {}, {}
    for g in names:
        cyc[g], miss[g], npass[g] = [], 0, 0
        for s in gaits[g]["seeds"]:
            cs = []
            for e in s["eps"]:
                if not e["task_ok"]:
                    continue
                npass[g] += 1
                c, mi = cycles(e, sub)
                miss[g] += mi
                cs.append(c)
            cyc[g].append(np.concatenate(cs) if cs else np.zeros((0, 194)))
    train = np.concatenate([cyc[g][0] for g in names])
    mu, sd = train.mean(0), train.std(0)+1e-6
    z = lambda x: (x-mu)/sd
    dtr = np.sqrt(((z(train)[:, None]-z(train)[None])**2).sum(-1))
    h = float(np.median(dtr[np.triu_indices(len(train), 1)]))
    test = {g: [z(c) for c in cyc[g][1:] if len(c) >= 2] for g in names}
    floors = []
    for g in names:
        T = test[g]
        for idx in itertools.combinations(range(len(T)), 2):
            a = np.concatenate([T[i] for i in idx])
            b = np.concatenate([T[i] for i in range(len(T)) if i not in idx])
            floors.append(mmd2(a, b, h))
    between = []
    A, B = test[names[0]], test[names[1]]
    for ka, kb in ((2, 3), (3, 2)):
        for ia in itertools.combinations(range(len(A)), ka):
            for ib in itertools.combinations(range(len(B)), kb):
                between.append(mmd2(np.concatenate([A[i] for i in ia]), np.concatenate([B[i] for i in ib]), h))
    floor95 = float(np.percentile(floors, 95))
    res = dict(task_pass_episodes=npass, missing_cycles=miss,
               cycles_per_seed={g: [len(c) for c in cyc[g]] for g in names}, bandwidth=h,
               floor_values=floors, floor95=floor95, between_median=float(np.median(between)),
               between_min=float(np.min(between)), ratio=float(np.median(between)/max(floor95, 1e-12)))
    res["gate0_pass"] = bool(min(npass.values()) >= 20 and res["between_median"] >= 2*floor95)
    # controls on gait A test seeds: common time shift and front-leg delay (coordination broken)
    ctrl = {}
    Aeps = [e for s in gaits[names[0]]["seeds"][1:] for e in s["eps"] if e["task_ok"]]
    Aall = np.concatenate(A)
    for label, fn in (("common_shift_7sub", lambda J: np.roll(J, 7, axis=0)),
                      ("front_delay_P4", lambda J: np.c_[J[:, :3], np.roll(J[:, 3:], P*sub//4, axis=0)])):
        mod = []
        for e in Aeps:
            e2 = dict(e, joints=fn(e["joints"]))
            mod.append(z(cycles(e2, sub)[0]))
        ctrl[label] = mmd2(Aall, np.concatenate(mod), h)
    res["controls_vs_A_test"] = ctrl
    # #28 Fourier diagnostics on frame-rate joint angles (last 128 frames)
    fr = {g: [e["frames_joints"][-128:] for s in gaits[g]["seeds"] for e in s["eps"] if e["task_ok"]] for g in names}
    seed_of = {g: [i for i, s in enumerate(gaits[g]["seeds"]) for e in s["eps"] if e["task_ok"]] for g in names}
    trainA = np.array([x for x, i in zip(fr[names[0]], seed_of[names[0]]) if i == 0])
    sdj = np.maximum(np.concatenate([trainA.reshape(-1, 6)]).std(0), 1e-3)
    refs = {k: v.mean(0) for k, v in features(trainA/sdj).items()}
    def dist(x):
        f = features(np.array(x)/sdj)
        return {k: np.linalg.norm((v-refs[k]).reshape(len(x), -1), axis=1)/np.linalg.norm(refs[k]) for k, v in f.items()}
    testA = [x for x, i in zip(fr[names[0]], seed_of[names[0]]) if i > 0]
    dA, dB = dist(testA), dist(fr[names[1]])
    res["fourier_auc_B_farther_than_A"] = {k: auc(dB[k], dA[k]) for k in dA}
    res["fourier_median_dist"] = {k: dict(A_test=float(np.median(dA[k])), B=float(np.median(dB[k]))) for k in dA}
    # per-joint PSD peak frequency check (cadence match) and speed match
    res["speed"] = {g: dict(mean=float(np.mean([e["v"] for s in gaits[g]["seeds"] for e in s["eps"] if e["task_ok"]])))
                    for g in names}
    # realized thigh phase difference at gait frequency (cross-spectrum angle)
    def thigh_lag(g):
        out = []
        for x in fr[g]:
            F = np.fft.rfft(x-x.mean(0), axis=0)
            k = int(round(128/P))
            out.append(float(np.angle(F[k, FTHIGH]*np.conj(F[k, BTHIGH]))))
        return out
    res["realized_thigh_phase_rad"] = {g: dict(median=float(np.median(thigh_lag(g))),
                                              circ_mean=float(np.angle(np.mean(np.exp(1j*np.array(thigh_lag(g)))))))
                                       for g in names}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vstar", type=float, default=1.5)
    a = ap.parse_args()
    sim = Sim()
    try:
        gaits = build(sim, a.vstar)
        res = evaluate(sim, gaits)
        res.update(vstar=a.vstar, period_steps=P, eval_horizon=H_EVAL, tail=TAIL,
                   params={g: [s["params"].tolist() for s in gaits[g]["seeds"]] for g in gaits},
                   episodes={g: [[dict(reset=e["reset"], v=e["v"], upright=e["upright"], task_ok=e["task_ok"])
                                  for e in s["eps"]] for s in gaits[g]["seeds"]] for g in gaits})
        np.savez_compressed(HERE/f"gaits_v{a.vstar}.npz", **{f"{g}_s{i}": s["params"] for g in gaits
                                                            for i, s in enumerate(gaits[g]["seeds"])})
        (HERE/f"gate0_v{a.vstar}.json").write_text(json.dumps(res, indent=1, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k not in ("params", "episodes", "floor_values")}, indent=1))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
