"""#34 SAC footfall-order experiment (PREREG.md in this folder).  S1 oracle positive control / S3 learned-style label swap.
All members (arm x target/label x seed) are stacked in ONE process (vec.BSAC batched members).  Run with fm34 venv python.
  python sac34.py --stage s1 --out <dir>
  python sac34.py --stage s3 --demos <dir>/demos.npz --w_lock 0.5 --out <dir>
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src" / "v2" / "il"))
from common import LOG_INV_DELTA  # noqa: E402
from vec import BSAC, bmlp  # noqa: E402
from phase_util import BPhaseFlow, phase_feats  # noqa: E402

VSTAR, L_R, L_F = 1.0, 0.25, 0.75
H, SUB, AIR = 1000, 5, 8                 # episode steps, substeps/step, min airborne substeps before a touchdown
TB_LO, TB_HI = 1.5 * SUB, 40 * SUB       # valid back-cycle range (substeps)
STALE = 40                               # steps after which the held order score decays to -1
K = 17                                   # style feature dim (env obs)
OD = K + 6


def dcirc(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def circ_mean(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), 0.0
    z = np.exp(2j * np.pi * x).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(abs(z))


def s_coord(l):
    """signed order coordinate (F side positive); undefined -> 0"""
    return 0.0 if (l is None or not np.isfinite(l)) else dcirc(l, L_R) - dcirc(l, L_F)


class Cheetah:
    """HalfCheetah-v5 stepped by hand (ctrl, 5 x mj_step) with substep-resolution foot-floor contact events."""
    def __init__(self):
        self.env = gym.make("HalfCheetah-v5"); u = self.env.unwrapped
        self.u, self.m, self.d, self.dt = u, u.model, u.data, u.dt
        nm = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.floor, self.feet = nm("floor"), (nm("bfoot"), nm("ffoot"))

    def reset(self, seed):
        self.env.reset(seed=int(seed)); self.t = 0; self.sub = 0
        self.air = [0, 0]; self.tb = []; self.tf = None
        self.l_last, self.l_sub = None, None       # last VALID front lag and its substep time
        self.lags = []                             # (step, lag) of valid front touchdowns
        self.Tbs = []
        return self.obs()

    def contacts(self):
        g = self.d.contact.geom[: self.d.ncon]
        if len(g) == 0:
            return False, False
        fl = (g == self.floor).any(1)
        return bool((fl & (g == self.feet[0]).any(1)).any()), bool((fl & (g == self.feet[1]).any(1)).any())

    def _td(self, j):
        if j == 0:
            self.tb = (self.tb + [self.sub])[-2:]
            if len(self.tb) == 2:
                self.Tbs.append((self.tb[1] - self.tb[0]) / SUB)
        else:
            self.tf = self.sub
            if len(self.tb) == 2:
                Tb = self.tb[1] - self.tb[0]
                if TB_LO <= Tb <= TB_HI:
                    l = (self.sub - self.tb[1]) / Tb
                    if l < 1.0:
                        self.l_last, self.l_sub = l, self.sub; self.lags.append((self.t + 1, l))

    def step(self, a):
        x0 = float(self.d.qpos[0]); self.d.ctrl[:] = a
        for _ in range(SUB):
            mujoco.mj_step(self.m, self.d); self.sub += 1
            for j, c in enumerate(self.contacts()):
                if c:
                    if self.air[j] >= AIR:
                        self._td(j)
                    self.air[j] = 0
                else:
                    self.air[j] += 1
        self.t += 1
        vx = (float(self.d.qpos[0]) - x0) / self.dt
        up = math.cos(float(self.d.qpos[2])) > 0
        ctrl = float(np.sum(np.square(a)))
        r_task = -abs(vx - VSTAR) - 0.1 * ctrl - (0.0 if up else 1.0)
        return self.obs(), r_task, dict(vx=vx, up=up, ctrl=ctrl)

    def order_score(self, target):
        if self.l_last is None or (self.sub - self.l_sub) > STALE * SUB:
            return -1.0
        return 1.0 - 4.0 * dcirc(self.l_last, target)

    def phase(self):
        """contact phase of the back-foot cycle in [0, 2pi]"""
        if len(self.tb) < 2 or not (TB_LO <= self.tb[1] - self.tb[0] <= TB_HI):
            return 0.0
        return 2 * math.pi * min((self.sub - self.tb[1]) / (self.tb[1] - self.tb[0]), 1.0)

    def obs(self):
        o = self.u._get_obs()
        sb = (self.sub - self.tb[-1]) / SUB if self.tb else STALE
        sf = (self.sub - self.tf) / SUB if self.tf is not None else STALE
        Tb = (self.tb[1] - self.tb[0]) / SUB if len(self.tb) == 2 else STALE
        v = self.l_last is not None
        l = self.l_last if v else 0.0
        ex = [min(sb, STALE) / STALE, min(sf, STALE) / STALE, min(Tb, STALE) / STALE,
              math.cos(2 * math.pi * l) * v, math.sin(2 * math.pi * l) * v, float(v)]
        return np.concatenate([o, ex]).astype(np.float32)


def episode_record(env, xs, ups, ctrls):
    """per-episode metrics on a finished 1000-step evaluation"""
    x = np.asarray(xs)
    v = float((x[H] - x[H // 2]) / ((H // 2) * env.dt))
    up = float(np.mean(ups))
    ls = [l for t, l in env.lags if t >= 200]
    lb, R = circ_mean(ls) if len(ls) >= 10 else (float("nan"), 0.0)
    rec = dict(v=v, upright=up, ctrl=float(np.mean(ctrls)), task_ok=bool(abs(v - VSTAR) <= .1 * VSTAR and up >= .9),
               lag=lb, lag_R=R, n_lag=len(ls), Tb_med=float(np.median(env.Tbs)) if env.Tbs else None,
               s=s_coord(lb))
    for nm, tg in (("F", L_F), ("R", L_R)):
        rec["order_ok_" + nm] = bool(np.isfinite(lb) and dcirc(lb, tg) < 0.125)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["s1", "s3", "s4prep"])
    p.add_argument("--steps", type=int, default=300_000); p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2"); p.add_argument("--eval_every", type=int, default=25_000)
    p.add_argument("--n_final", type=int, default=10); p.add_argument("--demos", default=None)
    p.add_argument("--w_lock", type=float, default=None); p.add_argument("--arms", default="amp_plain,amp_phase,fm_phase,template")
    p.add_argument("--fm_iters", type=int, default=30_000); p.add_argument("--reg", default=None)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(1234); np.random.seed(1234)
    dev = torch.device("cuda")
    seeds = [int(s) for s in args.seeds.split(",")]
    eval_base = 700_000 if args.stage == "s3" else 500_000

    # ---------------- members
    members = []   # dict(arm, label, target, w, seed)
    if args.stage == "s1":
        members += [dict(arm="task", label="-", target=None, w=0.0, seed=s) for s in seeds]
        for w in (0.5, 1.0):
            for lab, tg in (("F", L_F), ("R", L_R)):
                members += [dict(arm="oracle", label=lab, target=tg, w=w, seed=s) for s in seeds]
    elif args.stage == "s4prep":    # third style for S4: oracle order target l=0.5 (front lands half a back-cycle later)
        for w in (0.5, 1.0):
            members += [dict(arm="oracle", label="M", target=0.5, w=w, seed=s) for s in seeds]
    else:
        assert args.demos and args.w_lock is not None
        D = np.load(args.demos)
        for arm in args.arms.split(","):
            for lab in ("F", "R"):
                tg = float(D[f"lagD_{lab}"]) if arm == "template" else None
                members += [dict(arm=arm, label=lab, target=tg, w=args.w_lock, seed=s) for s in seeds]
    S = len(members)
    print("members", S, [(m["arm"], m["label"], m["w"], m["seed"]) for m in members], flush=True)

    # ---------------- style models (S3)
    MU = SD = None; style = {}
    if args.stage == "s3":
        Xtr = {lab: torch.tensor(D[f"X_{lab}"], device=dev) for lab in ("F", "R")}      # raw (N, 34)
        Ptr = {lab: torch.tensor(D[f"P_{lab}"], device=dev) for lab in ("F", "R")}      # contact phase (N,)
        allx = torch.cat([Xtr["F"], Xtr["R"]])[:, :K]
        MU = allx.mean(0).repeat(2); SD = (allx.std(0) + 1e-3).repeat(2)
        XE = {lab: (Xtr[lab] - MU) / SD for lab in ("F", "R")}

    def t_draw(shape):
        s_ = torch.rand(shape, device=dev) * LOG_INV_DELTA; t = 1 - torch.exp(-s_); return t, LOG_INV_DELTA * t

    def loss_ps(net, x1, phi, M):
        Sx, B, d = x1.shape
        xr, pr = x1.repeat(1, M, 1), phi.repeat(1, M)
        t, w = t_draw((Sx, B * M)); x0 = torch.randn_like(xr)
        xt = (1 - t[..., None]) * x0 + t[..., None] * xr
        return (((net(xt, t, pr) - (xr - x0)) ** 2).sum(-1) * w).view(Sx, M, B).mean(1)

    def loss_train(net, x1, phi):
        Sx, B, d = x1.shape
        t = torch.where(torch.rand(Sx, B, device=dev) < 0.5, torch.rand(Sx, B, device=dev),
                        1 - torch.exp(-torch.rand(Sx, B, device=dev) * LOG_INV_DELTA))
        x0 = torch.randn_like(x1); xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        return ((net(xt, t, phi) - (x1 - x0)) ** 2).sum(-1).mean(1).sum()

    arms = sorted({m["arm"] for m in members})
    ids = lambda pred: torch.tensor([i for i, m in enumerate(members) if pred(m)], device=dev, dtype=torch.long)
    fm_info = {}
    REG = json.load(open(args.reg)) if args.reg else None      # S3 수정 1: LODO-selected regularisation
    FMCFG = {"fm_phase": (0.0, args.fm_iters)}
    if REG:
        c_ = REG["fm"][REG["chosen_fm"]]; FMCFG["fm_reg"] = (c_["sigma"], c_["iters"])
    GPW = {"amp_plain": 10.0, "amp_phase": 10.0}
    if REG:
        GPW["amp_phase_reg"] = REG["amp"][REG["chosen_amp"]]["lam"]
    for fa in [a for a in arms if a in FMCFG]:
      sig_, its_ = FMCFG[fa]
      for lab in ("F", "R"):
            net = BPhaseFlow(1, 2 * K, "film4").to(dev)
            opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, its_)
            for i in range(its_):
                j = torch.randint(0, XE[lab].shape[0], (1, 1024), device=dev)
                l = loss_train(net, XE[lab][j] + sig_ * torch.randn_like(XE[lab][j]), Ptr[lab][j]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
            net.requires_grad_(False)
            with torch.no_grad():
                res = torch.cat([loss_ps(net, XE[lab][None, a:a + 4096], Ptr[lab][None, a:a + 4096], 16)[0]
                                 for a in range(0, XE[lab].shape[0], 4096)])
            q = float(res.median()); style[(fa, lab)] = (net, q)
            # static diagnostic (not a verdict): held-out demo windows scored by this prior
            sc = {}
            for hl in ("F", "R"):
                Xh = (torch.tensor(D[f"Xho_{hl}"], device=dev) - MU) / SD; Ph = torch.tensor(D[f"Pho_{hl}"], device=dev)
                with torch.no_grad():
                    rr = torch.cat([loss_ps(net, Xh[None, a:a + 4096], Ph[None, a:a + 4096], 16)[0] for a in range(0, Xh.shape[0], 4096)])
                sc[hl] = (-rr).cpu().numpy()
            own, oth = sc[lab], sc["R" if lab == "F" else "F"]
            rng_ = np.random.default_rng(0); a_ = rng_.choice(own, 4000); b_ = rng_.choice(oth, 4000)
            fm_info[f"{fa}_{lab}"] = dict(sigma=sig_, iters=its_, q_D=q, auc_window_own_vs_other=float((a_[:, None] > b_[None]).mean()),
                                rstyle_med_own=float(np.median(np.clip(1 - np.log1p(-own / q) / math.log(2), -1, 1))),
                                rstyle_med_other=float(np.median(np.clip(1 - np.log1p(-oth / q) / math.log(2), -1, 1))))
            print("FM prior", fa, lab, fm_info[f"{fa}_{lab}"], flush=True)
    discs = {}
    for arm in ("amp_plain", "amp_phase", "amp_phase_reg"):
        if arm in arms:
            I = ids(lambda m, a=arm: m["arm"] == a)
            din = 2 * K + (0 if arm == "amp_plain" else 8)
            dn = bmlp(len(I), [din, 256, 256, 1], nn.ReLU).to(dev)
            discs[arm] = (I, dn, torch.optim.Adam(dn.parameters(), 1e-4))

    # ---------------- agent, paired init (same seed -> identical initial nets across arms/labels)
    agent = BSAC(S, OD, 6, dev)
    ref = [next(j for j, mm in enumerate(members) if mm["seed"] == m["seed"]) for m in members]
    with torch.no_grad():
        for mod in (agent.actor, agent.critic, agent.targ):
            for prm in mod.parameters():
                prm.copy_(prm[ref])
        agent.log_alpha.copy_(agent.log_alpha[ref])

    NB = args.steps
    B_o = torch.zeros(S, NB, OD, device=dev); B_a = torch.zeros(S, NB, 6, device=dev); B_o2 = torch.zeros(S, NB, OD, device=dev)
    B_rt = torch.zeros(S, NB, device=dev); B_ro = torch.zeros(S, NB, device=dev); B_p = torch.zeros(S, NB, device=dev)
    ar = torch.arange(S, device=dev)[:, None]
    g = lambda buf, idx, rows=None: buf[ar if rows is None else rows[:, None], idx]
    W = torch.tensor([m["w"] for m in members], device=dev)[:, None]
    I_ord = ids(lambda m: m["arm"] in ("oracle", "template"))
    I_fm = {(fa, lab): ids(lambda m, l=lab, a=fa: m["arm"] == a and m["label"] == l) for fa in FMCFG for lab in ("F", "R")}

    def style_reward(idx):
        r = torch.zeros(S, idx.shape[1], device=dev)
        if len(I_ord):
            r[I_ord] = g(B_ro, idx)[I_ord]
        if not (discs or style):
            return r
        x = (torch.cat([g(B_o, idx)[..., :K], g(B_o2, idx)[..., :K]], -1) - MU) / SD; ph = g(B_p, idx)
        with torch.no_grad():
            for arm, (I, dn, _) in discs.items():
                z = x[I] if arm == "amp_plain" else torch.cat([x[I], phase_feats(ph[I], 4)], -1)
                r[I] = 2 * torch.clamp(1 - 0.25 * (dn(z).squeeze(-1) - 1) ** 2, min=0) - 1
            for key, I in I_fm.items():
                if len(I) == 0:
                    continue
                net, q = style[key]; n, Bn = len(I), x.shape[1]
                res = loss_ps(net, x[I].reshape(1, n * Bn, -1), ph[I].reshape(1, n * Bn), 4).view(n, Bn)
                r[I] = torch.clamp(1 - torch.log1p(res / q) / math.log(2), -1, 1)
        return r

    def disc_step(step):
        lo = max(0, step - 100_000)
        for arm, (I, dn, opt) in discs.items():
            n = len(I)
            labs = [members[i]["label"] for i in I.tolist()]
            je = [torch.randint(0, XE[l].shape[0], (256,), device=dev) for l in labs]
            xe = torch.stack([XE[l][j] for l, j in zip(labs, je)])
            jp = torch.randint(lo, step + 1, (n, 256), device=dev)
            xp = (torch.cat([g(B_o, jp, I)[..., :K], g(B_o2, jp, I)[..., :K]], -1) - MU) / SD
            if arm != "amp_plain":
                pe = torch.stack([Ptr[l][j] for l, j in zip(labs, je)])
                xe = torch.cat([xe, phase_feats(pe, 4)], -1); xp = torch.cat([xp, phase_feats(g(B_p, jp, I), 4)], -1)
            xe = xe.requires_grad_(True)
            de, dp = dn(xe), dn(xp)
            gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(-1).mean(1)
            ld = (((de - 1) ** 2).mean((1, 2)) + ((dp + 1) ** 2).mean((1, 2)) + GPW[arm] * gp).sum()
            opt.zero_grad(); ld.backward(); opt.step()

    envs = [Cheetah() for _ in members]; eval_envs = [Cheetah() for _ in members]
    rngs = [np.random.default_rng(10_000 + m["seed"]) for m in members]

    def evaluate(n_ep, base):
        recs = [[] for _ in range(S)]
        for e in range(n_ep):
            obs = np.stack([ev.reset(base + e) for ev in eval_envs])
            XS = [[float(ev.d.qpos[0])] for ev in eval_envs]; UP = [[] for _ in range(S)]; CT = [[] for _ in range(S)]
            for t in range(H):
                A = agent.act(torch.tensor(obs, device=dev), det=True).cpu().numpy()
                for i, ev in enumerate(eval_envs):
                    o, _, inf = ev.step(A[i]); obs[i] = o
                    XS[i].append(float(ev.d.qpos[0])); UP[i].append(inf["up"]); CT[i].append(inf["ctrl"])
            for i, ev in enumerate(eval_envs):
                recs[i].append(episode_record(ev, XS[i], UP[i], CT[i]))
        return recs

    names = [f"{args.stage}_{m['arm']}_{m['label']}_w{m['w']}_s{m['seed']}" for m in members]
    logs = [dict(args=vars(args), member=m, evals=[]) for m in members]
    obs = np.stack([e.reset(1000 * j + m["seed"]) for j, (e, m) in enumerate(zip(envs, members))])
    t0 = time.time(); START = 5000; ep_t = 0
    for step in range(args.steps):
        A = np.stack([r.uniform(-1, 1, 6) for r in rngs]).astype(np.float32) if step < START else \
            agent.act(torch.tensor(obs, device=dev)).cpu().numpy()
        O2 = np.empty_like(obs); RT = np.empty(S, np.float32); RO = np.zeros(S, np.float32); PH = np.empty(S, np.float32)
        for i, e in enumerate(envs):
            PH[i] = e.phase()
            o2, rt, _ = e.step(A[i]); O2[i] = o2; RT[i] = rt
            if members[i]["target"] is not None:
                RO[i] = e.order_score(members[i]["target"])
        B_o[:, step] = torch.from_numpy(obs).to(dev); B_a[:, step] = torch.from_numpy(A).to(dev)
        B_o2[:, step] = torch.from_numpy(O2).to(dev); B_rt[:, step] = torch.from_numpy(RT).to(dev)
        B_ro[:, step] = torch.from_numpy(RO).to(dev); B_p[:, step] = torch.from_numpy(PH).to(dev)
        ep_t += 1
        if ep_t >= H:      # time limit only (HalfCheetah never terminates); done stays 0 -> bootstrap
            obs = np.stack([e.reset(1000 * (step + 1) + 7 * j + m["seed"]) for j, (e, m) in enumerate(zip(envs, members))]); ep_t = 0
        else:
            obs = O2
        if step >= START:
            if discs and step % 2 == 0:
                disc_step(step)
            idx = torch.randint(0, step + 1, (S, 256), device=dev)
            r = g(B_rt, idx) + W * style_reward(idx)
            agent.update(g(B_o, idx), g(B_a, idx), r, g(B_o2, idx), torch.zeros_like(r))
        if (step + 1) % args.eval_every == 0 or step + 1 == 2000:
            sps = (step + 1) / (time.time() - t0)
            if (step + 1) % args.eval_every == 0:
                ev = evaluate(2, 900_000)
                for i in range(S):
                    L = ev[i]
                    logs[i]["evals"].append(dict(step=step + 1, sps=sps, v=float(np.mean([r_["v"] for r_ in L])),
                                                 task_ok=float(np.mean([r_["task_ok"] for r_ in L])),
                                                 lag=[r_["lag"] for r_ in L], s=float(np.mean([r_["s"] for r_ in L]))))
                    json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
                summ = {}
                for i, m in enumerate(members):
                    key = f"{m['arm']}/{m['label']}/{m['w']}"; summ.setdefault(key, []).append(
                        f"v{np.mean([r_['v'] for r_ in ev[i]]):.2f} ok{np.mean([r_['task_ok'] for r_ in ev[i]]):.1f} s{np.mean([r_['s'] for r_ in ev[i]]):+.2f}")
                print(f"{step+1:>7d} sps {sps:.0f} min {(time.time()-t0)/60:.1f} | " + " | ".join(f"{k}: " + ",".join(v) for k, v in summ.items()), flush=True)
            else:
                print(f"{step+1} sps {sps:.0f}  Tb_med {[np.median(e.Tbs) if e.Tbs else None for e in envs][:5]} nlag {[len(e.lags) for e in envs][:5]}", flush=True)

    torch.save(dict(actor=agent.actor.state_dict(), members=members, OD=OD), os.path.join(args.out, f"{args.stage}_actors.pt"))
    fin = evaluate(args.n_final, eval_base)
    for i in range(S):
        logs[i]["final"] = dict(episodes=fin[i], minutes=(time.time() - t0) / 60)
        if fm_info:
            logs[i]["fm_info"] = fm_info
        json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
    if fm_info:
        json.dump(fm_info, open(os.path.join(args.out, "fm_static_diag.json"), "w"), indent=1)
    print("DONE", args.stage, (time.time() - t0) / 60, "min", flush=True)


if __name__ == "__main__":
    main()
