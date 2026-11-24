"""S4 (PREREG.md 'S4' + '수정 1'): multi-style mode collapse.  Demo mixture of 3 footfall styles F/R/M at v*=1.0.
Arms (one process, batched members): amp_phase, fm_phase, template_any, amp_latent, oracle_mix (positive control).
  python sac34_s4.py --demos s2/demos.npz --demosM s2/demosM.npz --w_lock 0.5 --out s4
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sac34 as C  # noqa: E402
from common import LOG_INV_DELTA  # noqa: E402
from vec import BSAC, bmlp  # noqa: E402
from phase_util import BPhaseFlow, phase_feats  # noqa: E402

STY = ["F", "R", "M"]; ORACLE_T = [C.L_F, C.L_R, 0.5]; K = C.K


def seg_lags(ev):
    out = []
    for j in range(4):
        a, b = 200 + 200 * j, 400 + 200 * j
        ls = [l for t_, l in ev.lags if a <= t_ < b]
        m, R = C.circ_mean(ls) if len(ls) >= 3 else (float("nan"), 0.0)
        out.append((m, R, len(ls), [float(x) for x in ls]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True); p.add_argument("--demosM", required=True); p.add_argument("--w_lock", type=float, required=True)
    p.add_argument("--steps", type=int, default=300_000); p.add_argument("--out", required=True); p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--arms", default="amp_phase,fm_phase,fm_contrast,template_any,amp_latent,oracle_mix")
    p.add_argument("--eval_every", type=int, default=25_000); p.add_argument("--n_eval", type=int, default=30)
    p.add_argument("--fm_iters", type=int, default=30_000); p.add_argument("--s2", required=True); p.add_argument("--s2M", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(4321); np.random.seed(4321)
    dev = torch.device("cuda")
    seeds = [int(s) for s in args.seeds.split(",")]
    D1, D2 = np.load(args.demos), np.load(args.demosM)
    src = {"F": D1, "R": D1, "M": D2}
    X = {k: src[k][f"X_{k}"] for k in STY}; P = {k: src[k][f"P_{k}"] for k in STY}
    n = min(len(X[k]) for k in STY)                       # equal amount per style (episodes have equal window count)
    centers = [float(src[k][f"lagD_{k}"]) for k in STY]
    import agg34_s4
    q = agg34_s4.qualify(args.s2, args.s2M, centers)
    print("classifier qualification (pre-training)", q, flush=True)
    assert q[0], "S4 stopped: held-out classifier qualification failed (PREREG S4 수정 2)"
    print("demo windows per style", {k: len(X[k]) for k in STY}, "-> using", n, "centers", centers, flush=True)
    XT = torch.tensor(np.concatenate([X[k][:n] for k in STY]), device=dev)
    PT = torch.tensor(np.concatenate([P[k][:n] for k in STY]), device=dev)
    ZT = torch.tensor(np.repeat(np.arange(3), n), device=dev)
    MU = XT[:, :K].mean(0).repeat(2); SD = (XT[:, :K].std(0) + 1e-3).repeat(2)
    XE = (XT - MU) / SD
    ZI = [torch.nonzero(ZT == k).squeeze(1) for k in range(3)]

    members = [dict(arm=a, seed=s, w=args.w_lock) for a in args.arms.split(",") for s in seeds]
    S = len(members); OD = C.OD + 3
    ids = lambda pred: torch.tensor([i for i, m in enumerate(members) if pred(m)], device=dev, dtype=torch.long)
    print("members", S, flush=True)

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

    arms = {m["arm"] for m in members}
    fm = None; diag = {}
    if arms & {"fm_phase", "fm_contrast"}:
        fm = BPhaseFlow(1, 2 * K, "film4").to(dev)
        opt = torch.optim.Adam(fm.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.fm_iters)
        for i in range(args.fm_iters):
            j = torch.randint(0, XE.shape[0], (1, 1024), device=dev)
            l = loss_train(fm, XE[j], PT[j]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        fm.requires_grad_(False)
        with torch.no_grad():
            res = torch.cat([loss_ps(fm, XE[None, a:a + 4096], PT[None, a:a + 4096], 16)[0] for a in range(0, XE.shape[0], 4096)])
        q_D = float(res.median())
        rs = torch.clamp(1 - torch.log1p(res / q_D) / math.log(2), -1, 1)
        diag["fm_q_D"] = q_D
        diag["fm_rstyle_mean_per_style"] = {STY[k]: float(rs[ZI[k]].mean()) for k in range(3)}   # codex r1: per-style mean score ranking
        print("FM prior", diag, flush=True)
    I_fc = None; learner = None
    if "fm_contrast" in arms:
        I_fc = ids(lambda m: m["arm"] == "fm_contrast")
        learner = BPhaseFlow(len(I_fc), 2 * K, "film4").to(dev)
        opt_l = torch.optim.Adam(learner.parameters(), 3e-4)
    discs = {}
    for arm, din in (("amp_phase", 2 * K + 8), ("amp_latent", 2 * K + 8 + 3)):
        if arm in arms:
            I = ids(lambda m, a=arm: m["arm"] == a)
            dn = bmlp(len(I), [din, 256, 256, 1], nn.ReLU).to(dev)
            discs[arm] = (I, dn, torch.optim.Adam(dn.parameters(), 1e-4))

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
    B_z = torch.zeros(S, NB, dtype=torch.long, device=dev)
    ar = torch.arange(S, device=dev)[:, None]
    g = lambda buf, idx, rows=None: buf[ar if rows is None else rows[:, None], idx]
    W = torch.tensor([m["w"] for m in members], device=dev)[:, None]
    I_ord = ids(lambda m: m["arm"] in ("oracle_mix", "template_any")); I_fm = ids(lambda m: m["arm"] == "fm_phase")
    onehot = lambda z: torch.nn.functional.one_hot(z, 3).float()

    def style_reward(idx):
        r = torch.zeros(S, idx.shape[1], device=dev)
        if len(I_ord):
            r[I_ord] = g(B_ro, idx)[I_ord]
        x = (torch.cat([g(B_o, idx)[..., :K], g(B_o2, idx)[..., :K]], -1) - MU) / SD; ph = g(B_p, idx)
        with torch.no_grad():
            for arm, (I, dn, _) in discs.items():
                z = torch.cat([x[I], phase_feats(ph[I], 4)], -1)
                if arm == "amp_latent":
                    z = torch.cat([z, onehot(g(B_z, idx)[I])], -1)
                r[I] = 2 * torch.clamp(1 - 0.25 * (dn(z).squeeze(-1) - 1) ** 2, min=0) - 1
            if I_fc is not None:
                nn_, Bn = len(I_fc), x.shape[1]
                rD = loss_ps(fm, x[I_fc].reshape(1, nn_ * Bn, -1), ph[I_fc].reshape(1, nn_ * Bn), 4).view(nn_, Bn)
                rP = loss_ps(learner, x[I_fc], ph[I_fc], 4)
                r[I_fc] = torch.clamp((rP - rD) / diag["fm_q_D"], -1, 1)
            if len(I_fm):
                nn_, Bn = len(I_fm), x.shape[1]
                res = loss_ps(fm, x[I_fm].reshape(1, nn_ * Bn, -1), ph[I_fm].reshape(1, nn_ * Bn), 4).view(nn_, Bn)
                r[I_fm] = torch.clamp(1 - torch.log1p(res / diag["fm_q_D"]) / math.log(2), -1, 1)
        return r

    def disc_step(step):
        lo = max(0, step - 100_000)
        for arm, (I, dn, opt) in discs.items():
            nI = len(I)
            je = torch.randint(0, XE.shape[0], (nI, 256), device=dev)
            xe = torch.cat([XE[je], phase_feats(PT[je], 4)], -1)
            jp = torch.randint(lo, step + 1, (nI, 256), device=dev)
            xp = torch.cat([(torch.cat([g(B_o, jp, I)[..., :K], g(B_o2, jp, I)[..., :K]], -1) - MU) / SD, phase_feats(g(B_p, jp, I), 4)], -1)
            if arm == "amp_latent":
                xe = torch.cat([xe, onehot(ZT[je])], -1); xp = torch.cat([xp, onehot(g(B_z, jp, I))], -1)
            xe = xe.requires_grad_(True)
            de, dp = dn(xe), dn(xp)
            gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(-1).mean(1)
            ld = (((de - 1) ** 2).mean((1, 2)) + ((dp + 1) ** 2).mean((1, 2)) + 10 * gp).sum()
            opt.zero_grad(); ld.backward(); opt.step()

    envs = [C.Cheetah() for _ in members]; eval_envs = [C.Cheetah() for _ in members]
    rngs = [np.random.default_rng(10_000 + m["seed"]) for m in members]
    zrng = np.random.default_rng(99)

    def ep_z(i, env, given=None):
        a = members[i]["arm"]
        if a == "oracle_mix":
            return int(np.argmax(env.d.qpos[3:6]))           # style fixed by the (observed) initial state
        if a == "amp_latent":
            return int(zrng.integers(3)) if given is None else given
        return -1

    def aug(o, z):
        e = np.zeros(3, np.float32)
        if z >= 0:
            e[z] = 1.0
        return np.concatenate([o, e])

    def order_r(i, env, z):
        a = members[i]["arm"]
        if a == "oracle_mix":
            return env.order_score(ORACLE_T[z])
        if a == "template_any":
            return max(env.order_score(c) for c in centers)
        return 0.0

    def evaluate(n_ep, base, det):
        recs = [[] for _ in range(S)]
        for e in range(n_ep):
            Z = []; obs = []
            for i, ev in enumerate(eval_envs):
                o = ev.reset(base + e); z = ep_z(i, ev, given=e % 3)
                Z.append(z); obs.append(aug(o, z if members[i]["arm"] == "amp_latent" else -1))
            obs = np.stack(obs)
            XS = [[float(ev.d.qpos[0])] for ev in eval_envs]; UP = [[] for _ in range(S)]; CT = [[] for _ in range(S)]
            for t in range(C.H):
                A = agent.act(torch.tensor(obs, device=dev), det=det).cpu().numpy()
                for i, ev in enumerate(eval_envs):
                    o, _, inf = ev.step(A[i]); obs[i] = aug(o, Z[i] if members[i]["arm"] == "amp_latent" else -1)
                    XS[i].append(float(ev.d.qpos[0])); UP[i].append(inf["up"]); CT[i].append(inf["ctrl"])
            for i, ev in enumerate(eval_envs):
                rec = C.episode_record(ev, XS[i], UP[i], CT[i]); rec["segs"] = seg_lags(ev); rec["z"] = Z[i]
                recs[i].append(rec)
        return recs

    names = [f"s4_{m['arm']}_s{m['seed']}" for m in members]
    logs = [dict(args=vars(args), member=m, centers=centers, diag=diag, evals=[]) for m in members]
    Zc = [0] * S; obs = []
    for j, (e, m) in enumerate(zip(envs, members)):
        o = e.reset(1000 * j + m["seed"]); Zc[j] = ep_z(j, e); obs.append(aug(o, Zc[j] if m["arm"] == "amp_latent" else -1))
    obs = np.stack(obs)
    t0 = time.time(); START = 5000; ep_t = 0
    for step in range(args.steps):
        A = np.stack([r.uniform(-1, 1, 6) for r in rngs]).astype(np.float32) if step < START else \
            agent.act(torch.tensor(obs, device=dev)).cpu().numpy()
        O2 = np.empty_like(obs); RT = np.empty(S, np.float32); RO = np.zeros(S, np.float32); PH = np.empty(S, np.float32)
        for i, e in enumerate(envs):
            PH[i] = e.phase()
            o2, rt, _ = e.step(A[i]); O2[i] = aug(o2, Zc[i] if members[i]["arm"] == "amp_latent" else -1); RT[i] = rt
            RO[i] = order_r(i, e, Zc[i])
        B_o[:, step] = torch.from_numpy(obs).to(dev); B_a[:, step] = torch.from_numpy(A).to(dev)
        B_o2[:, step] = torch.from_numpy(O2).to(dev); B_rt[:, step] = torch.from_numpy(RT).to(dev)
        B_ro[:, step] = torch.from_numpy(RO).to(dev); B_p[:, step] = torch.from_numpy(PH).to(dev)
        B_z[:, step] = torch.tensor([max(z, 0) for z in Zc], device=dev)
        ep_t += 1
        if ep_t >= C.H:
            nobs = []
            for j, (e, m) in enumerate(zip(envs, members)):
                o = e.reset(1000 * (step + 1) + 7 * j + m["seed"]); Zc[j] = ep_z(j, e)
                nobs.append(aug(o, Zc[j] if m["arm"] == "amp_latent" else -1))
            obs = np.stack(nobs); ep_t = 0
        else:
            obs = O2
        if step >= START:
            if discs and step % 2 == 0:
                disc_step(step)
            if learner is not None and step % 1000 == 0:
                lo_r = max(0, step - 20_000)
                for _ in range(2000 if step == START else 250):
                    jr = torch.randint(lo_r, step + 1, (len(I_fc), 512), device=dev)
                    xr_ = (torch.cat([g(B_o, jr, I_fc)[..., :K], g(B_o2, jr, I_fc)[..., :K]], -1) - MU) / SD
                    l = loss_train(learner, xr_, g(B_p, jr, I_fc)); opt_l.zero_grad(); l.backward(); opt_l.step()
            idx = torch.randint(0, step + 1, (S, 256), device=dev)
            r = g(B_rt, idx) + W * style_reward(idx)
            agent.update(g(B_o, idx), g(B_a, idx), r, g(B_o2, idx), torch.zeros_like(r))
        if (step + 1) % args.eval_every == 0:
            sps = (step + 1) / (time.time() - t0)
            ev = evaluate(3, 900_000, False)
            summ = {}
            for i, m in enumerate(members):
                L = ev[i]
                logs[i]["evals"].append(dict(step=step + 1, sps=sps, v=float(np.mean([r_["v"] for r_ in L])),
                                             task_ok=float(np.mean([r_["task_ok"] for r_ in L])), lag=[r_["lag"] for r_ in L]))
                json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
                summ.setdefault(m["arm"], []).append(f"ok{np.mean([r_['task_ok'] for r_ in L]):.1f} l{[round(r_['lag'], 2) if np.isfinite(r_['lag']) else None for r_ in L]}")
            print(f"{step+1:>7d} sps {sps:.0f} min {(time.time()-t0)/60:.1f} | " + " | ".join(f"{k}: " + ",".join(v) for k, v in summ.items()), flush=True)

    torch.save(dict(actor=agent.actor.state_dict(), members=members, OD=OD), os.path.join(args.out, "s4_actors.pt"))
    fin = {"stoch": evaluate(args.n_eval, 800_000, False), "det": evaluate(args.n_eval, 800_000, True)}
    for i in range(S):
        logs[i]["final"] = {k: v[i] for k, v in fin.items()}; logs[i]["final"]["minutes"] = (time.time() - t0) / 60
        json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
    print("DONE s4", (time.time() - t0) / 60, "min", flush=True)


if __name__ == "__main__":
    main()
