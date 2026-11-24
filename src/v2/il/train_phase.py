"""
Phase-timing experiment v2 (11-22, after GPT research review): does a phase-conditioned FM reward let the imitator
RE-SYNCHRONISE to an external beat after a phase jump, better than phase-AMP and same-information controls?
v2 changes: same-harmonic controls (concat4, film4), AMP with the same 4 harmonics, random clock jumps during training,
independent phase reference = foot touchdowns (not the Hilbert labeller), failures counted (no survivorship),
circular statistics, forward speed / fall / control cost reported. v1 kept as train_phase_v1_backup.py.

Policy observes env obs + [cos phi_c, sin phi_c];  phi_c(t) = phi_0 + 2*pi*t/T  (T = expert gait period, phi_0 random).
methods (all members stacked in one process; 5 seeds each):
  none_like_state   : unconditional prior + state correction (no timing information in the reward)
  concat_like_state : prior p(s,s'|phi) with sinusoidal phase concatenated
  rope_like_state   : prior p(s,s'|phi) with RoPE (hidden-pair rotation by w*phi, w in {1,2,3,4})
  rope_like_none    : RoPE prior without learner correction (ablation)
  amp_phase         : AMP discriminator on (s,s',cos phi,sin phi)
metrics: return, PLV between gait phase (PCA-Hilbert, same reference as expert labels) and clock, mean phase offset,
         after a clock jump of pi at t=500, and at tempo x0.85 / x1.15.
"""
import argparse, json, os, time, math
import numpy as np, torch, torch.nn as nn
from common import FEAT_DIM, make_env, LOG_INV_DELTA
from vec import BSAC, bmlp
from phase_util import load_expert_phase, BPhaseFlow, phase_feats, FootTD, NQ, wrap

p = argparse.ArgumentParser()
p.add_argument("--env", default="Walker2d-v5")
p.add_argument("--methods", default="none_like_state,concat4_like_state,film4_like_state,rope_like_state,amp_phase4")
p.add_argument("--jump_rate", type=float, default=1 / 300); p.add_argument("--tag", default="")
p.add_argument("--recent", type=int, default=20_000, help="_rwin learner: fit only the last N policy steps")
# reward suffixes: _state = expert window - learner STATE density (replay, last 100k)
#                  _rwin  = expert window - learner WINDOW density fitted on recent policy data only (11-22 redesign)
#                  _cond  = fixed transition log p_E(s'|s,phi);  _plain = fixed expert window density only
p.add_argument("--seeds", default="0,1,2,3,4"); p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--ntraj", type=int, default=10); p.add_argument("--out", default="E:/0.Project/fm34_env/phase")
p.add_argument("--eval_every", type=int, default=20_000); p.add_argument("--M", type=int, default=4)
args = p.parse_args()
torch.set_num_threads(1); torch.manual_seed(4321); np.random.seed(4321)
dev = torch.device("cuda")
methods = args.methods.split(","); seeds = [int(s) for s in args.seeds.split(",")]
members = [(m, s) for m in methods for s in seeds]; S = len(members)
k = FEAT_DIM[args.env]; os.makedirs(args.out, exist_ok=True)
Xtr, Ptr, Xho, Pho, mu, sd, EXP, ref, T = load_expert_phase(args.env, args.ntraj)
print("expert", {kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in EXP.items()}, "period", T, flush=True)
MU = torch.tensor(np.concatenate([mu, mu]), device=dev); SD = torch.tensor(np.concatenate([sd, sd]), device=dev)
XE = (torch.tensor(Xtr, device=dev) - MU) / SD; PE = torch.tensor(Ptr, device=dev)
envs = [make_env(args.env, True) for _ in members]; eval_envs = [make_env(args.env, False) for _ in members]
od0, ad = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]; od = od0 + 2
COND = {m: m.split("_")[0] for m in methods if not m.startswith("amp")}
AMPS = [m for m in methods if m.startswith("amp")]; assert len(AMPS) <= 1
AK = 4 if (AMPS and AMPS[0].endswith("4")) else 1


def pfe(phi):
    return phase_feats(phi, AK)


def ids(pred):
    return torch.tensor([i for i, (m, _) in enumerate(members) if pred(m)], device=dev, dtype=torch.long)


def t_draw(shape, like):
    if like:
        s = torch.rand(shape, device=dev) * LOG_INV_DELTA; t = 1 - torch.exp(-s); return t, LOG_INV_DELTA * t
    return torch.rand(shape, device=dev), torch.ones(shape, device=dev)


def loss_ps(net, x1, phi, like, M):
    """per-sample weighted residual, x1 (S,B,d) phi (S,B) -> (S,B)"""
    Sx, B, d = x1.shape
    xr, pr = x1.repeat(1, M, 1), phi.repeat(1, M)
    t, w = t_draw((Sx, B * M), like); x0 = torch.randn_like(xr)
    xt = (1 - t[..., None]) * x0 + t[..., None] * xr
    return (((net(xt, t, pr) - (xr - x0)) ** 2).sum(-1) * w).view(Sx, M, B).mean(1)


def loss_train(net, x1, phi):
    Sx, B, d = x1.shape
    t = torch.where(torch.rand(Sx, B, device=dev) < 0.5, torch.rand(Sx, B, device=dev),
                    1 - torch.exp(-torch.rand(Sx, B, device=dev) * LOG_INV_DELTA))
    x0 = torch.randn_like(x1); xt = (1 - t[..., None]) * x0 + t[..., None] * x1
    return ((net(xt, t, phi) - (x1 - x0)) ** 2).sum(-1).mean(1).sum()


# expert priors (one per conditioning type), cached
experts = {}
for c in sorted(set(COND.values())):
    path = f"E:/0.Project/fm34_env/cache/{args.env}_n{args.ntraj}_phaseflow_{c}.pt"
    net = BPhaseFlow(1, 2 * k, c.replace("shuf", "")).to(dev)      # '*shuf' = same net, prior fitted on SHUFFLED phase labels
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        PEc = PE[torch.tensor(np.random.default_rng(0).permutation(len(PE)), device=dev)] if c.endswith("shuf") else PE
        opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30_000)
        for i in range(30_000):
            j = torch.randint(0, XE.shape[0], (1, 1024), device=dev)
            l = loss_train(net, XE[j], PEc[j]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        torch.save(net.state_dict(), path)
    net.requires_grad_(False); experts[c] = net

# expert STATE priors p_E(s|phi) for the fixed transition reward log p_E(s'|s,phi) = log p_E(s,s'|phi) - log p_E(s|phi)
# (methods "*_like_cond": no online learner model; GPT 11-22 cheap baseline)
experts_s = {}
for c in sorted({COND[m] for m in methods if m.endswith("_cond")}):
    path = f"E:/0.Project/fm34_env/cache/{args.env}_n{args.ntraj}_phaseflow_state_{c}.pt"
    net = BPhaseFlow(1, k, c).to(dev)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30_000)
        for i in range(30_000):
            j = torch.randint(0, XE.shape[0], (1, 1024), device=dev)
            l = loss_train(net, XE[j][..., :k], PE[j]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        torch.save(net.state_dict(), path)
    net.requires_grad_(False); experts_s[c] = net

groups = {}   # method -> member indices
for m in methods:
    groups[m] = ids(lambda mm, m=m: mm == m)
learners = {m: BPhaseFlow(len(groups[m]), k, COND[m]).to(dev) for m in methods if m.endswith("_state")}
learners.update({m: BPhaseFlow(len(groups[m]), 2 * k, COND[m]).to(dev) for m in methods if m.endswith("_rwin")})
LDIM = {m: (2 * k if m.endswith("_rwin") else k) for m in learners}
opt_l = torch.optim.Adam([q for L in learners.values() for q in L.parameters()], 3e-4) if learners else None
I_amp = ids(lambda m: m.startswith("amp"))
disc = bmlp(len(I_amp), [2 * k + 2 * AK, 256, 256, 1], nn.ReLU).to(dev) if len(I_amp) else None
opt_d = torch.optim.Adam(disc.parameters(), 1e-4) if disc is not None else None
RS_m = torch.zeros(S, 1, device=dev); RS_v = torch.ones(S, 1, device=dev); RS_n = [0]


def reward(x, phi):
    with torch.no_grad():
        r = torch.zeros(x.shape[:2], device=dev)
        RS_n[0] += 1; b = min(0.999, 1 - 1 / RS_n[0])
        for m in methods:
            I = groups[m]
            if m.startswith("amp"):
                z = torch.cat([x[I], pfe(phi[I])], -1)
                r[I] = torch.clamp(1 - 0.25 * (disc(z).squeeze(-1) - 1) ** 2, min=0); continue
            like = "_like_" in m; ex = experts[COND[m]]
            n, B = len(I), x.shape[1]
            rr = -loss_ps(ex, x[I].reshape(1, n * B, -1), phi[I].reshape(1, n * B), like, args.M).view(n, B)
            if m in learners:
                rr = rr + loss_ps(learners[m], x[I][..., :LDIM[m]], phi[I], like, args.M)
            elif m.endswith("_cond"):
                rr = rr + loss_ps(experts_s[COND[m]], x[I][..., :k].reshape(1, n * B, -1), phi[I].reshape(1, n * B), like, args.M).view(n, B)
            RS_m[I] = b * RS_m[I] + (1 - b) * rr.mean(1, keepdim=True); RS_v[I] = b * RS_v[I] + (1 - b) * rr.var(1, keepdim=True)
            r[I] = (rr - RS_m[I]) / (RS_v[I].sqrt() + 1e-6)
        return r


NB = args.steps
B_o = torch.zeros(S, NB, od, device=dev); B_a = torch.zeros(S, NB, ad, device=dev); B_o2 = torch.zeros(S, NB, od, device=dev)
B_d = torch.zeros(S, NB, device=dev); B_x = torch.zeros(S, NB, 2 * k, device=dev); B_p = torch.zeros(S, NB, device=dev)
agent = BSAC(S, od, ad, dev); ar = torch.arange(S, device=dev)[:, None]
g = lambda buf, idx, rows=None: buf[ar if rows is None else rows[:, None], idx]
W = 2 * math.pi / T


def aug(o, ph):
    return np.concatenate([o, [math.cos(ph), math.sin(ph)]]).astype(np.float32)


FT = [FootTD(ev, args.env) for ev in eval_envs]; TOL = EXP["td_tol"]; TDP = EXP["td_phase"]; VXE = EXP["vx"]; IVX = NQ[args.env]
JUMPS = [(t_, d_) for t_ in (300, 600) for d_ in (math.pi / 2, -math.pi / 2, math.pi)]


def evaluate(n_ep, tempo=1.0, jumps=False):
    """per member, per episode: return, fell, forward speed, ctrl cost, touchdown phase error vs clock (independent of
    the Hilbert labeller), lock fraction, and (jumps) resync time in gait cycles; failures are kept (no survivorship)."""
    recs = [[] for _ in range(S)]
    for e in range(n_ep):
        jt, dj = JUMPS[e % len(JUMPS)] if jumps else (None, 0.0)
        clock = lambda t, ph0: ph0 + W * tempo * t + (dj if (jt is not None and t >= jt) else 0.0)
        obs = [ev.reset(seed=20_000 + e)[0] for ev in eval_envs]; ph0 = 2 * math.pi * ((e * 0.37) % 1.0)
        Z = [[] for _ in range(S)]; CL = [[] for _ in range(S)]; VX = [[] for _ in range(S)]
        R = np.zeros(S); C = np.zeros(S); fell = [False] * S; alive = [True] * S; t = 0
        while any(alive):
            A = agent.act(torch.tensor(np.array([aug(o, clock(t, ph0)) for o in obs]), device=dev), det=True).cpu().numpy()
            t += 1
            for i in range(S):
                if not alive[i]:
                    continue
                o, r, te, tr, info = eval_envs[i].step(A[i]); R[i] += r; C[i] += info.get("reward_ctrl", 0.0)
                Z[i].append(FT[i].z_live(eval_envs[i].unwrapped.data)); CL[i].append(clock(t, ph0)); VX[i].append(o[IVX])
                obs[i] = o
                if te:
                    fell[i] = True
                if te or tr:
                    alive[i] = False
        for i in range(S):
            evs = FT[i].events(np.array(Z[i])); CLi = np.array(CL[i])
            tds = sorted([(int(ix) + 1, float(wrap(CLi[ix] - TDP[j]))) for j, ev in enumerate(evs) for ix in ev])
            vx = float(np.mean(VX[i])); walking = bool(vx >= 0.5 * VXE)
            pre = [abs(er) for tt, er in tds if tt >= 100 and (jt is None or tt < jt)]
            rec = dict(ret=float(R[i]), fell=fell[i], len=len(Z[i]), vx=vx, ctrl=float(C[i]), walking=walking,
                       td_abs_err=(float(np.mean(pre)) if pre else None),
                       lock=(float(np.mean(np.array(pre) < TOL)) if (pre and walking) else 0.0))
            if jt is not None:
                post = [(tt, abs(er)) for tt, er in tds if tt >= jt]; rs = None
                for q in range(len(post) - 2):
                    if all(post[q + u][1] < TOL for u in range(3)):
                        rs = (post[q][0] - jt) / T; break
                rec.update(jump=(jt, dj), resync_cycles=rs, success=bool((not fell[i]) and walking and rs is not None))
            recs[i].append(rec)
    out = []
    for i in range(S):
        L = recs[i]; d = {}
        for kk in ("ret", "vx", "ctrl", "len"):
            d[kk] = float(np.mean([r[kk] for r in L]))
        d["fall_rate"] = float(np.mean([r["fell"] for r in L])); d["walk_rate"] = float(np.mean([r["walking"] for r in L]))
        d["lock"] = float(np.mean([r["lock"] for r in L]))
        te_ = [r["td_abs_err"] for r in L if r["td_abs_err"] is not None]; d["td_abs_err"] = float(np.mean(te_)) if te_ else None
        if jumps:
            d["success_rate"] = float(np.mean([r["success"] for r in L]))
            d["resync_median_cycles"] = float(np.median([r["resync_cycles"] if r["success"] else np.inf for r in L]))
        d["episodes"] = L
        out.append(d)
    return out


names = [f"{args.env}{args.tag}_{m}_s{s}" for m, s in members]
logs = [dict(args=dict(vars(args), method=m, seed=s), expert={kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in EXP.items()}, period=T, evals=[]) for m, s in members]
obs = [e.reset(seed=1000 * j + s)[0] for j, (e, (m, s)) in enumerate(zip(envs, members))]
phs = np.random.uniform(0, 2 * math.pi, S); t0 = time.time(); START = 5000; alpha = None
for step in range(args.steps):
    Oa = np.array([aug(o, phs[i]) for i, o in enumerate(obs)])
    A = np.stack([e.action_space.sample() for e in envs]) if step < START else \
        agent.act(torch.tensor(Oa, device=dev)).cpu().numpy()
    O2, D, nxt, ph2 = [], [], [], phs + W
    jump = (np.random.rand(S) < args.jump_rate) * np.random.choice([-1, 1], S) * np.random.uniform(math.pi / 4, math.pi, S)
    for i in range(S):
        o2, r, te, tr, _ = envs[i].step(A[i]); D.append(float(te))
        if not (te or tr):
            ph2[i] += jump[i]                      # external beat jumps mid-episode: learn to re-synchronise
        O2.append(aug(o2, ph2[i]))
        if te or tr:
            nxt.append(envs[i].reset()[0]); ph2[i] = np.random.uniform(0, 2 * math.pi)
        else:
            nxt.append(o2)
    Ot = torch.tensor(Oa, device=dev); O2t = torch.tensor(np.array(O2), device=dev)
    B_o[:, step] = Ot; B_a[:, step] = torch.tensor(A, dtype=torch.float32, device=dev); B_o2[:, step] = O2t
    B_d[:, step] = torch.tensor(D, device=dev); B_p[:, step] = torch.tensor(phs, dtype=torch.float32, device=dev)
    B_x[:, step] = (torch.cat([Ot[:, :k], O2t[:, :k]], 1) - MU) / SD
    obs = nxt; phs = np.mod(ph2, 2 * math.pi)
    if step >= START:
        lo = max(0, step - 100_000)
        if opt_l is not None and step % 1000 == 0:
            for _ in range(2000 if step == START else 250):
                l = 0
                for m, L in learners.items():
                    I = groups[m]; lo_m = max(0, step - args.recent) if m.endswith("_rwin") else lo
                    j = torch.randint(lo_m, step + 1, (len(I), 512), device=dev)
                    l = l + loss_train(L, g(B_x, j, I)[..., :LDIM[m]], g(B_p, j, I))
                opt_l.zero_grad(); l.backward(); opt_l.step()
        if disc is not None and step % 2 == 0:
            je = torch.randint(0, XE.shape[0], (len(I_amp), 256), device=dev)
            xe = torch.cat([XE[je], pfe(PE[je])], -1).requires_grad_(True)
            jp = torch.randint(lo, step + 1, (len(I_amp), 256), device=dev); pp = g(B_p, jp, I_amp)
            xp = torch.cat([g(B_x, jp, I_amp), pfe(pp)], -1)
            de, dp = disc(xe), disc(xp)
            gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(-1).mean(1)
            ld = (((de - 1) ** 2).mean((1, 2)) + ((dp + 1) ** 2).mean((1, 2)) + 10 * gp).sum()
            opt_d.zero_grad(); ld.backward(); opt_d.step()
        idx = torch.randint(0, step + 1, (S, 256), device=dev)
        r = reward(g(B_x, idx), g(B_p, idx))
        alpha = agent.update(g(B_o, idx), g(B_a, idx), r, g(B_o2, idx), g(B_d, idx))
    if (step + 1) % args.eval_every == 0:
        ev = evaluate(3); sps = (step + 1) / (time.time() - t0)
        for i in range(S):
            logs[i]["evals"].append(dict(step=step + 1, sps=sps, **{kk: v for kk, v in ev[i].items() if kk != "episodes"}))
            json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
        f = lambda m, kk: np.mean([ev[i][kk] for i in groups[m].tolist()])
        print(f"[{args.env}] {step+1:>7d} sps {sps:.0f} | " + " ".join(f"{m}:R{f(m,'ret'):.0f}/vx{f(m,'vx'):.2f}/lock{f(m,'lock'):.2f}" for m in methods), flush=True)

final = {"nominal": evaluate(10), "jump": evaluate(24, jumps=True), "tempo0.85": evaluate(10, tempo=0.85), "tempo1.15": evaluate(10, tempo=1.15)}
for i in range(S):
    logs[i]["final"] = {kk: v[i] for kk, v in final.items()}
    logs[i]["final"]["minutes"] = (time.time() - t0) / 60
    json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
for m in methods:
    f = lambda kk, part: np.mean([final[part][i][kk] for i in groups[m].tolist()])
    print(f"FINAL {m}: R {f('ret','nominal'):.0f} vx {f('vx','nominal'):.2f} fall {f('fall_rate','nominal'):.2f} lock {f('lock','nominal'):.2f} | "
          f"jump success {f('success_rate','jump'):.2f} resync_med {np.median([final['jump'][i]['resync_median_cycles'] for i in groups[m].tolist()]):.2f} cyc", flush=True)
torch.save(dict(actor=agent.actor.state_dict(), members=members), os.path.join(args.out, f"{args.env}{args.tag}_actors.pt"))
print("FINAL", args.env, flush=True)
