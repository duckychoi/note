"""
All (method, seed) pairs of one env in ONE process: members are stacked on the batch dim S of every network,
so the per-step kernel count is ~independent of the number of runs (launch-bound workload).
Same methods/rewards/metrics/JSON schema as train.py / train_vec.py.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from common import FEAT_DIM, make_env, load_expert, FlowNet, fm_train_loss
from vec import BSAC, BFlow, bmlp, bfm_loss_mixed, bfm_train_loss

p = argparse.ArgumentParser()
p.add_argument("--env", default="Hopper-v5")
p.add_argument("--methods", default="sac_true,amp,smp,fm_unif_none,fm_unif_state,fm_unif_window,fm_like_none,fm_like_state,fm_like_window")
p.add_argument("--seeds", default="0,1,2,3,4"); p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--buf", type=int, default=300_000); p.add_argument("--ntraj", type=int, default=10)
p.add_argument("--out", default="E:/0.Project/fm34_env/runs"); p.add_argument("--eval_every", type=int, default=20_000)
p.add_argument("--M", type=int, default=4); p.add_argument("--bf16_obs", action="store_true")
args = p.parse_args()

torch.set_num_threads(1); torch.manual_seed(1234); np.random.seed(1234)
dev = torch.device("cuda")
methods = args.methods.split(","); seeds = [int(s) for s in args.seeds.split(",")]
members = [(m, s) for m in methods for s in seeds]; S = len(members)
k = FEAT_DIM[args.env]
os.makedirs(args.out, exist_ok=True)
envs = [make_env(args.env, True) for _ in members]; eval_envs = [make_env(args.env, False) for _ in members]
od, ad = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]
X_tr, X_ho, mu, sd, exp_ret = load_expert(args.env, args.ntraj, seed=0)
MU = torch.tensor(np.concatenate([mu, mu]), device=dev); SD = torch.tensor(np.concatenate([sd, sd]), device=dev)
XE = (torch.tensor(X_tr, device=dev) - MU) / SD; XH = (torch.tensor(X_ho, device=dev) - MU) / SD


def ids(pred):
    return torch.tensor([i for i, (m, _) in enumerate(members) if pred(m)], device=dev, dtype=torch.long)


I_true = ids(lambda m: m == "sac_true"); I_amp = ids(lambda m: m == "amp"); I_smp = ids(lambda m: m == "smp")
I_fm = ids(lambda m: m.startswith("fm_")); I_exp = ids(lambda m: m == "smp" or m.startswith("fm_"))
I_st = ids(lambda m: m.endswith("_state")); I_win = ids(lambda m: m.endswith("_window"))
LIKE = torch.tensor([m.startswith("fm_like") for m, _ in members], device=dev)[:, None]   # (S,1)
NORM = ids(lambda m: m.startswith("fm_"))

expert = None
if len(I_exp):
    cpath = f"E:/0.Project/fm34_env/cache/{args.env}_n{args.ntraj}_flow.pt"
    expert = FlowNet(2 * k).to(dev)
    if os.path.exists(cpath):
        expert.load_state_dict(torch.load(cpath))
    else:
        opt = torch.optim.Adam(expert.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30_000)
        for i in range(30_000):
            l = fm_train_loss(expert, XE[torch.randint(0, XE.shape[0], (1024,), device=dev)])
            opt.zero_grad(); l.backward(); opt.step(); sch.step()
        os.makedirs(os.path.dirname(cpath), exist_ok=True); torch.save(expert.state_dict(), cpath)
    expert.requires_grad_(False)
L_st = BFlow(len(I_st), k).to(dev) if len(I_st) else None
L_win = BFlow(len(I_win), 2 * k).to(dev) if len(I_win) else None
opt_l = torch.optim.Adam([q for L in (L_st, L_win) if L is not None for q in L.parameters()], 3e-4) if (len(I_st) or len(I_win)) else None
disc = bmlp(len(I_amp), [2 * k, 256, 256, 1], nn.ReLU).to(dev) if len(I_amp) else None
opt_d = torch.optim.Adam(disc.parameters(), 1e-4) if disc is not None else None

RS_m = torch.zeros(S, 1, device=dev); RS_v = torch.ones(S, 1, device=dev); RS_n = [0]
# map member index -> row inside learner groups
pos_st = torch.full((S,), -1, dtype=torch.long, device=dev); pos_st[I_st] = torch.arange(len(I_st), device=dev)
pos_win = torch.full((S,), -1, dtype=torch.long, device=dev); pos_win[I_win] = torch.arange(len(I_win), device=dev)


def rs_update(I, x):
    RS_n[0] += 1; b = min(0.999, 1 - 1 / RS_n[0])
    RS_m[I] = b * RS_m[I] + (1 - b) * x.mean(1, keepdim=True); RS_v[I] = b * RS_v[I] + (1 - b) * x.var(1, keepdim=True)


def reward(x, r_env):                      # x (S,B,2k)
    with torch.no_grad():
        r = torch.zeros(x.shape[:2], device=dev)
        if len(I_true):
            r[I_true] = r_env[I_true]
        if len(I_amp):
            r[I_amp] = torch.clamp(1 - 0.25 * (disc(x[I_amp]).squeeze(-1) - 1) ** 2, min=0)
        if len(I_exp):
            lE = bfm_loss_mixed(expert, x[I_exp], LIKE[I_exp], args.M, shared=True)
            lE_full = torch.zeros_like(r); lE_full[I_exp] = lE
            if len(I_smp):
                rs_update(I_smp, lE_full[I_smp])
                r[I_smp] = torch.exp(-lE_full[I_smp] / RS_m[I_smp].clamp(min=1e-6))
            rf = -lE_full
            if len(I_st):
                rf[I_st] += bfm_loss_mixed(L_st, x[I_st][..., :k], LIKE[I_st], args.M)
            if len(I_win):
                rf[I_win] += bfm_loss_mixed(L_win, x[I_win], LIKE[I_win], args.M)
            rs_update(NORM, rf[NORM])
            r[NORM] = (rf[NORM] - RS_m[NORM]) / (RS_v[NORM].sqrt() + 1e-6)
        return r


NB = args.buf
ODT = torch.bfloat16 if args.bf16_obs else torch.float32
B_o = torch.zeros(S, NB, od, device=dev, dtype=ODT); B_a = torch.zeros(S, NB, ad, device=dev); B_o2 = torch.zeros(S, NB, od, device=dev, dtype=ODT)
B_r = torch.zeros(S, NB, device=dev); B_d = torch.zeros(S, NB, device=dev); B_x = torch.zeros(S, NB, 2 * k, device=dev)
agent = BSAC(S, od, ad, dev)
ar = torch.arange(S, device=dev)[:, None]


def g(buf, idx, rows=None):
    return buf[ar if rows is None else rows[:, None], idx]


def sample_idx(n_rows, size, step, recent=None):
    filled = min(step + 1, NB)
    if recent is None or step + 1 <= NB:
        lo = 0 if recent is None else max(0, step + 1 - recent)
        return torch.randint(lo, step + 1, (n_rows, size), device=dev) % NB
    # ring buffer: most recent `recent` entries
    return (step + 1 - torch.randint(1, recent + 1, (n_rows, size), device=dev)) % NB


def evaluate(n_ep, collect=False):
    rets = np.zeros((S, n_ep)); feats = [[] for _ in range(S)]
    for e in range(n_ep):
        obs = [ev.reset(seed=10_000 + e)[0] for ev in eval_envs]; alive = [True] * S; prev = [o[:k] for o in obs]
        while any(alive):
            A = agent.act(torch.tensor(np.array(obs), dtype=torch.float32, device=dev), det=True).cpu().numpy()
            for i in range(S):
                if not alive[i]:
                    continue
                o, r, te, tr, _ = eval_envs[i].step(A[i]); rets[i, e] += r; obs[i] = o
                if collect:
                    feats[i].append(np.concatenate([prev[i], o[:k]])); prev[i] = o[:k]
                if te or tr:
                    alive[i] = False
    return rets.mean(1), [np.array(f, np.float32) for f in feats]


def dist_metrics(F_):
    P = (torch.tensor(F_, device=dev) - MU) / SD
    Q = XH[torch.randperm(XH.shape[0], device=dev)[:4000]]; P = P[torch.randperm(P.shape[0], device=dev)[:4000]]
    Z = torch.cat([P, Q]); D = torch.cdist(Z, Z) ** 2; K = torch.exp(-D / D.median()); n = P.shape[0]
    mmd = K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
    th = torch.randn(2 * k, 256, device=dev); th /= th.norm(dim=0, keepdim=True); m = min(P.shape[0], Q.shape[0])
    sw2 = (((P[:m] @ th).sort(0)[0] - (Q[torch.randperm(Q.shape[0], device=dev)[:m]] @ th).sort(0)[0]) ** 2).mean().sqrt()
    return float(mmd), float(sw2)


names = [f"{args.env}_{m}_s{s}" for m, s in members]
logs = [dict(args=dict(vars(args), method=m, seed=s), expert_return=exp_ret, evals=[]) for m, s in members]
obs = [e.reset(seed=1000 * j + s)[0] for j, (e, (m, s)) in enumerate(zip(envs, members))]
t0 = time.time(); START = 5000; alpha = None
for step in range(args.steps):
    j = step % NB
    if step < START:
        A = np.stack([e.action_space.sample() for e in envs])
    else:
        A = agent.act(torch.tensor(np.array(obs), dtype=torch.float32, device=dev)).cpu().numpy()
    O2, R, D, nxt = [], [], [], []
    for i in range(S):
        o2, r, te, tr, _ = envs[i].step(A[i]); O2.append(o2); R.append(r); D.append(float(te))
        nxt.append(envs[i].reset()[0] if (te or tr) else o2)
    Ot = torch.tensor(np.array(obs), dtype=torch.float32, device=dev)
    O2t = torch.tensor(np.array(O2), dtype=torch.float32, device=dev)
    B_o[:, j] = Ot.to(ODT); B_a[:, j] = torch.tensor(A, dtype=torch.float32, device=dev); B_o2[:, j] = O2t.to(ODT)
    B_r[:, j] = torch.tensor(R, dtype=torch.float32, device=dev); B_d[:, j] = torch.tensor(D, device=dev)
    B_x[:, j] = (torch.cat([Ot[:, :k], O2t[:, :k]], 1) - MU) / SD
    obs = nxt

    if step >= START:
        if opt_l is not None and step % 1000 == 0:
            for _ in range(2000 if step == START else 250):
                l = 0
                if L_st is not None:
                    l = l + bfm_train_loss(L_st, g(B_x, sample_idx(len(I_st), 512, step, 100_000), I_st)[..., :k])
                if L_win is not None:
                    l = l + bfm_train_loss(L_win, g(B_x, sample_idx(len(I_win), 512, step, 100_000), I_win))
                opt_l.zero_grad(); l.backward(); opt_l.step()
        if disc is not None and step % 2 == 0:
            xe = XE[torch.randint(0, XE.shape[0], (len(I_amp), 256), device=dev)].requires_grad_(True)
            xp = g(B_x, sample_idx(len(I_amp), 256, step, 100_000), I_amp)
            de, dp = disc(xe), disc(xp)
            gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(-1).mean(1)
            ld = (((de - 1) ** 2).mean((1, 2)) + ((dp + 1) ** 2).mean((1, 2)) + 10 * gp).sum()
            opt_d.zero_grad(); ld.backward(); opt_d.step()
        idx = sample_idx(S, 256, step)
        r = reward(g(B_x, idx), g(B_r, idx))
        alpha = agent.update(g(B_o, idx).float(), g(B_a, idx), r, g(B_o2, idx).float(), g(B_d, idx))

    if (step + 1) % args.eval_every == 0:
        rets, _ = evaluate(5); sps = (step + 1) / (time.time() - t0)
        for i in range(S):
            logs[i]["evals"].append(dict(step=step + 1, ret=float(rets[i]), sps=sps,
                                         alpha=float(alpha[i]) if alpha is not None else None))
            json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
        summ = {m: float(np.mean([rets[i] for i, (mm, _) in enumerate(members) if mm == m])) for m in methods}
        print(f"[{args.env}] {step+1:>7d} sps {sps:.0f} | " + " ".join(f"{m}:{v:.0f}" for m, v in summ.items()), flush=True)

rets, FE = evaluate(10, collect=True)
for i in range(S):
    mmd, sw2 = dist_metrics(FE[i])
    logs[i]["final"] = dict(ret=float(rets[i]), norm_ret=float(rets[i] / exp_ret), mmd=mmd, sw2=sw2,
                            minutes=(time.time() - t0) / 60)
    json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
    np.savez_compressed(os.path.join(args.out, names[i] + "_feats.npz"), F=FE[i])
torch.save(dict(actor=agent.actor.state_dict(), members=members), os.path.join(args.out, f"{args.env}_actors.pt"))
print("FINAL", args.env, flush=True)
