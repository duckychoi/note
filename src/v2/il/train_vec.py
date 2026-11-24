"""
Seed-batched version of train.py (same methods, same rewards, S seeds in one process).
Writes one JSON (+ feats npz) per seed with the same schema as train.py so aggregate.py works.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from common import FEAT_DIM, make_env, load_expert, FlowNet, fm_train_loss
from vec import BSAC, BFlow, bmlp, bfm_loss_per_sample, bfm_train_loss

p = argparse.ArgumentParser()
p.add_argument("--env", default="Hopper-v5"); p.add_argument("--method", default="fm_like_state")
p.add_argument("--seeds", default="0,1,2,3,4"); p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--ntraj", type=int, default=10); p.add_argument("--out", default="E:/0.Project/fm34_env/runs")
p.add_argument("--eval_every", type=int, default=20_000); p.add_argument("--M", type=int, default=4)
args = p.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]; S = len(seeds)
torch.set_num_threads(1); torch.manual_seed(1000 + seeds[0]); np.random.seed(seeds[0])
dev = torch.device("cuda")
k = FEAT_DIM[args.env]; meth = args.method
os.makedirs(args.out, exist_ok=True)
envs = [make_env(args.env, True) for _ in seeds]; eval_envs = [make_env(args.env, False) for _ in seeds]
od, ad = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]
X_tr, X_ho, mu, sd, exp_ret = load_expert(args.env, args.ntraj, seed=0)
MU = torch.tensor(np.concatenate([mu, mu]), device=dev); SD = torch.tensor(np.concatenate([sd, sd]), device=dev)
XE = (torch.tensor(X_tr, device=dev) - MU) / SD
XH = (torch.tensor(X_ho, device=dev) - MU) / SD
wmode = meth.split("_")[1] if meth.startswith("fm_") else "unif"
corr = meth.split("_")[2] if meth.startswith("fm_") else "none"

expert = None
if meth.startswith("fm_") or meth == "smp":
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

learner = None
if corr in ("state", "window"):
    learner = BFlow(S, k if corr == "state" else 2 * k).to(dev); opt_l = torch.optim.Adam(learner.parameters(), 3e-4)
disc = None
if meth == "amp":
    disc = bmlp(S, [2 * k, 256, 256, 1], nn.ReLU).to(dev); opt_d = torch.optim.Adam(disc.parameters(), 1e-4)

RS_m = torch.zeros(S, 1, device=dev); RS_v = torch.ones(S, 1, device=dev); RS_n = [0]


def rs_update(x):
    RS_n[0] += 1; b = min(0.999, 1 - 1 / RS_n[0])
    RS_m.mul_(b).add_((1 - b) * x.mean(1, keepdim=True)); RS_v.mul_(b).add_((1 - b) * x.var(1, keepdim=True))


def reward(x):                              # x (S,B,2k)
    with torch.no_grad():
        if meth == "amp":
            return torch.clamp(1 - 0.25 * (disc(x).squeeze(-1) - 1) ** 2, min=0)
        if meth == "smp":
            lu = bfm_loss_per_sample(expert, x, "unif", args.M, shared=True); rs_update(lu)
            return torch.exp(-lu / RS_m.clamp(min=1e-6))
        r = -bfm_loss_per_sample(expert, x, wmode, args.M, shared=True)
        if corr == "state":
            r = r + bfm_loss_per_sample(learner, x[..., :k], wmode, args.M)
        elif corr == "window":
            r = r + bfm_loss_per_sample(learner, x, wmode, args.M)
        rs_update(r)
        return (r - RS_m) / (RS_v.sqrt() + 1e-6)


N = args.steps
B_o = torch.zeros(S, N, od, device=dev); B_a = torch.zeros(S, N, ad, device=dev); B_o2 = torch.zeros(S, N, od, device=dev)
B_r = torch.zeros(S, N, device=dev); B_d = torch.zeros(S, N, device=dev); B_x = torch.zeros(S, N, 2 * k, device=dev)
agent = BSAC(S, od, ad, dev)
ar = torch.arange(S, device=dev)[:, None]


def gather(buf, idx):                       # buf (S,N,·), idx (S,B)
    return buf[ar, idx]


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


logs = [dict(args=dict(vars(args), seed=s), expert_return=exp_ret, evals=[]) for s in seeds]
names = [f"{args.env}_{meth}_s{s}" for s in seeds]
obs = [e.reset(seed=s)[0] for e, s in zip(envs, seeds)]
t0 = time.time(); START = 5000; alpha = None
for step in range(N):
    if step < START:
        A = np.stack([e.action_space.sample() for e in envs])
    else:
        A = agent.act(torch.tensor(np.array(obs), dtype=torch.float32, device=dev)).cpu().numpy()
    O2, R, D = [], [], []
    for i in range(S):
        o2, r, te, tr, _ = envs[i].step(A[i]); O2.append(o2); R.append(r); D.append(float(te))
        if te or tr:
            o2r, _ = envs[i].reset()
        else:
            o2r = o2
        O2[-1] = (o2, o2r)
    Ot = torch.tensor(np.array(obs), dtype=torch.float32, device=dev)
    O2t = torch.tensor(np.array([x[0] for x in O2]), dtype=torch.float32, device=dev)
    B_o[:, step] = Ot; B_a[:, step] = torch.tensor(A, dtype=torch.float32, device=dev); B_o2[:, step] = O2t
    B_r[:, step] = torch.tensor(R, dtype=torch.float32, device=dev); B_d[:, step] = torch.tensor(D, device=dev)
    B_x[:, step] = (torch.cat([Ot[:, :k], O2t[:, :k]], 1) - MU) / SD
    obs = [x[1] for x in O2]

    if step >= START:
        lo = max(0, step - 100_000)
        if learner is not None and step % 1000 == 0:
            for _ in range(2000 if step == START else 250):
                b = gather(B_x, torch.randint(lo, step + 1, (S, 512), device=dev))
                l = bfm_train_loss(learner, b[..., :k] if corr == "state" else b)
                opt_l.zero_grad(); l.backward(); opt_l.step()
        if disc is not None and step % 2 == 0:
            xe = XE[torch.randint(0, XE.shape[0], (S, 256), device=dev)].requires_grad_(True)
            xp = gather(B_x, torch.randint(lo, step + 1, (S, 256), device=dev))
            de, dp = disc(xe), disc(xp)
            gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(-1).mean(1)
            ld = (((de - 1) ** 2).mean((1, 2)) + ((dp + 1) ** 2).mean((1, 2)) + 10 * gp).sum()
            opt_d.zero_grad(); ld.backward(); opt_d.step()
        idx = torch.randint(0, step + 1, (S, 256), device=dev)
        r = gather(B_r, idx) if meth == "sac_true" else reward(gather(B_x, idx))
        alpha = agent.update(gather(B_o, idx), gather(B_a, idx), r, gather(B_o2, idx), gather(B_d, idx))

    if (step + 1) % args.eval_every == 0:
        rets, _ = evaluate(5); sps = (step + 1) / (time.time() - t0)
        for i in range(S):
            logs[i]["evals"].append(dict(step=step + 1, ret=float(rets[i]), sps=sps,
                                         alpha=float(alpha[i]) if alpha is not None else None))
            json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
        print(f"[{args.env}_{meth}] {step+1:>7d} ret {np.round(rets,0)} (expert {exp_ret:.0f}) sps {sps:.0f}", flush=True)

rets, FE = evaluate(10, collect=True)
for i in range(S):
    mmd, sw2 = dist_metrics(FE[i])
    logs[i]["final"] = dict(ret=float(rets[i]), norm_ret=float(rets[i] / exp_ret), mmd=mmd, sw2=sw2,
                            minutes=(time.time() - t0) / 60)
    json.dump(logs[i], open(os.path.join(args.out, names[i] + ".json"), "w"))
    np.savez_compressed(os.path.join(args.out, names[i] + "_feats.npz"), F=FE[i])
torch.save(agent.actor.state_dict(), os.path.join(args.out, f"{args.env}_{meth}_actor_seeds{args.seeds}.pt"))
print("FINAL", args.env, meth, [round(l["final"]["ret"], 1) for l in logs], flush=True)
