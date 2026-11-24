"""
State-only imitation on MuJoCo v5 with generative-prior rewards (SAC learner).

methods
  sac_true                 : env reward (RL sanity check)
  amp                      : LSGAN discriminator + grad penalty on (s,s'), r=max(0,1-0.25(D-1)^2)
  smp                      : fixed expert flow, r = exp(-l_unif(x)/running_mean)     (SMP-style)
  fm_<w>_<c>               : w in {unif, like}  (time weight of the residual)
                             c in {none, state, window}  (learner-density correction)
                             r = -l_w^E(s,s') + l_w^L(s)  [state]   |  + l_w^L(s,s') [window]
                             (= log p(s,s') - log d(s)  or  log p - log d(s,s')  up to constants)
Training env never terminates on falling (no survival bias); evaluation uses the standard env.
"""
import argparse, json, os, time, math
import numpy as np, torch, torch.nn as nn
from common import (FEAT_DIM, make_env, load_expert, FlowNet, fm_loss_per_sample, fm_train_loss,
                    SAC, RunningStat)

p = argparse.ArgumentParser()
p.add_argument("--env", default="Hopper-v5"); p.add_argument("--method", default="fm_like_state")
p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--ntraj", type=int, default=10); p.add_argument("--out", default="E:/0.Project/fm34_env/runs")
p.add_argument("--eval_every", type=int, default=10_000); p.add_argument("--M", type=int, default=4)
p.add_argument("--device", default="cuda")
args = p.parse_args()

torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device(args.device)
k = FEAT_DIM[args.env]
run_name = f"{args.env}_{args.method}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
log_path = os.path.join(args.out, run_name + ".json")

env, eval_env = make_env(args.env, True), make_env(args.env, False)
od, ad = env.observation_space.shape[0], env.action_space.shape[0]
X_tr, X_ho, mu, sd, exp_ret = load_expert(args.env, args.ntraj, seed=0)
MU = torch.tensor(np.concatenate([mu, mu]), device=dev); SD = torch.tensor(np.concatenate([sd, sd]), device=dev)
XE = (torch.tensor(X_tr, device=dev) - MU) / SD
XH = (torch.tensor(X_ho, device=dev) - MU) / SD

meth = args.method
wmode = meth.split("_")[1] if meth.startswith("fm_") else "unif"
corr = meth.split("_")[2] if meth.startswith("fm_") else "none"

# ---------------- expert flow (cached, shared across methods/seeds)
expert = None
if meth.startswith("fm_") or meth == "smp":
    cpath = f"E:/0.Project/fm34_env/cache/{args.env}_n{args.ntraj}_flow.pt"
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    expert = FlowNet(2 * k).to(dev)
    if os.path.exists(cpath):
        expert.load_state_dict(torch.load(cpath))
    else:
        g = torch.Generator(device=dev).manual_seed(0)
        opt = torch.optim.Adam(expert.parameters(), 1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30_000)
        for i in range(30_000):
            b = XE[torch.randint(0, XE.shape[0], (1024,), device=dev, generator=g)]
            l = fm_train_loss(expert, b); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        torch.save(expert.state_dict(), cpath)
    expert.requires_grad_(False)

learner = None
if corr in ("state", "window"):
    learner = FlowNet(k if corr == "state" else 2 * k).to(dev)
    opt_l = torch.optim.Adam(learner.parameters(), 3e-4)

disc = None
if meth == "amp":
    disc = nn.Sequential(nn.Linear(2 * k, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    opt_d = torch.optim.Adam(disc.parameters(), 1e-4)

rs = RunningStat()


def reward(x):
    """x: normalized windows (B,2k) -> reward (B,)."""
    with torch.no_grad():
        if meth == "amp":
            d = disc(x).squeeze(1)
            return torch.clamp(1 - 0.25 * (d - 1) ** 2, min=0)
        if meth == "smp":
            lu = fm_loss_per_sample(expert, x, "unif", M=args.M)
            rs.update(lu)
            return torch.exp(-lu / max(rs.m, 1e-6))
        r = -fm_loss_per_sample(expert, x, wmode, M=args.M)
        if corr == "state":
            r = r + fm_loss_per_sample(learner, x[:, :k], wmode, M=args.M)
        elif corr == "window":
            r = r + fm_loss_per_sample(learner, x, wmode, M=args.M)
        rs.update(r)
        return (r - rs.m) / rs.sd


# ---------------- buffer
N = args.steps
B_o = torch.zeros(N, od, device=dev); B_a = torch.zeros(N, ad, device=dev)
B_o2 = torch.zeros(N, od, device=dev); B_r = torch.zeros(N, device=dev); B_d = torch.zeros(N, device=dev)
B_x = torch.zeros(N, 2 * k, device=dev)
agent = SAC(od, ad, dev)


def evaluate(n_ep, collect=False):
    rets, feats = [], []
    for e in range(n_ep):
        o, _ = eval_env.reset(seed=10_000 + e); done = False; R = 0.0; prev = o[:k]
        while not done:
            a = agent.act(torch.tensor(o, dtype=torch.float32, device=dev), det=True)
            o, r, te, tr, _ = eval_env.step(a); R += r; done = te or tr
            if collect:
                feats.append(np.concatenate([prev, o[:k]])); prev = o[:k]
        rets.append(R)
    return float(np.mean(rets)), (np.array(feats, np.float32) if collect else None)


def dist_metrics(F_):
    """MMD (RBF, median bandwidth) and sliced W2 between policy and held-out expert windows."""
    P = (torch.tensor(F_, device=dev) - MU) / SD
    Q = XH[torch.randperm(XH.shape[0], device=dev)[:4000]]
    P = P[torch.randperm(P.shape[0], device=dev)[:4000]]
    Z = torch.cat([P, Q]); D = torch.cdist(Z, Z) ** 2
    h = D.median()
    K = torch.exp(-D / h); n = P.shape[0]
    mmd = K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
    th = torch.randn(2 * k, 256, device=dev); th /= th.norm(dim=0, keepdim=True)
    m = min(P.shape[0], Q.shape[0])
    pp = (P[:m] @ th).sort(0)[0]; qq = (Q[torch.randperm(Q.shape[0], device=dev)[:m]] @ th).sort(0)[0]
    sw2 = ((pp - qq) ** 2).mean().sqrt()
    return float(mmd), float(sw2)


log = dict(args=vars(args), expert_return=exp_ret, evals=[])
o, _ = env.reset(seed=args.seed); t0 = time.time(); ep_true = 0.0
START = 5000
for step in range(N):
    if step < START:
        a = env.action_space.sample()
    else:
        a = agent.act(torch.tensor(o, dtype=torch.float32, device=dev))
    o2, r_env, te, tr, _ = env.step(a); ep_true += r_env
    B_o[step] = torch.as_tensor(o, device=dev); B_a[step] = torch.as_tensor(a, device=dev)
    B_o2[step] = torch.as_tensor(o2, device=dev); B_r[step] = float(r_env); B_d[step] = float(te)
    B_x[step] = (torch.as_tensor(np.concatenate([o[:k], o2[:k]]), dtype=torch.float32, device=dev) - MU) / SD
    o = o2
    if te or tr:
        o, _ = env.reset(); ep_true = 0.0

    if step >= START:
        lo = max(0, step - 100_000)
        # reward-model updates
        if learner is not None and step % 1000 == 0:
            iters = 2000 if step == START else 250
            for _ in range(iters):
                b = B_x[torch.randint(lo, step + 1, (512,), device=dev)]
                l = fm_train_loss(learner, b[:, :k] if corr == "state" else b)
                opt_l.zero_grad(); l.backward(); opt_l.step()
        if disc is not None and step % 2 == 0:
            xe = XE[torch.randint(0, XE.shape[0], (256,), device=dev)].requires_grad_(True)
            xp = B_x[torch.randint(lo, step + 1, (256,), device=dev)]
            de, dp = disc(xe), disc(xp)
            gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(1).mean()
            ld = ((de - 1) ** 2).mean() + ((dp + 1) ** 2).mean() + 10 * gp
            opt_d.zero_grad(); ld.backward(); opt_d.step()
        # SAC update
        i = torch.randint(0, step + 1, (256,), device=dev)
        r = B_r[i] if meth == "sac_true" else reward(B_x[i])
        lc, alpha = agent.update(B_o[i], B_a[i], r, B_o2[i], B_d[i])

    if (step + 1) % args.eval_every == 0:
        ret, _ = evaluate(5)
        log["evals"].append(dict(step=step + 1, ret=ret, alpha=float(alpha) if step >= START else None,
                                 sps=(step + 1) / (time.time() - t0)))
        print(f"[{run_name}] {step+1:>7d} ret {ret:8.1f} (expert {exp_ret:.0f}) sps {(step+1)/(time.time()-t0):.0f}",
              flush=True)
        json.dump(log, open(log_path, "w"))

ret, F_ = evaluate(10, collect=True)
np.savez_compressed(os.path.join(args.out, run_name + "_feats.npz"), F=F_)
mmd, sw2 = dist_metrics(F_)
log["final"] = dict(ret=ret, norm_ret=ret / exp_ret, mmd=mmd, sw2=sw2, minutes=(time.time() - t0) / 60)
json.dump(log, open(log_path, "w"))
torch.save(agent.actor.state_dict(), os.path.join(args.out, run_name + "_actor.pt"))
print("FINAL", run_name, log["final"], flush=True)
