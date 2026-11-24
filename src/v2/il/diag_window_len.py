"""
Diagnosis 2: does a LONGER motion window fix the 'standing is rated above walking' failure of the density reward?
Hypothesis: in 2-frame windows joint velocities oscillate around 0, so a motionless pose sits near the 'ring centre'
of the expert distribution (high smoothed density). An H-frame window of a standing robot is constant in time,
which expert motion never is -> should be far out of distribution.
For H in {2,4,8,16}: fit an FM prior on expert H-frame windows (10 train episodes, same normalisation), then score
held-out expert / medium / simple / stand / random H-windows with the likelihood-weighted residual (delta 1e-2).
Metrics as diag_reward_landscape.py. Results -> results/v2/diag_window_<env>.json
"""
import os, json, math, argparse
import numpy as np, torch
from scipy.stats import spearmanr
from common import FEAT_DIM, DATASET, FlowNet, make_env
from diag_reward_landscape import residual, auc, NQ

p = argparse.ArgumentParser(); p.add_argument("--envs", default="Hopper-v5,Walker2d-v5,HalfCheetah-v5")
p.add_argument("--Hs", default="2,4,8,16"); p.add_argument("--n", type=int, default=3000); p.add_argument("--iters", type=int, default=20000)
args = p.parse_args(); dev = torch.device("cuda"); torch.manual_seed(0)


def hwin(O, k, H):
    T = len(O) - H + 1
    if T <= 0:
        return None
    return np.stack([O[i:i + T, :k] for i in range(H)], 1).reshape(T, H * k)


def ds_seqs(env_id, q):
    import minari
    return [e.observations for e in minari.load_dataset(f"mujoco/{DATASET[env_id]}/{q}-v0").iterate_episodes()]


def roll_seqs(env_id, kind, n_steps, rng):
    env = make_env(env_id, kind == "random"); out = []; tot = 0
    while tot < n_steps:
        o = env.reset(seed=int(rng.integers(1e6)))[0]; obs = [o]
        for t in range(300):
            a = np.zeros(env.action_space.shape) if kind == "stand" else env.action_space.sample()
            o, r, te, tr, _ = env.step(a); obs.append(o)
            if te or tr:
                break
        out.append(np.array(obs)); tot += len(obs)
    return out


def sample_windows(seqs, k, H, n, rng, speed_idx):
    X, V = [], []
    for O in seqs:
        w = hwin(O, k, H)
        if w is None:
            continue
        X.append(w); V.append(O[H - 1:H - 1 + len(w), speed_idx])
    X, V = np.concatenate(X), np.concatenate(V); j = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[j], V[j]


def fit(X, iters):
    net = FlowNet(X.shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters); Xt = torch.tensor(X, device=dev)
    for i in range(iters):
        x1 = Xt[torch.randint(0, len(Xt), (1024,), device=dev)]
        t = torch.where(torch.rand(1024, device=dev) < 0.5, torch.rand(1024, device=dev), 1 - torch.exp(-torch.rand(1024, device=dev) * math.log(1e3)))
        x0 = torch.randn_like(x1); xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        l = ((net(xt, t) - (x1 - x0)) ** 2).sum(-1).mean(); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    net.eval(); return net


def main():
    rng = np.random.default_rng(0)
    for env_id in args.envs.split(","):
        k = FEAT_DIM[env_id]; si = NQ[env_id]
        ex = ds_seqs(env_id, "expert"); perm = np.random.default_rng(0).permutation(len(ex))
        tr, ho = [ex[i] for i in perm[:10]], [ex[i] for i in perm[10:]]
        seqs = {"expert_ho": ho, "medium": ds_seqs(env_id, "medium"), "simple": ds_seqs(env_id, "simple"),
                "stand": roll_seqs(env_id, "stand", 4 * args.n, rng), "random": roll_seqs(env_id, "random", 4 * args.n, rng)}
        res = {}
        for H in [int(h) for h in args.Hs.split(",")]:
            Xtr = np.concatenate([hwin(O, k, H) for O in tr])
            mu = Xtr.reshape(-1, H, k).mean((0, 1)); sd = Xtr.reshape(-1, H, k).std((0, 1)) + 1e-3
            MU, SD = np.tile(mu, H), np.tile(sd, H)
            path = f"E:/0.Project/fm34_env/cache/{env_id}_n10_flowH{H}.pt"
            if os.path.exists(path):
                net = FlowNet(H * k).to(dev); net.load_state_dict(torch.load(path)); net.eval()
            else:
                net = fit(((Xtr - MU) / SD).astype(np.float32), args.iters); torch.save(net.state_dict(), path)
            S, Vs = {}, {}
            for c, sq in seqs.items():
                X, V = sample_windows(sq, k, H, args.n, rng, si)
                S[c] = -residual(net, torch.tensor((X - MU) / SD, dtype=torch.float32, device=dev), "like", 1e-2, 16).cpu().numpy() / H
                Vs[c] = V
            m0, s0 = S["expert_ho"].mean(), S["expert_ho"].std() + 1e-9
            z = {c: float(np.median((S[c] - m0) / s0)) for c in S}
            non = np.concatenate([S["simple"], S["medium"]]); vn = np.concatenate([Vs["simple"], Vs["medium"]])
            r = dict(median_z=z, spearman_speed_nonexpert=float(spearmanr(non, vn).correlation),
                     auc_expert_gt_medium=auc(S["expert_ho"], S["medium"]), auc_medium_gt_simple=auc(S["medium"], S["simple"]),
                     auc_medium_gt_stand=auc(S["medium"], S["stand"]), auc_simple_gt_stand=auc(S["simple"], S["stand"]))
            res[f"H{H}"] = r
            print(f"{env_id} H={H:2d} | z: " + " ".join(f"{c} {v:8.1f}" for c, v in z.items()) +
                  f" | rho {r['spearman_speed_nonexpert']:+.2f} | AUC E>M {r['auc_expert_gt_medium']:.2f} M>S {r['auc_medium_gt_simple']:.2f} "
                  f"M>stand {r['auc_medium_gt_stand']:.2f} S>stand {r['auc_simple_gt_stand']:.2f}", flush=True)
        od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
        json.dump(res, open(os.path.join(od, f"diag_window_{env_id}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
