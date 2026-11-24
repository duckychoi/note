"""
Reward-landscape diagnosis (no RL): does the FM motion reward rank gaits by quality?

Gait quality ladder from Minari: simple-v0 < medium-v0 < expert-v0 (held-out episodes, not used to fit the prior),
plus synthetic 'stand' (zero action, standard env until it falls) and 'random' (uniform actions).
For every 2-frame window we compute several candidate rewards with the SAME cached expert FlowNet:
  unif      : -E_t~U ||v - (x1-x0)||^2                 (concept-note reward)
  like_d    : -likelihood-weighted residual, truncation delta d  (larger delta = smoother density, broader basin)
A good shaping reward must (i) increase along simple < medium < expert, (ii) correlate with forward speed
within the non-expert data (gradient to climb), (iii) not prefer 'stand' over moving non-expert gaits.
Output: per category mean reward (in expert-holdout std units, expert-holdout mean = 0), Spearman(reward, forward speed)
on simple+medium windows, AUCs. Results -> results/v2/diag_reward_<env>.json
"""
import os, sys, json, math, argparse
import numpy as np, torch
from scipy.stats import spearmanr
from common import FEAT_DIM, DATASET, FlowNet, load_expert, make_env

p = argparse.ArgumentParser(); p.add_argument("--envs", default="Hopper-v5,Walker2d-v5,HalfCheetah-v5")
p.add_argument("--n", type=int, default=3000); p.add_argument("--M", type=int, default=32)
args = p.parse_args()
dev = torch.device("cuda"); torch.manual_seed(0)
NQ = {"Hopper-v5": 5, "Walker2d-v5": 8, "HalfCheetah-v5": 8}     # obs index of forward speed qvel[0]


def residual(net, X, mode, delta, M):
    out = []
    for i in range(0, len(X), 2048):
        x1 = X[i:i + 2048]; B, d = x1.shape; xr = x1.repeat(M, 1)
        if mode == "unif":
            t = torch.rand(B * M, device=dev); w = torch.ones_like(t)
        else:
            L = math.log(1 / delta); s = torch.rand(B * M, device=dev) * L; t = 1 - torch.exp(-s); w = L * t
        x0 = torch.randn_like(xr); xt = (1 - t[:, None]) * x0 + t[:, None] * xr
        with torch.no_grad():
            r = ((net(xt, t) - (xr - x0)) ** 2).sum(-1) * w
        out.append(r.view(M, B).mean(0))
    return torch.cat(out)


def windows_from_dataset(env_id, q, k, n, rng, skip_ids=None):
    import minari
    ds = minari.load_dataset(f"mujoco/{DATASET[env_id]}/{q}-v0")
    X, V, R = [], [], []
    eps = list(ds.iterate_episodes())
    order = rng.permutation(len(eps))
    for i in order:
        if skip_ids is not None and i in skip_ids:
            continue
        o = eps[i].observations
        if len(o) < 3:
            continue
        X.append(np.concatenate([o[:-1, :k], o[1:, :k]], 1)); V.append(o[1:, NQ[env_id]]); R.append(eps[i].rewards)
        if sum(len(x) for x in X) > 4 * n:
            break
    X, V, R = np.concatenate(X), np.concatenate(V), np.concatenate(R)
    j = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[j], V[j], R[j]


def rollout_windows(env_id, k, n, rng, kind):
    env = make_env(env_id, kind == "random")        # random: no-termination env (includes fallen states)
    X, V = [], []
    while sum(len(x) for x in X) < n:
        o = env.reset(seed=int(rng.integers(1e6)))[0]; obs = [o]
        for t in range(300):
            a = np.zeros(env.action_space.shape) if kind == "stand" else env.action_space.sample()
            o, r, te, tr, _ = env.step(a); obs.append(o)
            if te or tr:
                break
        O = np.array(obs)
        if len(O) > 2:
            X.append(np.concatenate([O[:-1, :k], O[1:, :k]], 1)); V.append(O[1:, NQ[env_id]])
    X, V = np.concatenate(X)[:n], np.concatenate(V)[:n]
    return X, V


def auc(a, b):   # P(score(a) > score(b))
    a, b = np.asarray(a), np.asarray(b); idx = np.random.default_rng(0)
    A = a[idx.choice(len(a), 2000)]; B = b[idx.choice(len(b), 2000)]
    return float((A[:, None] > B[None]).mean())


def main():
    rng = np.random.default_rng(0)
    for env_id in args.envs.split(","):
        k = FEAT_DIM[env_id]
        X_tr, X_ho, mu, sd, _ = load_expert(env_id, 10, seed=0)
        MU = np.concatenate([mu, mu]); SD = np.concatenate([sd, sd])
        net = FlowNet(2 * k).to(dev); net.load_state_dict(torch.load(f"E:/0.Project/fm34_env/cache/{env_id}_n10_flow.pt")); net.eval()
        # expert held-out: load_expert uses permutation(seed 0) first 10 = train, next 10 = holdout
        cats = {}
        import minari
        eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/expert-v0").iterate_episodes())
        perm = np.random.default_rng(0).permutation(len(eps)); train_ids = set(perm[:10].tolist())
        cats["expert_ho"] = windows_from_dataset(env_id, "expert", k, args.n, rng, skip_ids=train_ids)[:2]
        cats["medium"] = windows_from_dataset(env_id, "medium", k, args.n, rng)[:2]
        cats["simple"] = windows_from_dataset(env_id, "simple", k, args.n, rng)[:2]
        cats["stand"] = rollout_windows(env_id, k, args.n, rng, "stand")
        cats["random"] = rollout_windows(env_id, k, args.n, rng, "random")
        variants = [("unif", None), ("like_1e-3", 1e-3), ("like_1e-2", 1e-2), ("like_1e-1", 1e-1), ("like_0.3", 0.3)]
        res = {"speed_mean": {c: float(v[1].mean()) for c, v in cats.items()}}
        print(f"\n=== {env_id}  mean forward speed: " + ", ".join(f"{c} {v[1].mean():.2f}" for c, v in cats.items()))
        for name, dl in variants:
            mode = "unif" if dl is None else "like"
            S = {}
            for c, (X, V) in cats.items():
                Xt = torch.tensor((X - MU) / SD, dtype=torch.float32, device=dev)
                S[c] = -residual(net, Xt, mode, dl, args.M).cpu().numpy()     # reward = -residual
            m0, s0 = S["expert_ho"].mean(), S["expert_ho"].std() + 1e-9
            z = {c: float(np.median((S[c] - m0) / s0)) for c in S}
            nonexp = np.concatenate([S["simple"], S["medium"]]); vnon = np.concatenate([cats["simple"][1], cats["medium"][1]])
            rho = float(spearmanr(nonexp, vnon).correlation)
            a_me, a_sm, a_ms = auc(S["expert_ho"], S["medium"]), auc(S["medium"], S["simple"]), auc(S["medium"], S["stand"])
            res[name] = dict(median_z=z, spearman_speed_nonexpert=rho, auc_expert_gt_medium=a_me, auc_medium_gt_simple=a_sm,
                             auc_medium_gt_stand=a_ms, auc_simple_gt_stand=auc(S["simple"], S["stand"]))
            print(f" {name:10s} median z: " + " ".join(f"{c} {v:9.1f}" for c, v in z.items()) +
                  f" | rho(speed) {rho:+.2f} | AUC E>M {a_me:.2f} M>S {a_sm:.2f} M>stand {a_ms:.2f} S>stand {res[name]['auc_simple_gt_stand']:.2f}", flush=True)
        od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2"); os.makedirs(od, exist_ok=True)
        json.dump(res, open(os.path.join(od, f"diag_reward_{env_id}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
