"""Diagnosis 5: does phase conditioning remove the rest trap?  (kNN, no flow, so it tests the target geometry.)
Phase of every episode = Hilbert phase of the expert's highest-variance joint signal (band-passed by mean removal),
computed identically for expert / medium / simple; 'stand' has no oscillation -> phase uniform at random
(the policy's clock keeps running while it stands). Distance in [window features, c*(cos phi, sin phi)].
Report P(category > stand) and P(category > simple) of the kNN log-density proxy for c in {0,1,2,4}."""
import numpy as np, torch
from scipy.signal import hilbert
import minari
from common import FEAT_DIM, DATASET, load_expert, make_env
from diag_reward_landscape import auc, NQ
dev = torch.device("cuda")


def seqs(env_id, q, n_steps, rng, skip=None):
    eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/{q}-v0").iterate_episodes()); out = []; tot = 0
    for i in rng.permutation(len(eps)):
        if skip is not None and i in skip:
            continue
        o = eps[i].observations
        if len(o) > 40:
            out.append(o); tot += len(o)
        if tot > n_steps:
            break
    return out


def stand_seqs(env_id, n_steps, rng):
    env = make_env(env_id, False); out = []; tot = 0
    while tot < n_steps:
        o = env.reset(seed=int(rng.integers(1e6)))[0]; obs = [o]
        for t in range(300):
            o, *_ = env.step(np.zeros(env.action_space.shape)); obs.append(o)
        out.append(np.array(obs)[50:]); tot += len(out[-1])      # drop the reset transient: genuine rest
    return out


def windows(S, k, j, rng, random_phase=False):
    X, P = [], []
    for O in S:
        if random_phase:
            ph = rng.uniform(-np.pi, np.pi, len(O))
        else:
            sig = O[:, j] - O[:, j].mean(); ph = np.angle(hilbert(sig))
        X.append(np.concatenate([O[:-1, :k], O[1:, :k]], 1)); P.append(ph[1:])
    return np.concatenate(X), np.concatenate(P)


for env_id in ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]:
    k = FEAT_DIM[env_id]; X_tr, _, mu, sd, _ = load_expert(env_id, 10, seed=0); MU = np.concatenate([mu, mu]); SD = np.concatenate([sd, sd])
    rng = np.random.default_rng(0)
    eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/expert-v0").iterate_episodes())
    perm = np.random.default_rng(0).permutation(len(eps)); trn = [eps[i].observations for i in perm[:10]]
    j = int(np.argmax(np.concatenate(trn)[:, 1:k].std(0) / sd[1:])) + 1 if False else int(np.argmax(np.concatenate(trn)[:, 2:k - (k // 2)].std(0))) + 2
    C = {"expert_ho": seqs(env_id, "expert", 6000, rng, set(perm[:10].tolist())), "medium": seqs(env_id, "medium", 6000, rng),
         "simple": seqs(env_id, "simple", 6000, rng), "stand": stand_seqs(env_id, 6000, rng)}
    XE, PE = windows(trn, k, j, rng)
    W = {c: windows(s, k, j, rng, random_phase=(c == "stand")) for c, s in C.items()}
    print(f"\n=== {env_id}  phase joint = obs[{j}]")
    for cc in [0.0, 1.0, 2.0, 4.0]:
        def emb(X, P):
            return torch.tensor(np.concatenate([(X - MU) / SD, cc * np.cos(P)[:, None], cc * np.sin(P)[:, None]], 1), dtype=torch.float32, device=dev)
        E = emb(XE, PE); S = {}
        for c, (X, P) in W.items():
            idx = rng.choice(len(X), min(2000, len(X)), replace=False); x = emb(X[idx], P[idx])
            d = torch.cat([torch.cdist(x[i:i+500], E).topk(5, largest=False).values[:, 4] for i in range(0, len(x), 500)]).cpu().numpy()
            S[c] = -(2 * k) * np.log(d + 1e-9)
        print(f" c={cc:3.1f} | " + " ".join(f"{c} P>stand {auc(S[c], S['stand']):.2f}" for c in ["expert_ho", "medium", "simple"]) +
              f" | stand P>simple {auc(S['stand'], S['simple']):.2f}")
