"""Diagnosis 4: is the rest trap a property of the expert data, or an artefact of the FM density estimator?
Compare, per category, the distance to the nearest expert-TRAIN window (normalised features) with the FM reward.
If 'stand'/FM-policy windows are FAR from every expert window yet the flow rates them near-expert -> estimator
smoothing (ring-centre filling). Also a kNN log-density proxy -d*log(r_k) ranks categories without the flow."""
import glob, os, numpy as np, torch
from common import FEAT_DIM, load_expert
from diag_reward_landscape import windows_from_dataset, rollout_windows, auc, NQ
import minari
from common import DATASET
dev = torch.device("cuda")
for env_id in ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]:
    k = FEAT_DIM[env_id]; X_tr, _, mu, sd, _ = load_expert(env_id, 10, seed=0)
    MU = np.concatenate([mu, mu]); SD = np.concatenate([sd, sd]); E = torch.tensor((X_tr - MU) / SD, dtype=torch.float32, device=dev)
    eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/expert-v0").iterate_episodes())
    tr = set(np.random.default_rng(0).permutation(len(eps))[:10].tolist()); rng = np.random.default_rng(1)
    cats = {"expert_ho": windows_from_dataset(env_id, "expert", k, 2000, rng, tr)[:2],
            "medium": windows_from_dataset(env_id, "medium", k, 2000, rng)[:2],
            "simple": windows_from_dataset(env_id, "simple", k, 2000, rng)[:2],
            "stand": rollout_windows(env_id, k, 2000, rng, "stand")}
    if env_id == "HalfCheetah-v5":
        for tag in ["fm_like_state_s0", "amp_s0", "sac_true_s0"]:
            F = np.load(f"E:/0.Project/fm34_env/ablate/HalfCheetah-v5_{tag}_feats.npz")["F"]; j = rng.choice(len(F), 2000, replace=False)
            cats["pol:" + tag] = (F[j], F[j, NQ[env_id]])
    print(f"\n=== {env_id} (dim {2*k})   NN dist to expert-train windows (median) | kNN(k=5) log-density proxy: P>stand  P>simple")
    D = {}
    for c, (X, V) in cats.items():
        x = torch.tensor((X - MU) / SD, dtype=torch.float32, device=dev)
        d = torch.cat([torch.cdist(x[i:i+500], E).topk(5, largest=False).values for i in range(0, len(x), 500)])
        D[c] = (d[:, 0].cpu().numpy(), -(2 * k) * np.log(d[:, 4].cpu().numpy() + 1e-9))
    for c in D:
        print(f" {c:22s} speed {cats[c][1].mean():6.2f} | NN {np.median(D[c][0]):7.2f} | P>stand {auc(D[c][1], D['stand'][1]):.2f} P>simple {auc(D[c][1], D['simple'][1]):.2f}")
