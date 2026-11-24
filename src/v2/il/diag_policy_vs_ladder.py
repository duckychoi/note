"""Diagnosis 3 (advisor check 2): where do the FAILED policies' own windows sit on the reward ladder?
Scores final-eval windows saved by train_multi (ablate/*_feats.npz, raw features (s,s')) with the expert term of the
exact training reward (bfm_loss_mixed, likelihood weight, delta 1e-3, same cached expert flow), next to
expert-holdout / medium / simple / stand / random. Robust scale: median and IQR of expert-holdout.
Reads: policy above simple -> wrong optimum (trap); policy far below simple -> RL not climbing."""
import glob, os, numpy as np, torch
from common import FEAT_DIM, FlowNet, load_expert
from vec import bfm_loss_mixed
from diag_reward_landscape import windows_from_dataset, rollout_windows, auc, NQ
env_id = "HalfCheetah-v5"; k = FEAT_DIM[env_id]; dev = torch.device("cuda"); torch.manual_seed(0)
_, _, mu, sd, _ = load_expert(env_id, 10, seed=0); MU = np.concatenate([mu, mu]); SD = np.concatenate([sd, sd])
net = FlowNet(2 * k).to(dev); net.load_state_dict(torch.load(f"E:/0.Project/fm34_env/cache/{env_id}_n10_flow.pt")); net.eval()
import minari
from common import DATASET
eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/expert-v0").iterate_episodes())
tr = set(np.random.default_rng(0).permutation(len(eps))[:10].tolist()); rng = np.random.default_rng(1)
cats = {"expert_ho": windows_from_dataset(env_id, "expert", k, 3000, rng, tr)[:2],
        "medium": windows_from_dataset(env_id, "medium", k, 3000, rng)[:2],
        "simple": windows_from_dataset(env_id, "simple", k, 3000, rng)[:2],
        "stand": rollout_windows(env_id, k, 3000, rng, "stand")}
for f in sorted(glob.glob("E:/0.Project/fm34_env/ablate/HalfCheetah-v5_*_feats.npz")):
    F = np.load(f)["F"]; j = rng.choice(len(F), 3000, replace=False)
    cats["pol:" + os.path.basename(f)[15:-10]] = (F[j], F[j, NQ[env_id]])
like = torch.ones(1, 1, dtype=torch.bool, device=dev); L = torch.tensor([[float(np.log(1e3))]], dtype=torch.float32, device=dev)
S = {}
for c, (X, V) in cats.items():
    x = torch.tensor((X - MU) / SD, dtype=torch.float32, device=dev)[None]
    with torch.no_grad():
        S[c] = -torch.cat([bfm_loss_mixed(net, x[:, i:i+1000], like, 32, shared=True, log_inv_delta=L) for i in range(0, x.shape[1], 1000)], 1)[0].cpu().numpy()
e = S["expert_ho"]; m0 = np.median(e); iqr = np.subtract(*np.percentile(e, [75, 25])) + 1e-9
print(f"{'category':32s} {'speed':>6s} {'med z(IQR)':>11s} {'P>stand':>8s} {'P>simple':>9s}")
for c in S:
    print(f"{c:32s} {cats[c][1].mean():6.2f} {np.median((S[c]-m0)/iqr):11.1f} {auc(S[c], S['stand']):8.2f} {auc(S[c], S['simple']):9.2f}")
