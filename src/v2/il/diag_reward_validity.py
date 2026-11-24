"""
#34 reward VALIDITY battery (no RL) - the original question: "does a motion that gets a higher reward really look
more natural and on-beat?"  (11-22, after the codex discussion: verify the reward reads each property before RL.)

Reward r(x, phi) = -likelihood-weighted FM residual (delta 1e-3) of the cached expert prior for each conditioning
(none / concat4 / film4 / rope).  Reference = held-out expert windows with their own (PCA-Hilbert) phase.
  T1 order       : time-reversed windows (s', s) and skipped windows (s_t, s'_{t+5})  -> must score LOWER
  T2 wrong beat  : same expert windows, phase label shifted by pi/4, pi/2, pi       -> phase-aware reward must score LOWER
                   (phase-free 'none' is 0.5 by construction = cannot read the beat)
  T3 start shift : style is phase-invariant when the beat command matches the motion: score of expert windows with
                   their own phase vs the same windows from a trajectory started half a cycle later (own phase again)
                   -> trivially equal for 2-frame windows; reported as AUC ~0.5 (sanity)
  T4 rest trap   : medium / simple gaits (own phase) vs genuine rest (phase uniform: the clock keeps running)
                   -> must score HIGHER
Metric: AUC = P(reference scores higher than variant).  Results -> results/v2/diag_validity_<env>.json
"""
import os, json, math, argparse
import numpy as np, torch
import minari
from common import FEAT_DIM, DATASET, make_env
from phase_util import load_expert_phase, BPhaseFlow
from diag_reward_landscape import auc

p = argparse.ArgumentParser(); p.add_argument("--env", default="Walker2d-v5"); p.add_argument("--conds", default="none,concat4,film4,rope")
args = p.parse_args(); dev = torch.device("cuda"); torch.manual_seed(0); rng = np.random.default_rng(0)
env_id = args.env; k = FEAT_DIM[env_id]
Xtr, Ptr, Xho, Pho, mu, sd, EXP, ref, T = load_expert_phase(env_id, 10)
MU = np.concatenate([mu, mu]); SD = np.concatenate([sd, sd]); LID = math.log(1e3)


def score(net, X, P, M=16):
    x = torch.tensor((X - MU) / SD, dtype=torch.float32, device=dev); ph = torch.tensor(P, dtype=torch.float32, device=dev)
    out = []
    for i in range(0, len(x), 1000):
        x1 = x[i:i + 1000][None].repeat(1, M, 1); p1 = ph[i:i + 1000][None].repeat(1, M); B = x1.shape[1]
        s = torch.rand(1, B, device=dev) * LID; t = 1 - torch.exp(-s); w = LID * t; x0 = torch.randn_like(x1)
        xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        with torch.no_grad():
            r = ((net(xt, t, p1) - (x1 - x0)) ** 2).sum(-1) * w
        out.append(-r.view(M, -1).mean(0))
    return torch.cat(out).cpu().numpy()


def gait_windows(q, n):
    eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/{q}-v0").iterate_episodes()); X, P = [], []
    for i in rng.permutation(len(eps)):
        O = eps[i].observations
        if len(O) < 60:
            continue
        ph = ref.phase(O); X.append(np.concatenate([O[:-1, :k], O[1:, :k]], 1)); P.append(ph[:-1])
        if sum(len(x) for x in X) > 3 * n:
            break
    X, P = np.concatenate(X), np.concatenate(P); j = rng.choice(len(X), n, replace=False); return X[j], P[j]


def rest_windows(n):
    env = make_env(env_id, True); X = []
    while sum(len(x) for x in X) < n:
        o = env.reset(seed=int(rng.integers(1e6)))[0]; O = [o]
        for _ in range(250):
            O.append(env.step(np.zeros(env.action_space.shape))[0])
        O = np.array(O)[50:]; X.append(np.concatenate([O[:-1, :k], O[1:, :k]], 1))
    X = np.concatenate(X)[:n]; return X, rng.uniform(-np.pi, np.pi, n)


HALF = max(1, int(round(T / 2)))          # half a gait cycle (v1 used 3 frames: not a half cycle for Walker, T=114.5)
n = 3000; j = rng.choice(len(Xho) - HALF - 6, n, replace=False)
REF = (Xho[j], Pho[j])
VAR = {"T1 reversed": (np.concatenate([Xho[j, k:], Xho[j, :k]], 1), Pho[j]),
       "T1 skip5": (np.concatenate([Xho[j, :k], Xho[j + 5, k:]], 1), Pho[j])}
for dlt, lab in [(math.pi / 4, "pi/4"), (math.pi / 2, "pi/2"), (math.pi, "pi")]:
    VAR[f"T2 beat+{lab}"] = (Xho[j], np.angle(np.exp(1j * (Pho[j] + dlt))))
VAR["T3 start+half cycle"] = (Xho[j + HALF], Pho[j + HALF])
GA = {"medium": gait_windows("medium", n), "simple": gait_windows("simple", n)}; REST = rest_windows(n)
res = {}
print(f"=== {env_id}: AUC = P(reference > variant); want high except T3 (~0.5)")
for c in args.conds.split(","):
    path = f"E:/0.Project/fm34_env/cache/{env_id}_n10_phaseflow_{c}.pt"
    if not os.path.exists(path):
        print(" skip", c, "(no cache)"); continue
    net = BPhaseFlow(1, 2 * k, c).to(dev); net.load_state_dict(torch.load(path)); net.eval()
    s_ref = score(net, *REF); r = {}
    for name, (X, P) in VAR.items():
        r[name] = auc(s_ref, score(net, X, P))
    s_rest = score(net, *REST)
    for g, (X, P) in GA.items():
        r[f"T4 {g}>rest"] = auc(score(net, X, P), s_rest)
    res[c] = r
    print(f" {c:8s} " + " | ".join(f"{kk} {v:.2f}" for kk, v in r.items()), flush=True)
od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
json.dump(res, open(os.path.join(od, f"diag_validity_{env_id}.json"), "w"), indent=1)
