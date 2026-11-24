"""Walker2d: the 300k pilot policies stand upright and still (vx~0, no falls). The earlier static test used zero-action
'rest' which collapses to the floor. Here: synthetic UPRIGHT stand windows (reset pose + small noise, velocities 0, s'=s)
vs simple/medium gaits, scored by each phase-conditioned expert prior (phase uniform for the stand)."""
import numpy as np, torch, math
from common import FEAT_DIM, make_env
from phase_util import BPhaseFlow
import diag_reward_validity as V        # reuses its globals (Walker2d by default)
env = make_env("Walker2d-v5", True); X = []
rng = np.random.default_rng(3)
for i in range(3000):
    o = env.reset(seed=i)[0][:V.k].copy(); o[V.k//2 - 0:] = 0.0 if False else o[V.k//2:]
    o[8:] = 0.0                                          # all velocities zero (qvel block starts at obs[8])
    X.append(np.concatenate([o, o]))
X = np.array(X); P = rng.uniform(-np.pi, np.pi, len(X))
for c in ["none", "concat4", "film4", "rope"]:
    net = BPhaseFlow(1, 2 * V.k, c).to(V.dev); net.load_state_dict(torch.load(f"E:/0.Project/fm34_env/cache/Walker2d-v5_n10_phaseflow_{c}.pt")); net.eval()
    s_up = V.score(net, X, P)
    print(c, " ".join(f"{g}>upright {V.auc(V.score(net, *V.GA[g]), s_up):.2f}" for g in V.GA), "| expert>upright", f"{V.auc(V.score(net, *V.REF), s_up):.2f}")
