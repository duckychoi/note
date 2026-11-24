"""
#34 pre-registered rest-trap DIAGNOSTIC (11-22, codex review): before any RL, does the training reward give a higher
episode-level (discounted) return to a FEASIBLE standing behaviour than to starting off and walking?

Trajectories (all from env.reset, 300 steps, no RL):
  stand_zero : zero torque (lies down / settles)                                  -- the old, unrealistic reference
  stand_pd   : PD controller holding the reset joint angles (upright balance; collected without any training)
  simple / medium / expert : Minari episodes, first 300 steps (start from the same reset distribution)
Phase for gait trajectories = their own PCA-Hilbert phase (optimistic: as if perfectly synchronised to the clock);
phase for stands = a running clock at the expert period with random offset (the clock keeps running while standing).
Reward = -likelihood-weighted FM residual (delta 1e-3) of the expert prior, per window; the running normalisation used
in training is affine and does not change rankings.  Return G = sum_t 0.99^t r_t.
Rewards (cond): none, film4 (true phase labels), film4shuf (same FiLM network, prior trained on SHUFFLED phase labels).

Diagnostic D_q = fraction of quality-q gait episodes whose return beats the median return of the BEST-scored realistic
stand (max over stand candidates).  Rule (v2, fixed BEFORE the new RL runs): predict 'escapes rest' (final eval forward
speed >= 1.0 m/s after 300k steps with the plain fixed-prior reward) iff D_medium > 0.5.
Calibration: v1 used D_simple; it mispredicted the one known escape (HalfCheetah film4: D_simple 0.00, real vx 2.01),
D_medium matches both known HalfCheetah outcomes (none 0.00 -> stall, film4 0.95 -> escape).  These two are calibration
points, NOT test points.  Walker2d stand: PD could not balance (grid search, height 0.25 m) -> two 300k pilot policies
that stand upright (trained with a different reward) are used as the realistic stand reference.
Old diagnostic for comparison: window AUC simple > stand_zero.
Results -> results/v2/diag_rollout_predict.json  (written BEFORE the new RL runs are launched)
"""
import os, json, math, argparse
import numpy as np, torch
import minari
from common import FEAT_DIM, DATASET, make_env
from phase_util import load_expert_phase, BPhaseFlow, NQ
from diag_reward_landscape import auc

p = argparse.ArgumentParser(); p.add_argument("--envs", default="HalfCheetah-v5,Hopper-v5,Walker2d-v5")
p.add_argument("--conds", default="none,film4,film4shuf"); args = p.parse_args()
dev = torch.device("cuda"); LID = math.log(1e3); H = 300; GAM = 0.99
JOINT = {"HalfCheetah-v5": (2, 8), "Hopper-v5": (2, 5), "Walker2d-v5": (2, 8)}      # obs slice of joint angles
JVEL = {"HalfCheetah-v5": 11, "Hopper-v5": 8, "Walker2d-v5": 11}                    # obs start of joint velocities


def prior(env_id, c, Xtr, Ptr, mu, sd, k):
    path = f"E:/0.Project/fm34_env/cache/{env_id}_n10_phaseflow_{c}.pt"
    net = BPhaseFlow(1, 2 * k, c.replace("shuf", "")).to(dev)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path)); net.eval(); return net
    torch.manual_seed(4321)
    MU = torch.tensor(np.concatenate([mu, mu]), device=dev); SD = torch.tensor(np.concatenate([sd, sd]), device=dev)
    XE = (torch.tensor(Xtr, device=dev) - MU) / SD
    P = np.random.default_rng(0).permutation(Ptr) if c.endswith("shuf") else Ptr
    PE = torch.tensor(P, device=dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30_000)
    for i in range(30_000):
        j = torch.randint(0, XE.shape[0], (1, 1024), device=dev); x1 = XE[j]; ph = PE[j]
        t = torch.where(torch.rand(1, 1024, device=dev) < 0.5, torch.rand(1, 1024, device=dev),
                        1 - torch.exp(-torch.rand(1, 1024, device=dev) * LID))
        x0 = torch.randn_like(x1); xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        l = ((net(xt, t, ph) - (x1 - x0)) ** 2).sum(-1).mean(1).sum(); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    torch.save(net.state_dict(), path); net.eval(); return net


def step_reward(net, X, P, MU, SD, M=16):
    x = torch.tensor((X - MU) / SD, dtype=torch.float32, device=dev)[None].repeat(1, M, 1)
    ph = torch.tensor(P, dtype=torch.float32, device=dev)[None].repeat(1, M)
    s = torch.rand(x.shape[:2], device=dev) * LID; t = 1 - torch.exp(-s); x0 = torch.randn_like(x)
    xt = (1 - t[..., None]) * x0 + t[..., None] * x
    with torch.no_grad():
        r = ((net(xt, t, ph) - (x - x0)) ** 2).sum(-1) * (LID * t)
    return -r.view(M, -1).mean(0).cpu().numpy()


PD = {"HalfCheetah-v5": (5, 0.5, 0, []), "Hopper-v5": (60, 2, 2, [0]), "Walker2d-v5": (5, 0.5, 6, [0, 3])}   # grid-searched, no RL


def stand_actor(member):
    """Walker2d only: PD cannot balance, so the upright reference is a 300k pilot policy that learned to stand
    (trained with the *_like_state* reward, a different reward from the plain rewards predicted here)."""
    import torch.nn as nn
    st = torch.load("E:/0.Project/fm34_env/phase2/Walker2d-v5_pilot300_actors.pt")["actor"]
    net = nn.Sequential(nn.Linear(19, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6), nn.Tanh())
    for layer, key in ((0, "b.0"), (2, "b.2"), (4, "mu")):
        net[layer].load_state_dict(dict(weight=st[key + ".W"][member].T.cpu(), bias=st[key + ".b"][member, 0].cpu()))
    return net.eval()


def stand_trajs(env_id, kind, n, T, rng):
    env = make_env(env_id, True); out = []; a0, a1 = JOINT[env_id]; jv = JVEL[env_id]; kp, kd, kt, hips = PD[env_id]
    pol = stand_actor({"policy0": 0, "policy4": 4}[kind]) if kind.startswith("policy") else None
    for i in range(n):
        o = env.reset(seed=50_000 + i)[0]; q0 = o[a0:a1].copy(); O = [o]
        ph = 2 * np.pi * rng.random() + 2 * np.pi / T * np.arange(H + 1)
        for st in range(H):
            if kind == "zero":
                a = np.zeros(env.action_space.shape)
            elif pol is not None:
                with torch.no_grad():
                    a = pol(torch.tensor(np.concatenate([o, [np.cos(ph[st]), np.sin(ph[st])]]), dtype=torch.float32)).numpy()
            else:
                a = kp * (q0 - o[a0:a1]) - kd * o[jv:jv + (a1 - a0)]
                for hp in hips:
                    a[hp] += kt * o[1] + 0.5 * kt * o[jv - 1]
                a = np.clip(a, -1, 1)
            o = env.step(a)[0]; O.append(o)
        O = np.array(O); out.append((O, np.angle(np.exp(1j * ph[:len(O)]))))
    return out


def gait_trajs(env_id, q, n, ref):
    eps = list(minari.load_dataset(f"mujoco/{DATASET[env_id]}/{q}-v0").iterate_episodes()); out = []
    for i in np.random.default_rng(1).permutation(len(eps)):
        O = eps[i].observations
        if len(O) > H:
            out.append((O[:H + 1], ref.phase(O)[:H + 1]))
        if len(out) == n:
            break
    return out


def main():
    res = {}
    for env_id in args.envs.split(","):
        k = FEAT_DIM[env_id]; rng = np.random.default_rng(0)
        Xtr, Ptr, _, _, mu, sd, EXP, ref, T = load_expert_phase(env_id, 10)
        MU = np.concatenate([mu, mu]); SD = np.concatenate([sd, sd])
        G = {"stand_zero": stand_trajs(env_id, "zero", 20, T, rng), "stand_pd": stand_trajs(env_id, "pd", 20, T, rng),
             **({"stand_pol0": stand_trajs(env_id, "policy0", 20, T, rng), "stand_pol4": stand_trajs(env_id, "policy4", 20, T, rng)} if env_id == "Walker2d-v5" else {}),
             "simple": gait_trajs(env_id, "simple", 20, ref), "medium": gait_trajs(env_id, "medium", 20, ref),
             "expert": gait_trajs(env_id, "expert", 20, ref)}
        info = {g: dict(height=float(np.median([O[:, 0].mean() for O, _ in L])), vx=float(np.mean([O[:, NQ[env_id]].mean() for O, _ in L]))) for g, L in G.items()}
        res[env_id] = dict(info=info)
        print(f"\n=== {env_id}  period {T:.1f}  " + " ".join(f"{g}: h {v['height']:.2f} vx {v['vx']:.2f}" for g, v in info.items()), flush=True)
        for c in args.conds.split(","):
            net = prior(env_id, c, Xtr, Ptr, mu, sd, k)
            R = {}
            for g, L in G.items():
                R[g] = np.array([np.sum(GAM ** np.arange(H) * step_reward(net, np.concatenate([O[:-1, :k], O[1:, :k]], 1), Ph[:-1], MU, SD)) for O, Ph in L])
            best_stand = max([g for g in R if g.startswith("stand")], key=lambda g: np.median(R[g]))
            D = float((R["simple"] > np.median(R[best_stand])).mean())
            Dm = float((R["medium"] > np.median(R[best_stand])).mean())
            # old diagnostic: window-level AUC simple > zero-torque rest
            ws = np.concatenate([step_reward(net, np.concatenate([O[:-1, :k], O[1:, :k]], 1), Ph[:-1], MU, SD) for O, Ph in G["simple"][:8]])
            wz = np.concatenate([step_reward(net, np.concatenate([O[50:-1, :k], O[51:, :k]], 1), Ph[50:-1], MU, SD) for O, Ph in G["stand_zero"][:8]])
            old = auc(ws, wz)
            pred = "escape" if Dm > 0.5 else "stall"     # rule fixed on the two known HalfCheetah outcomes (see header)
            res[env_id][c] = dict(D_simple=D, D_medium=Dm, best_stand=best_stand, old_window_auc=old, prediction=pred,
                                  old_prediction=("escape" if old > 0.5 else "stall"),
                                  median_return={g: float(np.median(v)) for g, v in R.items()})
            print(f" {c:10s} median G: " + " ".join(f"{g} {np.median(v):9.0f}" for g, v in R.items()) +
                  f" | best stand {best_stand} | D_simple {D:.2f} D_medium {Dm:.2f} -> {pred} | old window AUC {old:.2f}", flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(res, open(os.path.join(od, "diag_rollout_predict.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
