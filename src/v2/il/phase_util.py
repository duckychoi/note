"""Expert gait phase labels (PCA-1 of joint positions + Hilbert) and phase-conditioned flow nets."""
import math, os
import numpy as np, torch, torch.nn as nn
from scipy.signal import hilbert
from common import FEAT_DIM, DATASET, TimeEmb
from vec import BLinear

os.environ.setdefault("MINARI_DATASETS_PATH", "E:/0.Project/fm34_env/minari")
NQ = {"Hopper-v5": 5, "Walker2d-v5": 8, "HalfCheetah-v5": 8, "Ant-v5": 13, "Humanoid-v5": 22}   # joint-position slice end


class PhaseRef:
    """Fixed PCA direction fitted on expert joint positions; phase(traj) = angle(hilbert(PC1)) (offline, non-causal)."""
    def __init__(self, env_id, joints):
        self.env_id = env_id; self.lo, self.hi = 1, NQ[env_id]
        J = joints - joints.mean(0)
        self.mean = joints.mean(0)
        self.dir = np.linalg.svd(J, full_matrices=False)[2][0]

    def phase(self, obs_traj):
        z = (obs_traj[:, self.lo:self.hi] - self.mean) @ self.dir
        return np.angle(hilbert(z - z.mean()))


def load_expert_phase(env_id, n_traj, seed=0, holdout=10):
    import minari
    ds = minari.load_dataset(f"mujoco/{DATASET[env_id]}/expert-v0")
    k = FEAT_DIM[env_id]
    eps = [e for e in ds.iterate_episodes() if len(e.rewards) >= 999]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(eps))
    tr_ids, ho_ids = idx[:n_traj], idx[n_traj:n_traj + holdout]
    ref = PhaseRef(env_id, np.concatenate([eps[i].observations[:, 1:NQ[env_id]] for i in tr_ids]))
    def win(ids):
        X, P = [], []
        for i in ids:
            o = eps[i].observations[:, :k]; ph = ref.phase(eps[i].observations)
            X.append(np.concatenate([o[:-1], o[1:]], 1)); P.append(ph[:-1])
        return np.concatenate(X).astype(np.float32), np.concatenate(P).astype(np.float32)
    Xtr, Ptr = win(tr_ids); Xho, Pho = win(ho_ids)
    # gait period from phase velocity (robust)
    dph = []
    for i in tr_ids:
        ph = np.unwrap(ref.phase(eps[i].observations)); dph.append((ph[-50] - ph[50]) / (len(ph) - 100))
    period = 2 * math.pi / float(np.mean(dph))
    rets = float(np.mean([eps[i].rewards.sum() for i in tr_ids]))
    mu, sd = Xtr[:, :k].mean(0), Xtr[:, :k].std(0) + 1e-3
    vx = float(np.mean([eps[i].observations[:, NQ[env_id]].mean() for i in tr_ids]))   # obs[NQ] = qvel[0] = forward speed
    from common import make_env
    env = make_env(env_id, False); env.reset(seed=0); ft = FootTD(env, env_id)
    tdph = [[] for _ in ft.gi]
    for i in tr_ids:
        O = eps[i].observations; ph = ref.phase(O)
        for j, ev in enumerate(ft.events(ft.z_from_obs(env.unwrapped.data, O))):
            tdph[j] += list(ph[ev])
    td_mean = np.array([np.angle(np.exp(1j * np.array(a)).mean()) for a in tdph])
    td_R = np.array([abs(np.exp(1j * np.array(a)).mean()) for a in tdph])
    env.close()
    err = np.concatenate([np.abs(np.angle(np.exp(1j * (np.array(a) - mm)))) for a, mm in zip(tdph, td_mean)])
    tol = float(np.quantile(err, 0.9))     # 90% of expert touchdowns lie within tol of the expert touchdown phase
    expert = dict(ret=rets, vx=vx, td_phase=td_mean, td_R=td_R, n_td=[len(a) for a in tdph], td_tol=tol)
    return Xtr, Ptr, Xho, Pho, mu, sd, expert, ref, period


FREQS = torch.tensor([1.0, 2.0, 3.0, 4.0])


def phase_feats(phi, K=4):
    """multi-harmonic sinusoidal features of phase (S,B) -> (S,B,2K)"""
    a = phi[..., None] * FREQS[:K].to(phi.device)
    return torch.cat([a.cos(), a.sin()], -1)


class BPhaseFlow(nn.Module):
    """Seed-batched FM net conditioned on phase.
       cond='none'    : ignores phase
       cond='concat'  : input concat [cos phi, sin phi] (1 harmonic, v1 control)
       cond='concat4' : input concat of 4 harmonics [cos k phi, sin k phi]_{k=1..4}  (same information as rope)
       cond='film4'   : FiLM on first hidden layer from the same 4 harmonics (h <- h*(1+gamma)+beta)
       cond='rope'    : RoPE on hidden units: pairs of first-layer features rotated by angle w_j*phi, w_j in {1,2,3,4}
    """
    def __init__(self, S, dim, cond, h=256):
        super().__init__()
        self.cond = cond; self.te = TimeEmb()
        extra = {"concat": 2, "concat4": 8}.get(cond, 0)
        self.l1 = BLinear(S, dim + 32 + extra, h); self.l2 = BLinear(S, h, h); self.l3 = BLinear(S, h, h); self.l4 = BLinear(S, h, dim)
        if cond == "film4":
            self.film = BLinear(S, 8, 2 * h)
        self.act = nn.SiLU()
        self.register_buffer("w", FREQS.repeat(h // 2 // len(FREQS) + 1)[: h // 2])

    def forward(self, x, t, phi):           # x (S,B,d), t (S,B), phi (S,B)
        S, B, _ = x.shape
        z = [x, self.te(t.reshape(-1)).view(S, B, -1)]
        if self.cond == "concat":
            z.append(torch.stack([phi.cos(), phi.sin()], -1))
        elif self.cond == "concat4":
            z.append(phase_feats(phi, 4))
        h = self.l1(torch.cat(z, -1))
        if self.cond == "rope":
            a = phi[..., None] * self.w                    # (S,B,h/2)
            c, s = a.cos(), a.sin(); h1, h2 = h[..., 0::2], h[..., 1::2]
            h = torch.stack([h1 * c - h2 * s, h1 * s + h2 * c], -1).flatten(-2)
        elif self.cond == "film4":
            gb = self.film(phase_feats(phi, 4)); g_, b_ = gb.chunk(2, -1)
            h = h * (1 + g_) + b_
        h = self.act(h); h = self.act(self.l2(h)); h = self.act(self.l3(h))
        return self.l4(h)


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


FOOT_GEOMS = {"Walker2d-v5": ["foot_geom", "foot_left_geom"], "Hopper-v5": ["foot_geom"], "HalfCheetah-v5": ["bfoot", "ffoot"]}


class FootTD:
    """Independent phase reference: foot touchdown events (hysteresis on capsule-bottom height).
       Expert touchdown phase per foot = circular mean of the expert clock label (Hilbert PC1) at expert touchdowns.
       A learner locked to the clock touches down at clock phase == expert touchdown phase."""
    def __init__(self, env, env_id, lo=0.02, hi=0.06):
        import mujoco
        self.m = env.unwrapped.model; self.lo, self.hi = lo, hi; self.env_id = env_id
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.m.ngeom)]
        self.gi = [names.index(n) for n in FOOT_GEOMS[env_id]]
        self.r = np.array([self.m.geom_size[g][0] for g in self.gi])

    def z_live(self, data):
        return np.array([data.geom_xpos[g][2] for g in self.gi]) - self.r

    def z_from_obs(self, data, O):
        import mujoco
        nq, nv = NQ[self.env_id], self.m.nv; Z = []
        for o in O:
            data.qpos[:] = np.concatenate([[0.0], o[:nq]]); data.qvel[:] = o[nq:nq + nv]; mujoco.mj_forward(self.m, data)
            Z.append(self.z_live(data))
        return np.array(Z)

    def events(self, Z):
        """Z (T, nfeet) -> list per foot of touchdown step indices (enter < lo after having been > hi)."""
        out = []
        for j in range(Z.shape[1]):
            z, up, td = Z[:, j], True, []
            for t in range(len(z)):
                if up and z[t] < self.lo:
                    td.append(t); up = False
                elif not up and z[t] > self.hi:
                    up = True
            out.append(np.array(td, int))
        return out


def plv(phase_gait, phase_clock, burn=100):
    d = phase_gait[burn:] - phase_clock[burn:]
    zc = np.exp(1j * d).mean()
    return float(abs(zc)), float(np.angle(zc))
