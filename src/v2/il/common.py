"""Shared pieces: env/feature spec, expert data, FM flow net, SAC."""
import os, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ.setdefault("MINARI_DATASETS_PATH", "E:/0.Project/fm34_env/minari")

# feature = qpos(no x/y) + qvel part of the observation (drop contact/inertia blocks)
FEAT_DIM = {"Hopper-v5": 11, "Walker2d-v5": 17, "HalfCheetah-v5": 17, "Ant-v5": 27, "Humanoid-v5": 45}
DATASET = {"Hopper-v5": "hopper", "Walker2d-v5": "walker2d", "HalfCheetah-v5": "halfcheetah",
           "Ant-v5": "ant", "Humanoid-v5": "humanoid"}
NO_TERM = {"Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5"}   # have terminate_when_unhealthy


def make_env(env_id, train):
    import gymnasium as gym
    kw = {}
    if train and env_id in NO_TERM:
        kw["terminate_when_unhealthy"] = False
    return gym.make(env_id, **kw)


def load_expert(env_id, n_traj, seed=0, holdout=10):
    """Returns (train windows (N,2k), holdout windows, mean, std) of features."""
    import minari
    ds = minari.load_dataset(f"mujoco/{DATASET[env_id]}/expert-v0")
    k = FEAT_DIM[env_id]
    eps = list(ds.iterate_episodes())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    def win(ids):
        out = []
        for i in ids:
            o = eps[i].observations[:, :k]
            out.append(np.concatenate([o[:-1], o[1:]], 1))
        return np.concatenate(out, 0).astype(np.float32)
    tr, ho = win(idx[:n_traj]), win(idx[n_traj:n_traj + holdout])
    rets = np.array([eps[i].rewards.sum() for i in idx[:n_traj]])
    mu, sd = tr[:, :k].mean(0), tr[:, :k].std(0) + 1e-3
    return tr, ho, mu, sd, float(rets.mean())


# ------------------------------------------------------------------ flow net
class TimeEmb(nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.register_buffer("f", torch.exp(torch.linspace(0, math.log(1000.0), n)))
    def forward(self, t):
        a = t[:, None] * self.f[None]
        return torch.cat([a.sin(), a.cos()], 1)


class FlowNet(nn.Module):
    """v(x_t, t [, c]) for the linear interpolant x_t=(1-t)x0+t x1."""
    def __init__(self, dim, cdim=0, h=256):
        super().__init__()
        self.te = TimeEmb()
        self.net = nn.Sequential(nn.Linear(dim + 32 + cdim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(self, x, t, c=None):
        z = [x, self.te(t)] + ([c] if c is not None else [])
        return self.net(torch.cat(z, 1))


LOG_INV_DELTA = math.log(1e3)       # truncation delta = 1e-3


def sample_t(n, dev, mode):
    if mode == "unif":
        return torch.rand(n, device=dev), torch.ones(n, device=dev)
    # likelihood weight t/(1-t) dt with t = 1-exp(-s), s~U[0,L]:  w dt = t ds  -> MC weight L*t
    s = torch.rand(n, device=dev) * LOG_INV_DELTA
    t = 1 - torch.exp(-s)
    return t, LOG_INV_DELTA * t


def fm_loss_per_sample(net, x1, mode, c=None, M=1):
    """Per-sample weighted FM residual  l(x) = E_t,x0 w(t)||v(x_t,t)-(x1-x0)||^2  (MC with M draws)."""
    n = x1.shape[0]
    xr = x1.repeat(M, 1); cr = c.repeat(M, 1) if c is not None else None
    t, w = sample_t(n * M, x1.device, mode)
    x0 = torch.randn_like(xr)
    xt = (1 - t[:, None]) * x0 + t[:, None] * xr
    r = ((net(xt, t, cr) - (xr - x0)) ** 2).sum(1) * w
    return r.view(M, n).mean(0)


def fm_train_loss(net, x1, c=None):
    """Training loss: mix of uniform and near-data t so both reward weightings are well served."""
    n = x1.shape[0]
    t = torch.where(torch.rand(n, device=x1.device) < 0.5, torch.rand(n, device=x1.device),
                    1 - torch.exp(-torch.rand(n, device=x1.device) * LOG_INV_DELTA))
    x0 = torch.randn_like(x1)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return ((net(xt, t, c) - (x1 - x0)) ** 2).sum(1).mean()


# ------------------------------------------------------------------ SAC
class Actor(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        self.b = nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu, self.ls = nn.Linear(h, ad), nn.Linear(h, ad)
    def forward(self, o, det=False):
        z = self.b(o); mu = self.mu(z); ls = self.ls(z).clamp(-5, 2)
        if det:
            return torch.tanh(mu), None
        e = mu + ls.exp() * torch.randn_like(mu)
        a = torch.tanh(e)
        lp = (-0.5 * ((e - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(1)
        lp = lp - (2 * (math.log(2) - e - F.softplus(-2 * e))).sum(1)
        return a, lp


class Critic(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(od + ad, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.q1, self.q2 = mk(), mk()
    def forward(self, o, a):
        x = torch.cat([o, a], 1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)


class SAC:
    def __init__(self, od, ad, dev, lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = Actor(od, ad).to(dev); self.critic = Critic(od, ad).to(dev)
        self.targ = Critic(od, ad).to(dev); self.targ.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, device=dev, requires_grad=True)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr)
        self.oc = torch.optim.Adam(self.critic.parameters(), lr)
        self.oal = torch.optim.Adam([self.log_alpha], lr)
        self.gamma, self.tau, self.tgt_ent = gamma, tau, -float(ad)

    def act(self, o, det=False):
        with torch.no_grad():
            return self.actor(o[None], det)[0][0].cpu().numpy()

    def update(self, o, a, r, o2, done):
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.actor(o2)
            q1t, q2t = self.targ(o2, a2)
            y = r + self.gamma * (1 - done) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = self.critic(o, a)
        lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        an, lp = self.actor(o)
        q1n, q2n = self.critic(o, an)
        la = (alpha * lp - torch.min(q1n, q2n)).mean()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        lal = -(self.log_alpha * (lp.detach() + self.tgt_ent)).mean()
        self.oal.zero_grad(); lal.backward(); self.oal.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return lc.detach(), alpha


class RunningStat:
    def __init__(self, beta=0.999):
        self.m, self.v, self.b, self.n = 0.0, 1.0, beta, 0
    def update(self, x):
        x = x.detach()
        self.n += 1
        b = min(self.b, 1 - 1 / self.n)
        self.m = b * self.m + (1 - b) * float(x.mean())
        self.v = b * self.v + (1 - b) * float(x.var())
    @property
    def sd(self):
        return math.sqrt(self.v) + 1e-6
