"""Seed-batched networks: S independent copies run as one batched matmul (one kernel per layer)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from common import TimeEmb, LOG_INV_DELTA


class BLinear(nn.Module):
    def __init__(self, S, i, o):
        super().__init__()
        b = 1 / math.sqrt(i)
        self.W = nn.Parameter(torch.empty(S, i, o).uniform_(-b, b))
        self.b = nn.Parameter(torch.empty(S, 1, o).uniform_(-b, b))
    def forward(self, x):            # x (S,B,i)
        return torch.baddbmm(self.b, x, self.W)


def bmlp(S, sizes, act):
    L = []
    for j in range(len(sizes) - 1):
        L.append(BLinear(S, sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            L.append(act())
    return nn.Sequential(*L)


class BActor(nn.Module):
    def __init__(self, S, od, ad, h=256):
        super().__init__()
        self.b = bmlp(S, [od, h, h], nn.ReLU); self.act = nn.ReLU()
        self.mu, self.ls = BLinear(S, h, ad), BLinear(S, h, ad)
    def forward(self, o, det=False):
        z = self.act(self.b(o)); mu = self.mu(z); ls = self.ls(z).clamp(-5, 2)
        if det:
            return torch.tanh(mu), None
        e = mu + ls.exp() * torch.randn_like(mu)
        a = torch.tanh(e)
        lp = (-0.5 * ((e - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        lp = lp - (2 * (math.log(2) - e - F.softplus(-2 * e))).sum(-1)
        return a, lp


class BCritic(nn.Module):
    def __init__(self, S, od, ad, h=256):
        super().__init__()
        self.q1 = bmlp(S, [od + ad, h, h, 1], nn.ReLU); self.q2 = bmlp(S, [od + ad, h, h, 1], nn.ReLU)
    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class BFlow(nn.Module):
    def __init__(self, S, dim, cdim=0, h=256):
        super().__init__()
        self.te = TimeEmb(); self.net = bmlp(S, [dim + 32 + cdim, h, h, h, dim], nn.SiLU)
    def forward(self, x, t, c=None):     # x (S,B,d), t (S,B)
        S, B, _ = x.shape
        z = [x, self.te(t.reshape(-1)).view(S, B, -1)] + ([c] if c is not None else [])
        return self.net(torch.cat(z, -1))


def bsample_t(shape, dev, mode):
    if mode == "unif":
        return torch.rand(shape, device=dev), torch.ones(shape, device=dev)
    s = torch.rand(shape, device=dev) * LOG_INV_DELTA
    t = 1 - torch.exp(-s)
    return t, LOG_INV_DELTA * t


def bfm_loss_per_sample(net, x1, mode, M=4, c=None, shared=False):
    """x1 (S,B,d) -> per-sample weighted residual (S,B).  shared=True: net is an unbatched FlowNet."""
    S, B, d = x1.shape
    xr = x1.repeat(1, M, 1); cr = c.repeat(1, M, 1) if c is not None else None
    t, w = bsample_t((S, B * M), x1.device, mode)
    x0 = torch.randn_like(xr)
    xt = (1 - t[..., None]) * x0 + t[..., None] * xr
    if shared:
        v = net(xt.reshape(S * B * M, d), t.reshape(-1), cr.reshape(S * B * M, -1) if cr is not None else None).view(S, B * M, d)
    else:
        v = net(xt, t, cr)
    r = ((v - (xr - x0)) ** 2).sum(-1) * w
    return r.view(S, M, B).mean(1)


def bfm_train_loss(net, x1, c=None):
    S, B, d = x1.shape; dev = x1.device
    t = torch.where(torch.rand(S, B, device=dev) < 0.5, torch.rand(S, B, device=dev),
                    1 - torch.exp(-torch.rand(S, B, device=dev) * LOG_INV_DELTA))
    x0 = torch.randn_like(x1)
    xt = (1 - t[..., None]) * x0 + t[..., None] * x1
    return ((net(xt, t, c) - (x1 - x0)) ** 2).sum(-1).mean(1).sum()


class BSAC:
    def __init__(self, S, od, ad, dev, lr=3e-4, gamma=0.99, tau=0.005):
        self.S = S
        self.actor = BActor(S, od, ad).to(dev); self.critic = BCritic(S, od, ad).to(dev)
        self.targ = BCritic(S, od, ad).to(dev); self.targ.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(S, 1, device=dev, requires_grad=True)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr)
        self.oc = torch.optim.Adam(self.critic.parameters(), lr)
        self.oal = torch.optim.Adam([self.log_alpha], lr)
        self.gamma, self.tau, self.tgt_ent = gamma, tau, -float(ad)

    def act(self, o, det=False):          # o (S,od)
        with torch.no_grad():
            return self.actor(o[:, None], det)[0][:, 0]

    def update(self, o, a, r, o2, done):  # all (S,B,·)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.actor(o2)
            q1t, q2t = self.targ(o2, a2)
            y = r + self.gamma * (1 - done) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = self.critic(o, a)
        lc = (((q1 - y) ** 2).mean(1) + ((q2 - y) ** 2).mean(1)).sum()
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        an, lp = self.actor(o)
        q1n, q2n = self.critic(o, an)
        la = (alpha * lp - torch.min(q1n, q2n)).mean(1).sum()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        lal = -(self.log_alpha * (lp.detach() + self.tgt_ent)).mean(1).sum()
        self.oal.zero_grad(); lal.backward(); self.oal.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.targ.parameters()):
                pt.lerp_(p, self.tau)
        return alpha


def bfm_loss_mixed(net, x1, like_mask, M=4, c=None, shared=False, log_inv_delta=None):
    """Per-member time weighting: like_mask (S,1) bool -> likelihood weight, else uniform.  x1 (S,B,d) -> (S,B).
    log_inv_delta: (S,1) per-member log(1/delta) for the likelihood weight (default: global LOG_INV_DELTA)."""
    S, B, d = x1.shape; dev = x1.device
    xr = x1.repeat(1, M, 1); cr = c.repeat(1, M, 1) if c is not None else None
    Li = LOG_INV_DELTA if log_inv_delta is None else log_inv_delta
    tu = torch.rand(S, B * M, device=dev)
    s = torch.rand(S, B * M, device=dev) * Li; tl = 1 - torch.exp(-s)
    t = torch.where(like_mask, tl, tu); w = torch.where(like_mask, Li * tl, torch.ones_like(tu))
    x0 = torch.randn_like(xr)
    xt = (1 - t[..., None]) * x0 + t[..., None] * xr
    if shared:
        v = net(xt.reshape(-1, d), t.reshape(-1), cr.reshape(S * B * M, -1) if cr is not None else None).view(S, B * M, d)
    else:
        v = net(xt, t, cr)
    r = ((v - (xr - x0)) ** 2).sum(-1) * w
    return r.view(S, M, B).mean(1)
