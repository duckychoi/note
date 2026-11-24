"""Rectified flow — SD3(2403.03206) · Wan(2503.20314) 규약.

x_t = (1 − t)·x₀ + t·ε   (t=0 데이터, t=1 잡음),  목표 속도 v = ε − x₀,  손실 MSE.
t ~ logit-normal(0,1) (SD3 최선) 또는 균등. 해상도 시프트: t' = α·t / (1 + (α − 1)·t), α = √(m/n)
(m = 현재 토큰 수, n = 기준 토큰 수; α>1 이면 잡음 쪽으로 이동 — 고해상도일수록 더 많은 잡음).
CFG: 학습 시 조건을 p_drop 확률로 무조건 토큰으로 바꾸고, 샘플링 때 v = v_u + w·(v_c − v_u).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def shift_t(t, alpha):
    return alpha * t / (1 + (alpha - 1) * t) if alpha != 1.0 else t


def resolution_alpha(num_tokens, base_tokens):
    """SD3: α = √(m/n)."""
    return math.sqrt(num_tokens / base_tokens)


def sample_t(B, device, mode="logit_normal", m=0.0, s=1.0):
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(B, device=device) * s + m)
    return torch.rand(B, device=device)


def flow_loss(model, x0, cond, alpha=1.0, t_mode="logit_normal", p_drop=0.1, null_cond=None):
    B = x0.shape[0]
    t = shift_t(sample_t(B, x0.device, t_mode), alpha)
    eps = torch.randn_like(x0)
    tt = t.view(B, *([1] * (x0.dim() - 1)))
    xt = (1 - tt) * x0 + tt * eps
    if p_drop > 0 and null_cond is not None:
        drop = torch.rand(B, device=x0.device) < p_drop
        cond = torch.where(drop, torch.full_like(cond, null_cond), cond) if cond.dim() == 1 else \
            torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
    v = model(xt, t, cond)
    return F.mse_loss(v.float(), (eps - x0).float())


@torch.no_grad()
def sample(model, shape, cond, steps=50, alpha=1.0, cfg_scale=1.0, null_cond=None, device="cpu", generator=None):
    """오일러 적분 t: 1 → 0 (시프트된 격자)."""
    x = torch.randn(shape, device=device, generator=generator)
    ts = shift_t(torch.linspace(1, 0, steps + 1, device=device), alpha)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        tb = torch.full((shape[0],), float(t), device=device)
        v = model(x, tb, cond)
        if cfg_scale != 1.0 and null_cond is not None:
            vu = model(x, tb, torch.full_like(cond, null_cond) if cond.dim() == 1 else torch.zeros_like(cond))
            v = vu + cfg_scale * (v - vu)
        x = x + (t_next - t) * v
    return x
