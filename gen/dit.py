"""Wan 2.1(2503.20314)식 DiT 축소 구현 — 이미지·영상 공용.

- 잠재 [B,C,T,H,W] (이미지는 T=1) → 3D 합성곱 patchify, 커널 (1,2,2) (Wan).
- 3D RoPE(시간·높이·너비로 헤드 차원 분할), QK-RMSNorm(SD3: bf16 어텐션 로짓 안정화).
- 조건(클래스 토큰 또는 텍스트 임베딩)은 교차 어텐션(Wan).
- 시간 변조: 모든 블록이 공유하는 MLP 하나가 6개 변조값을 내고, 블록마다 학습 바이어스만 더함(Wan, 파라미터 ~25% 절감).
  shared_mod=False 면 블록별 adaLN(비교용, HG4).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def rope_3d(T, H, W, head_dim, device, base=10000.0):
    """헤드 차원을 (t,h,w) 3구간으로 나눈 RoPE 각도. 반환 cos,sin [T*H*W, head_dim/2]."""
    half = head_dim // 2
    dt = half - 2 * (half // 3)
    dh = dw = half // 3
    def ang(n, d):
        inv = 1.0 / (base ** (torch.arange(d, device=device, dtype=torch.float32) / max(d, 1)))
        return torch.arange(n, device=device, dtype=torch.float32)[:, None] * inv[None]
    at, ah, aw = ang(T, dt), ang(H, dh), ang(W, dw)
    a = torch.cat([at[:, None, None, :].expand(T, H, W, dt),
                   ah[None, :, None, :].expand(T, H, W, dh),
                   aw[None, None, :, :].expand(T, H, W, dw)], -1).reshape(T * H * W, half)
    return a.cos(), a.sin()


def apply_rope(x, cos, sin):  # x [B,h,N,d]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


class Attention(nn.Module):
    def __init__(self, d, heads, cross=False, d_ctx=None):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d_ctx or d, 2 * d)
        self.o = nn.Linear(d, d)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.cross = cross

    def forward(self, x, ctx=None, rope=None):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.h, self.hd).transpose(1, 2)
        src = ctx if self.cross else x
        k, v = self.kv(src).view(B, src.shape[1], 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        if rope is not None and not self.cross:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.o(y.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, heads, d_ctx, mlp_ratio=4.0, shared_mod=True):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, heads)
        self.n2 = nn.LayerNorm(d, eps=1e-6)
        self.xattn = Attention(d, heads, cross=True, d_ctx=d_ctx)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(d * mlp_ratio), d))
        self.shared_mod = shared_mod
        if shared_mod:
            self.mod_bias = nn.Parameter(torch.zeros(6, d))          # Wan: 블록별 학습 바이어스만
        else:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))  # 블록별 adaLN
            nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, ctx, temb_mod, temb, rope):
        mod = (temb_mod + self.mod_bias[None]) if self.shared_mod else self.ada(temb).view(-1, 6, x.shape[-1])
        sh1, sc1, g1, sh2, sc2, g2 = mod.unbind(1)
        x = x + g1[:, None] * self.attn(self.n1(x) * (1 + sc1[:, None]) + sh1[:, None], rope=rope)
        x = x + self.xattn(self.n2(x), ctx)
        x = x + g2[:, None] * self.mlp(self.n3(x) * (1 + sc2[:, None]) + sh2[:, None])
        return x


class DiT(nn.Module):
    def __init__(self, in_ch=4, d=384, depth=12, heads=6, d_ctx=384, n_classes=None, patch=(1, 2, 2),
                 mlp_ratio=4.0, shared_mod=True):
        super().__init__()
        self.in_ch, self.patch, self.d, self.heads = in_ch, patch, d, heads
        self.embed = nn.Conv3d(in_ch, d, patch, patch)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.shared_mod = shared_mod
        if shared_mod:
            self.t_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))  # 공유 변조 MLP
            nn.init.zeros_(self.t_mod[-1].weight); nn.init.zeros_(self.t_mod[-1].bias)
        self.n_classes = n_classes
        if n_classes is not None:                                   # 클래스 조건: 마지막 인덱스 = 무조건(CFG)
            self.cls = nn.Embedding(n_classes + 1, d_ctx)
        self.blocks = nn.ModuleList([Block(d, heads, d_ctx, mlp_ratio, shared_mod) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, in_ch * patch[0] * patch[1] * patch[2])
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def null_label(self):
        return self.n_classes

    def forward(self, x, t, cond):
        """x [B,C,T,H,W], t [B] in [0,1], cond = 클래스 인덱스 [B] 또는 텍스트 임베딩 [B,L,d_ctx]."""
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch
        h = self.embed(x)                                           # [B,d,T',H',W']
        Tp, Hp, Wp = h.shape[2:]
        h = h.flatten(2).transpose(1, 2)
        ctx = self.cls(cond)[:, None] if cond.dim() == 1 else cond
        temb = self.t_mlp(timestep_embedding(t, 256))
        temb_mod = self.t_mod(temb).view(B, 6, self.d) if self.shared_mod else None
        rope = rope_3d(Tp, Hp, Wp, self.d // self.heads, x.device)
        for blk in self.blocks:
            h = blk(h, ctx, temb_mod, temb, rope)
        sh, sc = self.final_ada(temb).chunk(2, -1)
        h = self.out(self.final_norm(h) * (1 + sc[:, None]) + sh[:, None])
        h = h.view(B, Tp, Hp, Wp, C, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return h.reshape(B, C, Tp * pt, Hp * ph, Wp * pw)


def num_params(m):
    return sum(p.numel() for p in m.parameters())
