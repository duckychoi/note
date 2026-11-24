"""소형 LLM — DeepSeek-V3(2412.19437) / Qwen3(2505.09388) 설계 요소의 축소 구현.

사전정규화 RMSNorm · SwiGLU · RoPE(기저 변경 가능: 문맥 확장) · GQA · QK-RMSNorm(Qwen3).
MoE(선택): 세분화 전문가 top-k + 공유 전문가, 균형 방식 2가지
  - balance="aux_free": 선택에만 쓰는 전문가 편향 b, 스텝마다 b ← b + γ·sign(평균부하 − 부하) (DeepSeek-V3)
  - balance="aux_loss": Switch식 보조손실 α·E·Σ f_i·P_i (비교용)
inputs_embeds 입력 지원(VLM용).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 50304
    d: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    ffn_mult: float = 8 / 3
    rope_base: float = 10000.0
    max_seq: int = 1024
    moe: bool = False
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 1
    expert_ffn: int | None = None          # 전문가 FFN 폭(기본: dense 폭 / top_k)
    first_dense: int = 1                   # 앞 몇 층은 dense (DeepSeek-V3: 3)
    balance: str = "aux_free"
    bias_gamma: float = 1e-3
    aux_alpha: float = 1e-2
    tie_embed: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope_cache(T, hd, base, device):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    a = torch.arange(T, device=device).float()[:, None] * inv[None]
    return a.cos(), a.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.h, self.kvh, self.hd = c.n_head, c.n_kv_head, c.d // c.n_head
        self.q = nn.Linear(c.d, c.d, bias=False)
        self.kv = nn.Linear(c.d, 2 * self.kvh * self.hd, bias=False)
        self.o = nn.Linear(c.d, c.d, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.kvh, self.hd).permute(2, 0, 3, 1, 4)
        k = self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kvh
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, c: GPTConfig, dense_hidden):
        super().__init__()
        self.c = c
        eh = c.expert_ffn or max(32, dense_hidden // c.top_k)
        self.experts = nn.ModuleList([SwiGLU(c.d, eh) for _ in range(c.n_experts)])
        self.shared = nn.ModuleList([SwiGLU(c.d, eh) for _ in range(c.n_shared)])
        self.gate = nn.Linear(c.d, c.n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(c.n_experts))      # aux-free 선택 편향(그래디언트 없음)
        self.last_load = None
        self.aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        h = x.reshape(-1, D)
        s = torch.sigmoid(self.gate(h).float())                       # DeepSeek-V3: 시그모이드 친화도
        sel = s + self.bias if self.c.balance == "aux_free" else s
        idx = sel.topk(self.c.top_k, -1).indices                      # 선택은 편향 포함
        w = s.gather(-1, idx)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)                 # 가중치는 편향 없는 점수로 정규화
        out = torch.zeros_like(h)
        load = torch.zeros(self.c.n_experts, device=x.device)
        for e, ex in enumerate(self.experts):
            mask = idx == e
            rows = mask.any(-1).nonzero(as_tuple=True)[0]
            load[e] = rows.numel()
            if rows.numel():
                out.index_add_(0, rows, ex(h[rows]) * (w * mask)[rows].sum(-1, keepdim=True))
        for sh in self.shared:
            out = out + sh(h)
        self.last_load = load / max(1, h.shape[0])
        if self.c.balance == "aux_loss" and self.training:
            f = load / (h.shape[0] * self.c.top_k)
            P = (s / s.sum(-1, keepdim=True)).mean(0)
            self.aux = self.c.aux_alpha * self.c.n_experts * (f * P).sum()
        else:
            self.aux = x.new_zeros(())
        return out.view(B, T, D)

    @torch.no_grad()
    def update_bias(self):
        """DeepSeek-V3: 과부하 전문가는 편향↓, 저부하는 ↑ (옵티마이저 스텝 후 호출)."""
        if self.c.balance == "aux_free" and self.last_load is not None:
            self.bias += self.c.bias_gamma * torch.sign(self.last_load.mean() - self.last_load)


class Block(nn.Module):
    def __init__(self, c: GPTConfig, use_moe):
        super().__init__()
        hidden = int(c.ffn_mult * c.d / 64) * 64 or 64
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.attn = Attention(c)
        self.ffn = MoE(c, hidden) if use_moe else SwiGLU(c.d, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ffn(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList([Block(c, c.moe and i >= c.first_dense) for i in range(c.n_layer)])
        self.norm = RMSNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        if c.tie_embed:
            self.head.weight = self.emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_rope_base(self, base):
        """문맥 확장(ABF): RoPE 기저를 키운다(Qwen3: 10K→1M)."""
        self.c.rope_base = base

    def forward(self, idx=None, targets=None, inputs_embeds=None, loss_mask=None):
        x = self.emb(idx) if inputs_embeds is None else inputs_embeds
        T = x.shape[1]
        cos, sin = rope_cache(T, self.c.d // self.c.n_head, self.c.rope_base, x.device)
        for b in self.blocks:
            x = b(x, cos.to(x.dtype), sin.to(x.dtype))
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        lf = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is not None:
            m = loss_mask.reshape(-1).float()
            loss = (lf * m).sum() / m.sum().clamp_min(1)
        else:
            loss = lf.mean()
        return logits, loss + self.aux_loss()

    def aux_loss(self):
        return sum((b.ffn.aux for b in self.blocks if isinstance(b.ffn, MoE)), torch.zeros(()))

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def num_params(self, active=False):
        n = sum(p.numel() for p in self.parameters())
        if active and self.c.moe:
            for m in self.moe_layers():
                per = sum(p.numel() for p in m.experts[0].parameters())
                n -= per * (self.c.n_experts - self.c.top_k)
        return n
