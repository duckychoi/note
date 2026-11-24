"""후학습 축소판: 콜드스타트 SFT → GRPO(검증가능 보상) → 온폴리시 증류.
근거: DeepSeek-R1(2501.12948) · Qwen3(2505.09388) · GLM-5(2602.15763, 클립 0.2/0.28) · DAPO 계열 · MAI-Thinking-1(#64).

핵심 함수는 모델 무관: `policy(ids) -> logits [B,T,V]`, 토크나이저는 encode/decode/eos_id/pad_id 를 가진 객체.
HF 모델(Qwen2.5-0.5B/1.5B)은 `HFPolicy`·`load_gsm8k` 로 연결(이 서버에선 가중치가 없어 미검증).

GRPO 한 스텝
 1) 프롬프트 P개 × G개 샘플(온도 1) → 2) 규칙 보상(정답 + 형식)
 3) 통과율 필터: 그룹 통과율이 0 또는 > max_pass 면 제외(Qwen3-VL·DAPO 관행)
 4) A_i = (R_i − mean) / (std + eps)  (그룹 상대)
 5) old/ref 로그확률 계산 → μ번 내부 업데이트, 토큰 레벨 전역 정규화:
    L = −(1/Σ|y|) ΣΣ [ min(r·A, clip(r, 1−ε_low, 1+ε_high)·A) − β·KL_k3 ]
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    group: int = 8
    prompts_per_step: int = 8
    max_new: int = 256
    temperature: float = 1.0
    eps_low: float = 0.2
    eps_high: float = 0.28
    beta_kl: float = 0.001
    inner_epochs: int = 1
    minibatch: int = 16
    lr: float = 1e-6
    min_pass: float = 0.0        # 통과율이 이 값 이하인 그룹 제외 (0 → '다 틀림' 제외)
    max_pass: float = 0.9        # 이 값 초과 그룹 제외
    format_reward: float = 0.1


# ----------------------------------------------------------------------------- 샘플링·로그확률
@torch.no_grad()
def generate(policy, prompt_ids: list[list[int]], max_new, temperature, eos_id, pad_id, device):
    """단순 자기회귀 샘플링(KV 캐시 없음 — 소형용). 반환: 완료 토큰 리스트들."""
    outs = []
    for p in prompt_ids:
        ids = torch.tensor([p], device=device)
        gen = []
        for _ in range(max_new):
            logits = policy(ids)[:, -1].float()
            nxt = torch.multinomial(F.softmax(logits / max(temperature, 1e-5), -1), 1) if temperature > 0 \
                else logits.argmax(-1, keepdim=True)
            tok = int(nxt)
            gen.append(tok)
            if tok == eos_id:
                break
            ids = torch.cat([ids, nxt], 1)
        outs.append(gen)
    return outs


def pack(prompts, comps, pad_id, device):
    """[프롬프트+완료] 오른쪽 패딩. 반환 ids, 완료 위치 마스크(다음 토큰 예측 기준)."""
    seqs = [p + c for p, c in zip(prompts, comps)]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T - 1))
    for i, (p, c) in enumerate(zip(prompts, comps)):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        mask[i, len(p) - 1:len(p) - 1 + len(c)] = 1.0         # 위치 t 의 로짓이 t+1 토큰을 예측
    return ids.to(device), mask.to(device)


def token_logprobs(policy, ids):
    logits = policy(ids[:, :-1]).float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0], logits


# ----------------------------------------------------------------------------- 손실
def grpo_loss(lp, old_lp, ref_lp, adv, mask, cfg: GRPOConfig):
    r = torch.exp(lp - old_lp)
    a = adv[:, None]
    surr = torch.minimum(r * a, torch.clamp(r, 1 - cfg.eps_low, 1 + cfg.eps_high) * a)
    if ref_lp is not None and cfg.beta_kl > 0:
        d = ref_lp - lp
        kl = torch.exp(d) - d - 1                                 # k3 추정
        surr = surr - cfg.beta_kl * kl
    return -(surr * mask).sum() / mask.sum().clamp_min(1)


def group_advantages(rewards, eps=1e-6):
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + eps)


# ----------------------------------------------------------------------------- 단계
def sft_step(policy, opt, tok, batch, device):
    """콜드스타트 SFT: 답 부분만 손실."""
    prompts = [tok.encode(x["prompt"]) for x in batch]
    comps = [tok.encode(x["target"]) + [tok.eos_id] for x in batch]
    ids, mask = pack(prompts, comps, tok.pad_id, device)
    lp, _ = token_logprobs(policy, ids)
    loss = -(lp * mask).sum() / mask.sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(_params(policy), 1.0); opt.step()
    return loss.item()


def grpo_step(policy, ref_policy, opt, tok, data, reward_fn, cfg: GRPOConfig, device, rng=random):
    batch = rng.sample(data, cfg.prompts_per_step)
    P, C, A, R_all, kept = [], [], [], [], 0
    pass_rates = []
    for ex in batch:
        p = tok.encode(ex["prompt"])
        comps = generate(policy, [p] * cfg.group, cfg.max_new, cfg.temperature, tok.eos_id, tok.pad_id, device)
        rewards = [reward_fn(tok.decode([t for t in c if t != tok.eos_id]), ex) for c in comps]
        pr = sum(1 for x in rewards if x >= 1.0) / len(rewards)
        pass_rates.append(pr); R_all += rewards
        if pr <= cfg.min_pass or pr > cfg.max_pass:
            continue
        kept += 1
        adv = group_advantages(rewards)
        P += [p] * cfg.group; C += comps; A.append(adv)
    stats = {"reward_mean": sum(R_all) / len(R_all), "pass_rate_mean": sum(pass_rates) / len(pass_rates),
             "groups_kept": kept, "loss": float("nan")}
    if kept == 0:
        return stats
    A = torch.cat(A).to(device)
    ids, mask = pack(P, C, tok.pad_id, device)
    with torch.no_grad():
        old_lp, _ = token_logprobs(policy, ids)
        ref_lp = token_logprobs(ref_policy, ids)[0] if (ref_policy is not None and cfg.beta_kl > 0) else None
    n = ids.shape[0]
    losses = []
    for _ in range(cfg.inner_epochs):
        perm = torch.randperm(n)
        # 토큰 레벨 전역 정규화: 미니배치 손실을 '전체 토큰 수'로 나눈 합이 되도록 가중
        total_tok = mask.sum()
        opt.zero_grad()
        for s in range(0, n, cfg.minibatch):
            j = perm[s:s + cfg.minibatch]
            lp, _ = token_logprobs(policy, ids[j])
            l = grpo_loss(lp, old_lp[j], None if ref_lp is None else ref_lp[j], A[j], mask[j], cfg)
            (l * mask[j].sum() / total_tok).backward(); losses.append(l.item())
        torch.nn.utils.clip_grad_norm_(_params(policy), 1.0)
        opt.step()
    stats["loss"] = sum(losses) / len(losses)
    return stats


def distill_step(student, teacher, opt, tok, data, cfg: GRPOConfig, device, rng=random):
    """온폴리시 증류(Qwen3 강→약): 학생 샘플 위에서 토큰별 KL(p_s || p_t) 최소화(전 어휘)."""
    batch = rng.sample(data, cfg.prompts_per_step)
    prompts = [tok.encode(ex["prompt"]) for ex in batch]
    comps = generate(student, prompts, cfg.max_new, cfg.temperature, tok.eos_id, tok.pad_id, device)
    ids, mask = pack(prompts, comps, tok.pad_id, device)
    with torch.no_grad():
        t_logits = teacher(ids[:, :-1]).float()
    s_logits = student(ids[:, :-1]).float()
    ls, lt = torch.log_softmax(s_logits, -1), torch.log_softmax(t_logits, -1)
    kl = (ls.exp() * (ls - lt)).sum(-1)
    loss = (kl * mask).sum() / mask.sum().clamp_min(1)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(_params(student), 1.0); opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(policy, tok, data, reward_fn, max_new, device):
    ok = 0
    for ex in data:
        c = generate(policy, [tok.encode(ex["prompt"])], max_new, 0.0, tok.eos_id, tok.pad_id, device)[0]
        ok += reward_fn(tok.decode([t for t in c if t != tok.eos_id]), ex) >= 1.0
    return ok / len(data)


def _params(policy):
    m = getattr(policy, "module", policy)
    return m.parameters() if hasattr(m, "parameters") else policy.parameters()


# ----------------------------------------------------------------------------- GSM8K / HF 연결 (미검증)
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_reward(text, ex, format_bonus=0.1):
    """정답 = 마지막 숫자 일치 → 1.0, '####' 형식 사용 시 +format_bonus (R1식 정확도+형식 보상)."""
    nums = NUM_RE.findall(text.split("####")[-1]) or NUM_RE.findall(text)
    pred = nums[-1].replace(",", "").rstrip(".") if nums else None
    gold = ex["answer"]
    r = 1.0 if pred is not None and _num_eq(pred, gold) else 0.0
    return r + (format_bonus if "####" in text else 0.0) * (r > 0)


def _num_eq(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False


def load_gsm8k(split="train"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for r in ds:
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"prompt": f"Question: {r['question']}\nAnswer: Let's think step by step.\n",
                    "target": r["answer"], "answer": gold})
    return out


class HFTokenizer:
    def __init__(self, name):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(name)
        self.eos_id = self.t.eos_token_id
        self.pad_id = self.t.pad_token_id if self.t.pad_token_id is not None else self.eos_id

    def encode(self, s):
        return self.t.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=True)


class HFPolicy(torch.nn.Module):
    def __init__(self, name, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    def forward(self, ids):
        return self.m(input_ids=ids).logits
