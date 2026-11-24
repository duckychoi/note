"""GRPO 방향(부호) 양성 대조: 같은 SFT 시작점에서 (a) 정상 GRPO vs (b) 어드밴티지를 무작위로 섞은 대조군.
정상 GRPO의 보상 상승이 대조군보다 커야 부호·마스크가 맞다. 시드 3개."""
import random, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpo as G
from model import GPT, GPTConfig
from smoke_test import CharTok, CHARS

def run(seed, shuffle_adv, steps=40, lr=5e-5):
    random.seed(seed); torch.manual_seed(seed)
    tok = CharTok()
    data = [{"prompt": f"{a}+{b}=", "target": str(a+b), "answer": str(a+b)} for a in range(10) for b in range(10)]
    reward = lambda t, ex: 1.0 if t.strip() == ex["answer"] else 0.0
    m = GPT(GPTConfig(vocab=len(CHARS)+2, d=64, n_layer=3, n_head=4, n_kv_head=2))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(150):
        G.sft_step(m, opt, tok, random.sample(data, 16), "cpu")
    base = G.evaluate(m, tok, data, reward, 3, "cpu")
    orig = G.group_advantages
    if shuffle_adv:
        G.group_advantages = lambda r, eps=1e-6: orig(r, eps)[torch.randperm(len(r))]
    try:
        cfg = G.GRPOConfig(group=8, prompts_per_step=8, max_new=3, lr=lr, minibatch=64, inner_epochs=1, max_pass=0.99, beta_kl=0.0)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        rs = [G.grpo_step(m, None, opt, tok, data, reward, cfg, "cpu")["reward_mean"] for _ in range(steps)]
    finally:
        G.group_advantages = orig
    return base, G.evaluate(m, tok, data, reward, 3, "cpu"), np.mean(rs[:10]), np.mean(rs[-10:])

for name, sh in (("GRPO", False), ("대조(어드밴티지 섞음)", True)):
    res = [run(s, sh) for s in range(3)]
    d_acc = [r[1]-r[0] for r in res]; d_rew = [r[3]-r[2] for r in res]
    print(f"{name}: 탐욕 정확도 변화 {np.round(d_acc,3)} 평균 {np.mean(d_acc):+.3f} | 샘플 보상 처음10→마지막10 변화 {np.round(d_rew,3)} 평균 {np.mean(d_rew):+.3f}")
