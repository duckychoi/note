"""HF 모델 후학습 실행기 (GPU 머신용; 이 서버에선 가중치가 없어 미검증).

    pip install transformers datasets accelerate
    # 티어 B(5090): Qwen2.5-0.5B
    python llm/run_posttrain.py --model Qwen/Qwen2.5-0.5B --out runs/post/0.5b --sft_steps 300 --grpo_steps 400
    # 티어 A(A6000x2): Qwen2.5-1.5B (생성 속도를 위해 vLLM 사용 권장 — 이 스크립트는 HF 샘플링, 느림)
    python llm/run_posttrain.py --model Qwen/Qwen2.5-1.5B --out runs/post/1.5b --group 16 --max_new 512
    # 증류(HL8): 교사 1.5B-Instruct → 학생 0.5B
    python llm/run_posttrain.py --model Qwen/Qwen2.5-0.5B --teacher Qwen/Qwen2.5-1.5B-Instruct --distill_steps 400 --out runs/post/distill

비교 팔(사전등록 HL5~HL8)은 --eps_high 0.2(대칭), --max_pass 1.01(필터 끔) 등으로 같은 스텝·같은 GPU시간에 돌린다.
"""
import argparse
import copy
import json
import random
import sys
import os
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpo as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--teacher", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft_steps", type=int, default=0); ap.add_argument("--sft_lr", type=float, default=1e-5)
    ap.add_argument("--grpo_steps", type=int, default=0); ap.add_argument("--distill_steps", type=int, default=0)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=384); ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--eps_high", type=float, default=0.28); ap.add_argument("--max_pass", type=float, default=0.9)
    ap.add_argument("--beta_kl", type=float, default=0.001); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=1319)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    tok = G.HFTokenizer(a.model)
    pol = G.HFPolicy(a.model).to(dev)
    pol.m.gradient_checkpointing_enable()
    train, test = G.load_gsm8k("train"), G.load_gsm8k("test")[:a.eval_n]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    cfg = G.GRPOConfig(group=a.group, prompts_per_step=a.prompts, max_new=a.max_new, lr=a.lr, eps_high=a.eps_high,
                       max_pass=a.max_pass, beta_kl=a.beta_kl)
    if a.sft_steps:
        opt = torch.optim.AdamW(pol.parameters(), lr=a.sft_lr, weight_decay=0.0)
        for s in range(a.sft_steps):
            l = G.sft_step(pol, opt, tok, random.sample(train, 8), dev)
            if s % 20 == 0:
                log.write(json.dumps({"phase": "sft", "step": s, "loss": l}) + "\n"); log.flush()
    if a.grpo_steps:
        ref = copy.deepcopy(pol).eval().requires_grad_(False)
        opt = torch.optim.AdamW(pol.parameters(), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.99))
        for s in range(a.grpo_steps):
            st = G.grpo_step(pol, ref, opt, tok, train, G.gsm8k_reward, cfg, dev)
            log.write(json.dumps({"phase": "grpo", "step": s, **st}) + "\n"); log.flush()
    if a.distill_steps:
        teacher = G.HFPolicy(a.teacher).to(dev).eval().requires_grad_(False)
        opt = torch.optim.AdamW(pol.parameters(), lr=1e-6, weight_decay=0.0)
        for s in range(a.distill_steps):
            l = G.distill_step(pol, teacher, opt, tok, train, cfg, dev)
            log.write(json.dumps({"phase": "distill", "step": s, "kl": l}) + "\n"); log.flush()
    acc = G.evaluate(pol, tok, test, G.gsm8k_reward, a.max_new, dev)
    (out / "eval.json").write_text(json.dumps({"gsm8k_test_acc": acc, "n": len(test), "args": vars(a)}, indent=1))
    print("GSM8K test acc", acc)


if __name__ == "__main__":
    main()
