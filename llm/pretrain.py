"""소형 LLM 사전학습 — 일반 → 어닐링(수학·코드↑, LR 감쇠) → 문맥 확장(RoPE 기저↑) 단계.

    python llm/pretrain.py --config llm/configs/tierB.json --out runs/llm/dense
    torchrun --nproc_per_node 2 llm/pretrain.py --config llm/configs/tierA.json --out runs/llm/dense

데이터: uint16 토큰 샤드 *.bin (prepare_data.py). 설정의 stages 마다 data=[[glob, weight], ...] 혼합.
LR 스케줄(단계별): warmup → shape ∈ {const, linear, cosine} 로 lr_start → lr_end.
  WSD = [일반: const] + [어닐링: linear → 10%]   /   코사인 기준선 = 단일 단계 cosine
MoE aux-free 편향은 옵티마이저 스텝 후 갱신. 전문가 부하 최대/평균을 로그.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig  # noqa: E402


class Mixture:
    def __init__(self, spec, seq, seed):
        self.srcs, self.w = [], []
        for pat, w in spec:
            files = sorted(glob.glob(pat))
            if not files:
                raise FileNotFoundError(pat)
            self.srcs.append([np.memmap(f, dtype=np.uint16, mode="r") for f in files]); self.w.append(w)
        self.w = np.array(self.w, float) / sum(self.w)
        self.seq, self.rng = seq, np.random.default_rng(seed)

    def batch(self, B):
        xs = []
        for _ in range(B):
            src = self.srcs[self.rng.choice(len(self.srcs), p=self.w)]
            arr = src[self.rng.integers(len(src))]
            i = self.rng.integers(0, len(arr) - self.seq - 1)
            xs.append(np.asarray(arr[i:i + self.seq + 1], dtype=np.int64))
        x = torch.from_numpy(np.stack(xs))
        return x[:, :-1], x[:, 1:]


def lr_at(step, total, st):
    w = st.get("warmup", 0)
    if step < w:
        return st["lr_start"] * (step + 1) / w
    p = (step - w) / max(1, total - w)
    a, b = st["lr_start"], st.get("lr_end", st["lr_start"])
    shape = st.get("shape", "const")
    if shape == "const":
        return a
    if shape == "linear":
        return a + (b - a) * p
    return b + (a - b) * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def val_loss(model, data, B, iters, dev, bf16):
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y = data.batch(B)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            _, l = model(x.to(dev), y.to(dev))
        tot += l.item()
    model.train()
    return tot / iters


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank() if world > 1 else 0
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}")
    torch.manual_seed(cfg.get("seed", 0))
    model = GPT(GPTConfig(**cfg["model"])).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    decay = [p for n, p in core.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in core.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.get("wd", 0.1)},
                             {"params": no_decay, "weight_decay": 0.0}], lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
    bf16 = dev.type == "cuda" and cfg.get("bf16", True)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        print(f"파라미터 {core.num_params()/1e6:.1f}M (활성 {core.num_params(True)/1e6:.1f}M)")
    log = open(out / "log.jsonl", "a") if rank == 0 else None
    t0, gstep = time.time(), 0
    for si, st in enumerate(cfg["stages"]):
        seq = st["seq_len"]
        if "rope_base" in st:
            core.set_rope_base(st["rope_base"])
        data = Mixture(st["data"], seq, cfg.get("seed", 0) * 100 + si * 10 + rank)
        val = Mixture(cfg["val_data"], seq, 12345) if cfg.get("val_data") else None
        mb, accum = st["micro_batch"], st["accum"]
        tok_per_step = mb * accum * seq * world
        steps = st.get("steps") or int(st["tokens"] // tok_per_step)
        for step in range(steps):
            lr = lr_at(step, steps, st)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for _ in range(accum):
                x, y = data.batch(mb)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    _, loss = model(x.to(dev), y.to(dev))
                (loss / accum).backward(); tot += loss.item() / accum
            gn = torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.get("clip", 1.0))
            opt.step()
            for m in core.moe_layers():
                m.update_bias()
            if rank == 0 and (step % cfg.get("log_every", 50) == 0 or step == steps - 1):
                rec = {"stage": si, "step": step, "gstep": gstep, "loss": tot, "lr": lr, "grad_norm": float(gn),
                       "tok_per_s": round(tok_per_step * (step + 1) / max(1e-9, time.time() - t0)), "sec": round(time.time() - t0, 1)}
                loads = [m.last_load for m in core.moe_layers() if m.last_load is not None]
                if loads:
                    L = torch.stack(loads)
                    rec["moe_load_max_over_mean"] = float((L.max(1).values / L.mean(1).clamp_min(1e-9)).mean())
                if val is not None and (step % cfg.get("eval_every", 500) == 0 or step == steps - 1):
                    rec["val_loss"] = val_loss(core, val, mb, cfg.get("eval_iters", 20), dev, bf16)
                log.write(json.dumps(rec) + "\n"); log.flush()
                if not math.isfinite(tot):
                    raise RuntimeError(f"발산 {rec}")
            gstep += 1
        if rank == 0:
            torch.save({"model": core.state_dict(), "stage": si, "config": cfg["model"]}, out / f"stage{si}.pt")
    if rank == 0:
        log.close()
    if world > 1:
        dist.destroy_process_group()
    return out


if __name__ == "__main__":
    main()
