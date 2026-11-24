"""Rectified-flow DiT 커리큘럼 학습 (Wan·Movie Gen 단계 구조 축소판).

    python gen/train.py --config gen/configs/tierB.json --out runs/gen
    torchrun --nproc_per_node 2 gen/train.py --config gen/configs/tierA.json --out runs/gen

잠재는 미리 계산(precompute_latents.py): <dir>/latents.npy [N,C,T,H,W] float16 + <dir>/labels.npy [N] int.
설정의 stages 를 순서대로 진행. 각 단계는 이미지 디렉터리와(선택) 영상 디렉터리를 가지며,
스텝마다 image_ratio 확률로 이미지 배치, 아니면 영상 배치(배치 안은 같은 모양) — 영상 단계에서도 이미지를 섞어 망각 방지(Wan·Hunyuan).
해상도 시프트 α = √(현재 토큰 / base_tokens), EMA 유지.
"""
from __future__ import annotations

import argparse
import copy
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
import flow as FL  # noqa: E402
from dit import DiT, num_params  # noqa: E402


class LatentSet:
    def __init__(self, d):
        d = Path(d)
        self.x = np.load(d / "latents.npy", mmap_mode="r")
        self.y = np.load(d / "labels.npy")

    def batch(self, B, rng):
        idx = rng.integers(0, len(self.y), B)
        return torch.from_numpy(np.ascontiguousarray(self.x[np.sort(idx)]).astype(np.float32)), \
            torch.from_numpy(self.y[np.sort(idx)].astype(np.int64))


def lr_at(step, total, base, warmup=1000, final_frac=1.0):
    if step < warmup:
        return base * (step + 1) / warmup
    if final_frac == 1.0:
        return base
    p = (step - warmup) / max(1, total - warmup)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--only_stage", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank() if world > 1 else 0
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}")
    torch.manual_seed(cfg.get("seed", 0) + rank)
    rng = np.random.default_rng(cfg.get("seed", 0) + rank)

    m = cfg["model"]
    model = DiT(in_ch=m["in_ch"], d=m["d"], depth=m["depth"], heads=m["heads"], d_ctx=m.get("d_ctx", m["d"]),
                n_classes=m["n_classes"], shared_mod=m.get("shared_mod", True)).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        print(f"DiT 파라미터 {num_params(core)/1e6:.1f}M")
    log = open(out / "log.jsonl", "a") if rank == 0 else None
    opt = torch.optim.AdamW(core.parameters(), lr=cfg["stages"][0]["lr"], weight_decay=cfg.get("wd", 1e-3),
                            betas=(0.9, 0.99))
    use_bf16 = dev.type == "cuda" and cfg.get("bf16", True)
    ema_decay = cfg.get("ema", 0.9999)
    gstep, t0 = 0, time.time()
    for si, st in enumerate(cfg["stages"]):
        if a.only_stage is not None and si != a.only_stage:
            continue
        img = LatentSet(st["image_dir"]) if st.get("image_dir") else None
        vid = LatentSet(st["video_dir"]) if st.get("video_dir") else None
        ratio = st.get("image_ratio", 1.0 if vid is None else 0.0)
        for step in range(st["steps"]):
            use_img = vid is None or (img is not None and rng.random() < ratio)
            src, B = (img, st["batch_image"]) if use_img else (vid, st["batch_video"])
            x0, y = src.batch(B, rng)
            x0, y = x0.to(dev), y.to(dev)
            tokens = x0.shape[2] * (x0.shape[3] // 2) * (x0.shape[4] // 2)
            alpha = FL.resolution_alpha(tokens, cfg.get("base_tokens", tokens)) if cfg.get("res_shift", True) else 1.0
            for g in opt.param_groups:
                g["lr"] = lr_at(step, st["steps"], st["lr"], st.get("warmup", 1000), st.get("final_lr_frac", 1.0))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = FL.flow_loss(model, x0, y, alpha, cfg.get("t_mode", "logit_normal"), cfg.get("p_drop", 0.1),
                                    core.null_label())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.get("clip", 1.0))
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), core.parameters()):
                    pe.lerp_(pm, 1 - ema_decay)
            if rank == 0 and (step % cfg.get("log_every", 100) == 0 or step == st["steps"] - 1):
                rec = {"stage": si, "step": step, "gstep": gstep, "loss": loss.item(), "grad_norm": float(gn),
                       "kind": "image" if use_img else "video", "alpha": round(alpha, 3), "sec": round(time.time() - t0, 1)}
                if dev.type == "cuda":
                    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.write(json.dumps(rec) + "\n"); log.flush()
                if not math.isfinite(rec["loss"]):
                    raise RuntimeError(f"발산 {rec}")
            gstep += 1
        if rank == 0:
            torch.save({"model": core.state_dict(), "ema": ema.state_dict(), "stage": si}, out / f"stage{si}.pt")
    if rank == 0:
        log.close()
    if world > 1:
        dist.destroy_process_group()
    return out


if __name__ == "__main__":
    main()
