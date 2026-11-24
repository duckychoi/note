"""깊이 모델 학습 — 단계: teacher | student | metric | sharpen.

    # 단일 GPU (티어 B)
    python depth/train.py --config depth/configs/tierB.json --stage teacher --out runs/depth/teacher
    # 2 GPU (티어 A)
    torchrun --nproc_per_node 2 depth/train.py --config depth/configs/tierA.json --stage teacher --out runs/depth/teacher

단계별 목표 공간
- teacher : 합성 라벨의 시차(1/깊이), 손실 = SSI-L1 + gm_weight·L_gm               (DA2 교사)
- student : 가짜 라벨 시차, 손실 = SSI-L1(상위 10% 무시) [+ 특징 정렬]             (DA2 학생)
- metric  : 정준 깊이 d·f_c/f (CSTM-label), 손실 = SIlog                          (Metric3D·DA3 f_c=300)
- sharpen : 합성만, 상대 모델이면 SSI 정규화 후 / 미터 모델이면 정준 역깊이에서
            MAE+MSE+MAGE+MALE+MSGE                                                  (Depth Pro 2단계)
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

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import losses as L  # noqa: E402
from model import DepthModel  # noqa: E402


def depth_pro_lr(step, total, base):
    """Depth Pro 스케줄: 1% 선형 워밍업, 80% 상수, 마지막 19% ×0.1."""
    w = max(1, int(0.01 * total))
    if step < w:
        return base * (step + 1) / w
    return base if step < int(0.81 * total) else base * 0.1


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, dist.get_world_size()
    return 0, 1


def build_loader(ds, batch, world, rank, workers):
    sampler = torch.utils.data.DistributedSampler(ds, world, rank, shuffle=True) if world > 1 else None
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=sampler is None, sampler=sampler,
                                       num_workers=workers, collate_fn=D.collate, drop_last=True,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def infinite(loader):
    epoch = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for b in loader:
            yield b
        epoch += 1


def compute_loss(stage, cfg, model, batch, frozen_enc, device):
    img = batch["image"].to(device, non_blocking=True)
    if stage == "student":
        disp, valid = batch["disp"].to(device), batch["valid"].to(device)
        pred, feats = model(img, return_feats=True)
        loss = L.student_loss(pred, disp, valid, trim=cfg.get("trim", 0.1))
        if frozen_enc is not None and cfg.get("feat_align", 0) > 0:
            with torch.no_grad():
                ref = frozen_enc.forward_intermediates(img, indices=getattr(model, "module", model).layers[-1:], intermediates_only=True,
                                                       output_fmt="NCHW")[0]
            loss = loss + cfg["feat_align"] * L.feature_alignment(feats[-1], ref, cfg.get("feat_margin", 0.85))
        return loss
    depth, valid = batch["depth"].to(device), batch["valid"].to(device)
    if stage == "teacher":
        target = torch.where(valid, 1.0 / depth.clamp_min(1e-6), torch.zeros_like(depth))
        return L.teacher_loss(model(img), target, valid, cfg.get("gm_weight", 2.0))
    fx = batch["fx"].to(device)
    if stage == "metric":
        ok = torch.isfinite(fx)
        if not ok.any():
            raise ValueError("metric 단계는 intrinsics.json(초점거리)이 필요")
        canon = depth * (cfg.get("f_c", 300.0) / fx.clamp_min(1e-6))[:, None, None]
        v = valid & ok[:, None, None]
        return L.silog(model(img), canon, v)
    if stage == "sharpen":
        pred = model(img)
        if cfg.get("sharpen_space", "ssi_disp") == "metric":
            canon_inv = (fx / cfg.get("f_c", 300.0))[:, None, None] / depth.clamp_min(1e-6)
            target = torch.where(valid, canon_inv, torch.zeros_like(depth))
            pred = 1.0 / pred.clamp_min(1e-6)
            return L.sharpen_loss(pred, target, valid)
        target = torch.where(valid, 1.0 / depth.clamp_min(1e-6), torch.zeros_like(depth))
        p, t = L.normalize_ssi(pred, valid), L.normalize_ssi(target, valid)
        return L.sharpen_loss(p, t, valid)
    raise ValueError(stage)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["teacher", "student", "metric", "sharpen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="이전 단계 체크포인트(.pt)")
    ap.add_argument("--steps", type=int, default=None, help="설정값 덮어쓰기(탐침·스모크용)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    full = json.loads(Path(args.config).read_text())
    cfg = {**full.get("common", {}), **full[args.stage]}
    steps = args.steps or cfg["steps"]
    rank, world = setup_dist()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}")
    torch.manual_seed(cfg.get("seed", 0) + rank)

    model = DepthModel(cfg["encoder"], pretrained=cfg.get("pretrained", True), timm_name=cfg.get("timm_name"),
                       layers=cfg.get("layers"), feat=cfg.get("feat"))
    if cfg.get("encoder_kwargs"):  # 스모크용 축소 인코더
        import timm
        model.encoder = timm.create_model(cfg["timm_name"], pretrained=False, dynamic_img_size=True,
                                          num_classes=0, **cfg["encoder_kwargs"])
        model.head = type(model.head)(model.encoder.embed_dim, cfg.get("feat", 64))
    frozen_enc = copy.deepcopy(model.encoder).eval().requires_grad_(False) if args.stage == "student" else None
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.to(device)
    if frozen_enc is not None:
        frozen_enc.to(device)
    if cfg.get("grad_ckpt") and hasattr(model.encoder, "set_grad_checkpointing"):
        model.encoder.set_grad_checkpointing(True)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=False)
    core = model.module if world > 1 else model

    size = cfg["size"]
    if args.stage == "student":
        ds = D.PseudoLabeled(cfg["pseudo_root"], size)
    else:
        roots = cfg["data_roots"]
        ds = torch.utils.data.ConcatDataset([D.LabeledDepth(r, size, True, max_depth=cfg.get("max_depth")) for r in roots])
    loader = infinite(build_loader(ds, cfg["batch_per_gpu"], world, rank, cfg.get("workers", 4)))

    lr = cfg["lr_decoder"]
    opt = torch.optim.AdamW(core.param_groups(lr, cfg.get("enc_ratio", 0.1), cfg.get("wd", 0.0)), betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    accum = cfg.get("accum", 1)
    use_bf16 = device.type == "cuda" and cfg.get("bf16", True)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config_used.json").write_text(json.dumps({"stage": args.stage, "steps": steps, **cfg}, indent=1))
    log = open(out / "log.jsonl", "a") if rank == 0 else None

    t0 = time.time()
    model.train()
    for step in range(steps):
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = depth_pro_lr(step, steps, b)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            batch = next(loader)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = compute_loss(args.stage, cfg, model, batch, frozen_enc, device) / accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        opt.step()
        if rank == 0 and (step % cfg.get("log_every", 50) == 0 or step == steps - 1):
            rec = {"step": step, "loss": tot, "grad_norm": float(gn), "lr": opt.param_groups[1]["lr"],
                   "sec": round(time.time() - t0, 1)}
            if device.type == "cuda":
                rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if not math.isfinite(tot):
                raise RuntimeError(f"손실 발산: {rec}")
        if rank == 0 and (step + 1) % cfg.get("save_every", 5000) == 0:
            torch.save({"model": core.state_dict(), "step": step + 1}, out / f"step{step+1}.pt")
    if rank == 0:
        torch.save({"model": core.state_dict(), "step": steps}, out / "final.pt")
        log.close()
    if world > 1:
        dist.destroy_process_group()
    return out / "final.pt"


if __name__ == "__main__":
    main()
