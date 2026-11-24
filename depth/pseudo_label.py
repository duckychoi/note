"""교사로 무라벨 실사진에 가짜 라벨(시차)을 붙인다 — DA2 2단계.

    python depth/pseudo_label.py --config depth/configs/tierA.json --ckpt runs/depth/teacher/final.pt \
        --images /data/coco_unlabeled --out /data/pseudo
출력: out/img/<stem>.jpg (처리된 입력 그대로) + out/disp/<stem>.npy (float16 시차, 같은 크기).
수평 뒤집기 TTA 평균. 여러 GPU면 --shard i --nshards n 으로 나눠 동시에 실행.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
from model import DepthModel  # noqa: E402


def build_model(cfg, ckpt, device):
    model = DepthModel(cfg["encoder"], pretrained=False, timm_name=cfg.get("timm_name"),
                       layers=cfg.get("layers"), feat=cfg.get("feat"))
    if cfg.get("encoder_kwargs"):
        import timm
        model.encoder = timm.create_model(cfg["timm_name"], pretrained=False, dynamic_img_size=True,
                                          num_classes=0, **cfg["encoder_kwargs"])
        model.head = type(model.head)(model.encoder.embed_dim, cfg.get("feat", 64))
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    return model.to(device).eval()


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    full = json.loads(Path(args.config).read_text())
    cfg = {**full.get("common", {}), **full["teacher"]}
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(cfg, args.ckpt, device)
    ds = D.UnlabeledImages(args.images, cfg["size"])
    idx = list(range(args.shard, len(ds), args.nshards))
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True); (out / "disp").mkdir(exist_ok=True)
    use_bf16 = device.type == "cuda"
    for s in range(0, len(idx), args.batch):
        items = [ds[i] for i in idx[s:s + args.batch]]
        x = torch.stack([it["image"] for it in items]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            y = model(x) + model(x.flip(-1)).flip(-1)
        y = (y.float() / 2).cpu().numpy().astype(np.float16)
        for it, d in zip(items, y):
            Image.fromarray(it["pil"]).save(out / "img" / f"{it['stem']}.jpg", quality=95)
            np.save(out / "disp" / f"{it['stem']}.npy", d)
    return len(idx)


if __name__ == "__main__":
    print(main(), "장 라벨링")
