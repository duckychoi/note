"""깊이 데이터셋 — 단순 폴더 형식(공개 데이터셋은 변환해서 이 형식으로 맞춘다).

라벨 있는 데이터(합성·실측·평가):
    root/rgb/<stem>.png|jpg
    root/depth/<stem>.npy          # float32, 미터 단위 깊이 (0 또는 inf = 무효)
    root/valid/<stem>.png          # (선택) 0/255 유효 마스크
    root/intrinsics.json           # (선택) {"<stem>": fx_px, ...} 또는 {"*": fx_px}
무라벨 실사진: root/**/*.jpg|png
가짜 라벨: pseudo_root/img/<stem>.jpg + pseudo_root/disp/<stem>.npy (pseudo_label.py 출력)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

IMG_EXT = {".jpg", ".jpeg", ".png"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img.convert("RGB"), np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def _resize_crop(img, maps, size, scale_range, rng, train):
    """짧은 변을 size*scale 로 맞춘 뒤 size×size 크롭(학습은 무작위, 평가는 중앙). maps 는 최근접 보간."""
    w, h = img.size
    s = rng.uniform(*scale_range) if train else 1.0
    short = int(round(size * s))
    if w < h:
        nw, nh = short, int(round(h * short / w))
    else:
        nw, nh = int(round(w * short / h)), short
    nw, nh = max(nw, size), max(nh, size)
    img = img.resize((nw, nh), Image.BICUBIC)
    maps = [Image.fromarray(m).resize((nw, nh), Image.NEAREST) if m is not None else None for m in maps]
    if train:
        x0, y0 = rng.randint(0, nw - size), rng.randint(0, nh - size)
    else:
        x0, y0 = (nw - size) // 2, (nh - size) // 2
    box = (x0, y0, x0 + size, y0 + size)
    img = img.crop(box)
    maps = [np.asarray(m.crop(box)) if m is not None else None for m in maps]
    return img, maps, nw / w      # 마지막: 리사이즈 배율(초점거리 보정용)


def _photometric(img, rng):
    from PIL import ImageEnhance
    if rng.random() < 0.75:                                   # Depth Pro: 색 변화 p=0.75
        for E in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = E(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:                                    # 블러 p=0.3
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.5)))
    return img


class LabeledDepth(torch.utils.data.Dataset):
    """반환: image[3,S,S], depth[S,S](미터), valid[S,S] bool, fx(리사이즈 반영, px; 없으면 nan)."""

    def __init__(self, root, size=518, train=True, scale_range=(1.0, 1.5), max_depth=None):
        self.root = Path(root)
        self.items = sorted(p for p in (self.root / "rgb").iterdir() if p.suffix.lower() in IMG_EXT)
        self.size, self.train, self.scale_range, self.max_depth = size, train, scale_range, max_depth
        intr = self.root / "intrinsics.json"
        self.fx = json.loads(intr.read_text()) if intr.exists() else {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = random.Random() if self.train else random.Random(i)
        p = self.items[i]
        img = Image.open(p).convert("RGB")
        depth = np.load(self.root / "depth" / f"{p.stem}.npy").astype(np.float32)
        vpath = self.root / "valid" / f"{p.stem}.png"
        valid = np.asarray(Image.open(vpath)) > 0 if vpath.exists() else np.ones_like(depth, bool)
        valid &= np.isfinite(depth) & (depth > 0)
        if self.max_depth:
            valid &= depth < self.max_depth
        depth = np.where(valid, depth, 0).astype(np.float32)
        img, (depth, valid), ratio = _resize_crop(img, [depth, valid.astype(np.uint8)], self.size,
                                                  self.scale_range, rng, self.train)
        if self.train:
            img = _photometric(img, rng)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                depth, valid = depth[:, ::-1], valid[:, ::-1]
        fx = self.fx.get(p.stem, self.fx.get("*"))
        fx = float(fx) * ratio if fx is not None else float("nan")
        return {"image": to_tensor(img), "depth": torch.from_numpy(depth.copy()),
                "valid": torch.from_numpy(valid.copy().astype(bool)), "fx": torch.tensor(fx), "stem": p.stem}


class UnlabeledImages(torch.utils.data.Dataset):
    """가짜 라벨 생성용: 짧은 변 size 로 리사이즈 + 중앙 크롭(정렬 보존을 위해 처리된 이미지를 함께 저장)."""

    def __init__(self, root, size=518):
        self.items = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        img, _, _ = _resize_crop(Image.open(p).convert("RGB"), [], self.size, (1, 1), random.Random(0), False)
        return {"image": to_tensor(img), "pil": np.asarray(img), "stem": f"{i:08d}_{p.stem}"}


class PseudoLabeled(torch.utils.data.Dataset):
    """학생 학습용: 가짜 라벨(시차) + 이미지. 같은 증강을 함께 적용."""

    def __init__(self, root, size=518, scale_range=(1.0, 1.3)):
        self.root = Path(root)
        self.items = sorted((self.root / "img").glob("*.jpg"))
        self.size, self.scale_range = size, scale_range

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = random.Random()
        p = self.items[i]
        img = Image.open(p).convert("RGB")
        disp = np.load(self.root / "disp" / f"{p.stem}.npy").astype(np.float32)
        valid = np.isfinite(disp) & (disp > 0)
        img, (disp, valid), _ = _resize_crop(img, [disp, valid.astype(np.uint8)], self.size, self.scale_range, rng, True)
        img = _photometric(img, rng)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            disp, valid = disp[:, ::-1], valid[:, ::-1]
        return {"image": to_tensor(img), "disp": torch.from_numpy(disp.copy()),
                "valid": torch.from_numpy(valid.copy().astype(bool))}


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    return out
