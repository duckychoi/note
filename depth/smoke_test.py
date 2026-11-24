"""깊이 파이프라인 전 단계 스모크 테스트(CPU, 수 분). 실제 성능 검증이 아니라 '끝까지 돈다 + 학습 신호가 있다' 확인.

    python depth/smoke_test.py
합성 장면(기울어진 평면 + 가까운 직사각형, 색이 깊이와 상관)을 만들어
teacher → pseudo_label → student(+특징 정렬) → metric → sharpen → eval(relative/metric/boundary) 순으로 실행.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E  # noqa: E402
import pseudo_label as P  # noqa: E402
import train as T  # noqa: E402

S = 56  # 14의 배수


def make_scene(rng, H=72, W=96):
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    depth = 2.0 + 3.0 * yy + rng.uniform(-1, 1) * xx            # 기울어진 바닥/벽
    for _ in range(rng.integers(1, 3)):                         # 가까운 물체
        y0, x0 = rng.integers(0, H - 20), rng.integers(0, W - 20)
        h, w = rng.integers(10, 20), rng.integers(10, 25)
        depth[y0:y0 + h, x0:x0 + w] = rng.uniform(0.8, 1.6)
    base = np.stack([1 / depth, 0.3 + 0.1 * np.sin(20 * xx), yy], -1)
    rgb = np.clip(base / base.max((0, 1)) * 255 + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    return rgb, depth.astype(np.float32)


def write_labeled(root, n, seed, fx=80.0):
    rng = np.random.default_rng(seed)
    for d in ("rgb", "depth"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rgb, depth = make_scene(rng)
        Image.fromarray(rgb).save(root / "rgb" / f"{i:04d}.png")
        np.save(root / "depth" / f"{i:04d}.npy", depth)
    (root / "intrinsics.json").write_text(json.dumps({"*": fx}))


def write_unlabeled(root, n, seed):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(make_scene(rng)[0]).save(root / f"u{i:04d}.jpg")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="depth_smoke_"))
    try:
        syn, unl, ev = tmp / "syn", tmp / "unl", tmp / "eval"
        write_labeled(syn, 24, 0); write_unlabeled(unl, 16, 1); write_labeled(ev, 6, 2)
        tiny = {"encoder": "custom", "timm_name": "vit_small_patch14_dinov2", "pretrained": False,
                "encoder_kwargs": {"depth": 4, "embed_dim": 96, "num_heads": 3}, "layers": [0, 1, 2, 3], "feat": 16,
                "size": S, "batch_per_gpu": 4, "accum": 1, "workers": 0, "log_every": 1, "save_every": 10**9}
        cfg = {"common": {"f_c": 300.0, "bf16": False, "wd": 0.0},
               "teacher": {**tiny, "steps": 60, "lr_decoder": 2e-3, "gm_weight": 2.0, "data_roots": [str(syn)]},
               "student": {**tiny, "steps": 8, "lr_decoder": 1e-3, "trim": 0.1, "feat_align": 1.0,
                           "pseudo_root": str(tmp / "pseudo")},
               "metric": {**tiny, "steps": 6, "lr_decoder": 1e-3, "data_roots": [str(syn)]},
               "sharpen": {**tiny, "steps": 6, "lr_decoder": 5e-4, "data_roots": [str(syn)], "sharpen_space": "ssi_disp"}}
        cpath = tmp / "smoke.json"; cpath.write_text(json.dumps(cfg))
        c = str(cpath)

        teacher = T.main(["--config", c, "--stage", "teacher", "--out", str(tmp / "r/teacher"), "--cpu"])
        losses = [json.loads(l)["loss"] for l in open(tmp / "r/teacher/log.jsonl")]
        first, last = np.mean(losses[:10]), np.mean(losses[-10:])
        print(f"[teacher] loss 처음10 {first:.3f} → 마지막10 {last:.3f}")
        assert np.all(np.isfinite(losses)) and last < first, "교사 손실이 줄지 않음"

        n = P.main(["--config", c, "--ckpt", str(teacher), "--images", str(unl), "--out", str(tmp / "pseudo"), "--batch", "4", "--cpu"])
        assert n == 16 and len(list((tmp / "pseudo/disp").glob("*.npy"))) == 16
        print(f"[pseudo] {n}장 라벨")

        student = T.main(["--config", c, "--stage", "student", "--out", str(tmp / "r/student"), "--cpu"])
        metric = T.main(["--config", c, "--stage", "metric", "--out", str(tmp / "r/metric"), "--init", str(student), "--cpu"])
        sharp = T.main(["--config", c, "--stage", "sharpen", "--out", str(tmp / "r/sharpen"), "--init", str(student), "--cpu"])
        for name in ("student", "metric", "sharpen"):
            ls = [json.loads(l)["loss"] for l in open(tmp / f"r/{name}/log.jsonl")]
            assert np.all(np.isfinite(ls)), name
            print(f"[{name}] loss {ls[0]:.3f} → {ls[-1]:.3f} (스텝 {len(ls)})")

        r_rel = E.main(["--config", c, "--ckpt", str(teacher), "--data", str(ev), "--mode", "relative", "--boundary",
                        "--out", str(tmp / "e/rel.json"), "--cpu"])
        r_met = E.main(["--config", c, "--ckpt", str(metric), "--data", str(ev), "--mode", "metric",
                        "--out", str(tmp / "e/met.json"), "--cpu"])
        for r in (r_rel, r_met):
            assert all(np.isfinite(v) for v in r["summary"].values())
        print("[eval] relative", r_rel["summary"], "| metric", r_met["summary"])
        print("SMOKE OK — 전 단계 실행·유한 손실·교사 손실 감소 확인 (성능 판정 아님)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
