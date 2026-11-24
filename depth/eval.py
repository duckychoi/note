"""제로샷 평가 — AbsRel, δ1, (선택) 경계 F1.

    python depth/eval.py --config depth/configs/tierA.json --ckpt runs/depth/student/final.pt \
        --mode relative --data /data/eval/nyu --out runs/depth/eval_nyu.json [--max_depth 10]

relative: 예측 시차를 GT 시차에 최소제곱 (스케일, 시프트) 정렬한 뒤 깊이로 바꿔 평가 (DA2 관행).
metric  : 예측 정준 깊이 × f/f_c (intrinsics.json 필요).
경계 F1: Depth Pro(2410.02073) 아이디어 — 이웃 픽셀 깊이비가 (1+t) 초과면 가림 경계, t=5..25%,
         높은 t 에 선형 가중 평균. (원문 정의를 단순화한 구현: 오른쪽·아래 이웃만 사용)
양성 대조: 공개 DA2-Small 가중치로 논문 표 값을 재현하는지 먼저 확인할 것(REPRO_PLAN D).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
from pseudo_label import build_model  # noqa: E402


def align_scale_shift(pred, gt, valid):
    """min_{s,t} Σ (s·pred + t − gt)^2 on valid."""
    p, g = pred[valid], gt[valid]
    A = np.stack([p, np.ones_like(p)], 1)
    (s, t), *_ = np.linalg.lstsq(A, g, rcond=None)
    return s * pred + t


def depth_metrics(pred_d, gt_d, valid):
    p, g = pred_d[valid], gt_d[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    return {"absrel": absrel, "delta1": float(np.mean(ratio < 1.25))}


def occlusion_edges(d, valid, t):
    e = np.zeros_like(valid)
    r = d[:, 1:] / np.maximum(d[:, :-1], 1e-6)
    vx = valid[:, 1:] & valid[:, :-1]
    e[:, :-1] |= vx & ((r > 1 + t) | (r < 1 / (1 + t)))
    r = d[1:, :] / np.maximum(d[:-1, :], 1e-6)
    vy = valid[1:, :] & valid[:-1, :]
    e[:-1, :] |= vy & ((r > 1 + t) | (r < 1 / (1 + t)))
    return e


def _dilate(e, r):
    if r <= 0:
        return e
    out = e.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(e, dy, 0), dx, 1)
    return out


def boundary_f1(pred_d, gt_d, valid, ts=(0.05, 0.10, 0.15, 0.20, 0.25), tol=0):
    """tol=0: 같은 픽셀만 일치(원 정의, 가혹). tol=r: r픽셀 이내 경계를 일치로 인정(우리 추가 옵션, 별도 보고)."""
    f1s, ws = [], []
    for t in ts:
        pe, ge = occlusion_edges(pred_d, valid, t), occlusion_edges(gt_d, valid, t)
        prec = (pe & _dilate(ge, tol)).sum() / max(pe.sum(), 1)
        rec = (ge & _dilate(pe, tol)).sum() / max(ge.sum(), 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)); ws.append(t)
    return float(np.average(f1s, weights=ws))


@torch.no_grad()
def evaluate(cfg, ckpt, data_root, mode, max_depth=None, boundary=False, device="cpu", stage_key="teacher"):
    model = build_model(cfg, ckpt, torch.device(device))
    ds = D.LabeledDepth(data_root, cfg["size"], train=False, max_depth=max_depth)
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        x = it["image"][None].to(device)
        y = (model(x) + model(x.flip(-1)).flip(-1))[0].float().cpu().numpy() / 2
        gt, valid = it["depth"].numpy(), it["valid"].numpy()
        if valid.sum() < 10:
            continue
        if mode == "relative":
            disp = align_scale_shift(y, 1.0 / np.maximum(gt, 1e-6), valid)
            pred_d = 1.0 / np.clip(disp, 1e-3, None)
        else:
            fx = float(it["fx"])
            pred_d = y * fx / cfg.get("f_c", 300.0)
        if max_depth:
            pred_d = np.clip(pred_d, 1e-3, max_depth)
        m = depth_metrics(pred_d, gt, valid)
        if boundary:
            m["boundary_f1"] = boundary_f1(pred_d, gt, valid)
            m["boundary_f1_tol2"] = boundary_f1(pred_d, gt, valid, tol=2)
        m["stem"] = it["stem"]
        rows.append(m)
    keys = [k for k in rows[0] if k != "stem"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return {"summary": summary, "n": len(rows), "per_image": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage_cfg", default="teacher", help="모델 크기 정보를 읽을 설정 섹션")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["relative", "metric"], default="relative")
    ap.add_argument("--max_depth", type=float, default=None)
    ap.add_argument("--boundary", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    full = json.loads(Path(a.config).read_text())
    cfg = {**full.get("common", {}), **full[a.stage_cfg]}
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    res = evaluate(cfg, a.ckpt, a.data, a.mode, a.max_depth, a.boundary, dev)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"]))
    return res


if __name__ == "__main__":
    main()
