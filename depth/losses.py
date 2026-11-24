"""깊이 학습 손실 — Depth Anything V2(2406.09414) · Depth Pro(2410.02073) · MiDaS 계열.

모든 손실은 (pred, target, valid) 를 받는다. 모양 [B,H,W], valid 는 bool.
상대 깊이 단계는 '시차(역깊이)' 공간에서 스케일·시프트 불변으로 비교한다(DA2·Depth Pro).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _median_mad(x: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    """배치별 중앙값과 중앙값 대비 평균 절대편차(MAD). Depth Pro/MiDaS 정규화."""
    meds, mads = [], []
    for b in range(x.shape[0]):
        v = x[b][valid[b]]
        if v.numel() == 0:
            meds.append(x.new_tensor(0.0)); mads.append(x.new_tensor(1.0)); continue
        m = v.median()
        meds.append(m)
        mads.append((v - m).abs().mean().clamp_min(eps))
    return torch.stack(meds)[:, None, None], torch.stack(mads)[:, None, None]


def normalize_ssi(x, valid):
    m, s = _median_mad(x, valid)
    return (x - m) / s


def ssi_l1(pred, target, valid, trim: float = 0.0):
    """스케일·시프트 불변 L1(SSI-MAE). trim>0 이면 픽셀 손실 상위 trim 비율을 버린다
    (DA2: 가짜 라벨 상위 10% 무시 / Depth Pro: 실측 상위 20% 제외)."""
    p, t = normalize_ssi(pred, valid), normalize_ssi(target, valid)
    err = (p - t).abs()
    losses = []
    for b in range(err.shape[0]):
        e = err[b][valid[b]]
        if e.numel() == 0:
            continue
        if trim > 0:
            k = int(e.numel() * (1 - trim))
            e = torch.topk(e, max(k, 1), largest=False).values
        losses.append(e.mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0


def gradient_matching(pred, target, valid, scales: int = 4):
    """MiDaS/DA2 기울기 정합 손실 L_gm: 정규화된 차이의 x·y 기울기 L1, 4개 스케일."""
    p, t = normalize_ssi(pred, valid), normalize_ssi(target, valid)
    diff = (p - t) * valid
    m = valid.float()
    total, count = 0.0, 0
    for s in range(scales):
        if s > 0:
            diff = diff[:, ::2, ::2]
            m = m[:, ::2, ::2]
        gx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * (m[:, :, 1:] * m[:, :, :-1])
        gy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * (m[:, 1:, :] * m[:, :-1, :])
        n = m.sum().clamp_min(1.0)
        total = total + (gx.sum() + gy.sum()) / n
        count += 1
    return total / count


_SCHARR_X = torch.tensor([[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]]) / 32.0
_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def multiscale_derivative(pred, target, valid, op: str = "scharr", p: int = 1, scales: int = 6, ssi: bool = False):
    """Depth Pro 식: L_{*,p,M} = (1/M) Σ_j (1/N_j) Σ_i |∇C − ∇Ĉ|^p, 블러+다운샘플한 M개 스케일.
    MAGE = (scharr,p=1), MALE = (laplace,p=1), MSGE = (scharr,p=2). ssi=True 면 SSI 정규화 후(SSI-MAGE)."""
    if ssi:
        pred, target = normalize_ssi(pred, valid), normalize_ssi(target, valid)
    x = torch.stack([pred, target], 1) * valid[:, None]
    m = valid[:, None].float()
    if op == "scharr":
        kx = _SCHARR_X.to(x)
        kernels = torch.stack([kx, kx.t()])[:, None]          # [2,1,3,3]
    else:
        kernels = _LAPLACE.to(x)[None, None]
    total = 0.0
    for j in range(scales):
        if j > 0:
            x = F.avg_pool2d(x, 2) if min(x.shape[-2:]) >= 6 else x
            m = (F.avg_pool2d(m, 2) > 0.999).float() if min(m.shape[-2:]) >= 6 else m
        B = x.shape[0]
        g = F.conv2d(x.reshape(B * 2, 1, *x.shape[-2:]), kernels, padding=1).reshape(B, 2, -1, *x.shape[-2:])
        d = (g[:, 0] - g[:, 1]).abs().pow(p).sum(1)          # [B,H,W]
        mm = F.max_pool2d(1 - m, 3, 1, 1)[:, 0] < 0.5         # 경계 픽셀 제외(커널이 무효 영역에 걸침)
        total = total + (d * mm).sum() / mm.sum().clamp_min(1.0)
    return total / scales


def mae(pred, target, valid):
    return (pred - target).abs()[valid].mean()


def mse(pred, target, valid):
    return (pred - target).pow(2)[valid].mean()


def silog(pred_depth, target_depth, valid, lam: float = 0.85, eps: float = 1e-6):
    """미터 단계용 스케일 불변 로그 손실(Eigen). pred_depth>0."""
    d = torch.log(pred_depth.clamp_min(eps)) - torch.log(target_depth.clamp_min(eps))
    d = d[valid]
    return torch.sqrt((d ** 2).mean() - lam * d.mean() ** 2 + 1e-8)


def feature_alignment(student_feat, frozen_feat, margin: float = 0.85):
    """DA1/DA2 특징 정렬: 동결 DINOv2 특징과 코사인 유사도를 높이되, 이미 margin 이상인 위치는 무시.
    feat: [B,C,h,w]."""
    cos = F.cosine_similarity(student_feat, frozen_feat.detach(), dim=1)
    mask = cos < margin
    return (1 - cos)[mask].mean() if mask.any() else cos.sum() * 0


# 단계별 손실 묶음 ------------------------------------------------------------
def teacher_loss(pred, target, valid, gm_weight: float = 2.0):
    """DA2 교사(합성): L_ssi + 2·L_gm (원문 1:2)."""
    return ssi_l1(pred, target, valid) + gm_weight * gradient_matching(pred, target, valid)


def student_loss(pred, pseudo, valid, trim: float = 0.1):
    """DA2 학생(가짜 라벨 실사진): L_ssi(상위 10% 무시). gm 은 가짜 라벨에선 도움 안 됨(원문)."""
    return ssi_l1(pred, pseudo, valid, trim=trim)


def sharpen_loss(pred, target, valid):
    """Depth Pro 2단계(합성만): MAE + MSE + MAGE + MALE + MSGE. (원문에 가중 미기재 → 동일 가중)"""
    return (mae(pred, target, valid) + mse(pred, target, valid)
            + multiscale_derivative(pred, target, valid, "scharr", 1)
            + multiscale_derivative(pred, target, valid, "laplace", 1)
            + multiscale_derivative(pred, target, valid, "scharr", 2))
