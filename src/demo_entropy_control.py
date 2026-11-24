#!/usr/bin/env python3
"""
메커니즘 점검(성능 비교 아님): MAI 적응 엔트로피 제어가 논문 설명대로 작동하는가.

무대: 표 형태 정책 θ[프롬프트, 위치, 토큰] (P=24, T=5, V=4). 롤아웃 1회당 5 그래디언트 스텝(오프폴리시),
     top-p(0.97) 마스크 재생, 외부 클립, 토큰레벨 정규화 — 전부 mai_grpo 함수 그대로.
비교: fixed = 고정 대칭 클립 ε=0.2 (표준 GRPO 형태) / adaptive = MAI(ε=0.6, k 적분제어, H*=0.3).
Part 1: 전체 레시피(통과율 필터 켬), 위치당 정답 1개, 보상=정답 3개 이상이면 1.
Part 2: 제어기 단독(필터 끔), 위치당 유효 답 2개, 보상=유효 토큰 비율 → '한 답으로 쏠리는 붕괴'를 막는가.
이 데모는 어느 방법이 더 좋은지 판정하지 않는다(판정은 PREREG.md 의 튜닝된 기준선·동일 계산량 실험).
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mai_grpo as M  # noqa: E402

warnings.filterwarnings("ignore")
P, T, V, NEED = 24, 5, 4, 3
PROMPTS_PER_ITER, STEPS_PER_ROLLOUT, LR = 8, 5, 0.5


def masked_logsoftmax_all(theta, top_p):
    m = M.topp_mask(theta, top_p)                       # [..., V]
    return M.masked_log_softmax(theta, m), m


def run(mode, seed, multi, filters, iters):
    rng = np.random.default_rng(seed)
    correct = rng.integers(0, V, size=(P, T))
    alt = (correct + 1 + rng.integers(0, V - 1, size=(P, T))) % V
    theta = np.zeros((P, T, V))
    cfg = M.MAIConfig(eps=0.2, k_max=0.0, delta=0.0) if mode == "fixed" else M.MAIConfig()
    cfg.G, cfg.G_early = 32, 8
    k, log = 0.0, []
    for it in range(iters):
        ps = rng.permutation(P)[:PROMPTS_PER_ITER]
        lsm, msk = masked_logsoftmax_all(theta[ps], cfg.top_p)          # [8,T,V]
        cdf = np.cumsum(np.exp(lsm), -1)
        u = rng.random((len(ps), cfg.G, T, 1))
        toks = (u > cdf[:, None]).sum(-1).clip(0, V - 1)                 # [8,G,T]
        if multi:
            ok = (toks == correct[ps][:, None]) | (toks == alt[ps][:, None])
            R = ok.mean(-1)
        else:
            R = ((toks == correct[ps][:, None]).sum(-1) >= NEED).astype(float)
        keep = np.ones(len(ps), bool)
        if filters:
            pe = (R[:, :cfg.G_early] > 0).mean(1); pf = (R > 0).mean(1)
            keep = (pe >= cfg.early_range[0]) & (pe <= cfg.early_range[1]) \
                 & (pf >= cfg.full_range[0]) & (pf <= cfg.full_range[1])
        if not keep.any():
            log.append((it, np.nan, np.nan, k)); continue
        A = np.stack([M.group_advantages(R[i]) for i in range(len(ps))])  # [8,G]
        sel = np.where(keep)[0]
        pi = np.repeat(ps[sel], cfg.G * T)
        ti = np.tile(np.arange(T), len(sel) * cfg.G)
        ai = toks[sel].reshape(-1)
        adv = np.repeat(A[sel].reshape(-1), T)
        lp_old = lsm[np.repeat(sel, cfg.G * T), ti, ai]
        mask_tok = msk[np.repeat(sel, cfg.G * T), ti]                     # 롤아웃 때 nucleus
        valid = np.ones_like(adv)
        for _ in range(STEPS_PER_ROLLOUT):
            lsn = M.masked_log_softmax(theta[pi, ti], mask_tok)           # 마스크 재생
            lp_new = lsn[np.arange(len(ai)), ai]
            _, g, _ = M.grpo_loss(lp_new, lp_old, adv, valid, k, cfg)
            pr = np.exp(lsn)
            onehot = np.zeros_like(pr); onehot[np.arange(len(ai)), ai] = 1.0
            grad = np.zeros_like(theta)
            np.add.at(grad, (pi, ti), g[:, None] * (onehot - pr))
            theta -= LR * grad * len(ai) / 50.0
            if mode == "adaptive":
                k = M.update_k(k, M.entropy_estimate(lp_new, lp_old, valid), cfg)
        pr_all = M.softmax(theta)
        H = float((-(pr_all * np.log(pr_all + 1e-30)).sum(-1)).mean())
        if multi:
            pv = np.take_along_axis(pr_all, correct[..., None], -1)[..., 0]
            pa = np.take_along_axis(pr_all, alt[..., None], -1)[..., 0]
            met = float(((pv > 0.1) & (pa > 0.1)).mean())
        else:
            met = float((pr_all.argmax(-1) == correct).mean())
        log.append((it, H, met, k))
    return log


def report(title, mode, metric, **kw):
    L = [run(mode, s, **kw) for s in range(5)]
    H = np.array([[x[1] for x in l] for l in L]); A = np.array([[x[2] for x in l] for l in L])
    K = np.array([[x[3] for x in l] for l in L]); n = H.shape[1]
    print(f"\n[{title} | {mode}] 시드5 평균 · 학습할 문제 0개인 반복 {np.isnan(H).mean():.0%}")
    for it in sorted({0, n // 6, n // 3, n // 2, 2 * n // 3, 5 * n // 6, n - 1}):
        print(f"  iter {it+1:>3}: H={np.nanmean(H[:, it]):.3f}  {metric}={np.nanmean(A[:, it]):.3f}  k={np.nanmean(K[:, it]):.2f}")
    return H, A


if __name__ == "__main__":
    print("=== Part 1. 전체 레시피(필터 켬), 위치당 정답 1개 ===")
    for m in ["fixed", "adaptive"]:
        report("단일정답+필터", m, "acc", multi=False, filters=True, iters=150)
    print("\n=== Part 2. 제어기 단독(필터 끔), 위치당 유효 답 2개 ===")
    out = {m: report("다중정답·필터끔", m, "둘다유지", multi=True, filters=False, iters=300) for m in ["fixed", "adaptive"]}
    print("\nH*=0.3(adaptive). 유효 답 2개를 고르게 유지하면 H≈ln2=0.69, 한쪽 붕괴면 H→0.")
    for m, (H, A) in out.items():
        print(f"  {m:<9} 최종 H(시드별) {np.round(H[:, -1], 3)}  둘다유지(시드별) {np.round(A[:, -1], 3)}")
    print("메커니즘 점검 — 성능 우열 판정 아님.")
