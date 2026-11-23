#!/usr/bin/env python3
"""
toy — 자기교정 디지털‑스튜디오 데이터 엔진 (#43)
=================================================
가설: 디지털 스튜디오가 합성 멀티태스크 데이터(포즈·뎁스·키포인트)를 생성할 때,
  - 태스크 간 **기하 일관성**(reprojection consistency)은 GT 없이 얻는 '공짜 관측'이고,
  - 소량 **실촬영**은 비싸지만 절대 보정 관측이다.
이 둘을 칼만 보정으로 묶으면(#42를 멀티태스크로 스케일) naive 합성·단순 실혼합보다
붕괴를 더 막고 실데이터를 훨씬 적게 쓴다.

모델(2D 오차공간):
  θ̂ = [포즈오차, 구조/스케일오차], 진값 θ*=0. 스튜디오는 세대마다 체계적 편향 b 를 더함(붕괴 원천).
  b = β_i·n̂ + β_g·ĝ 로 분해:
    n̂ = '비일관(reprojection을 깨는)' 방향 → 일관성 관측이 공짜로 잡음(대부분의 편향)
    ĝ = '게이지(일관되지만 틀린, 좌표/스케일 drift)' 방향 → 일관성은 못 봄, 실데이터만 보정
  → 핵심 통찰: 일관성이 오차 부분공간(n̂)을 공짜로 죽이므로, 비싼 실데이터는 '게이지' 한 축만
    고치면 됨 → 아주 조금으로 충분(gauge ambiguity = bundle adjustment의 그 구조).

4법: naive(둘 다 없음) / cons(일관성만) / mix(실데이터만) / ours(둘 다). numpy only, 300 시드.
"""
import numpy as np

s2 = np.sqrt(2.0)
NHAT = np.array([1.0, -1.0]) / s2     # 비일관 방향(일관성이 봄)
GHAT = np.array([1.0,  1.0]) / s2     # 게이지 방향(실데이터만 봄)
B_I, B_G = 0.060, 0.025               # 스튜디오 편향: 비일관 多 / 게이지 少
BIAS = B_I * NHAT + B_G * GHAT
Q = np.eye(2) * 4e-4                  # 공정노이즈
RC, RR = 0.01, 0.05                   # 일관성 관측노이즈(신뢰↑) / 실관측노이즈
T = 120


def run(method, seed, k_real=10):
    rng = np.random.default_rng(seed)
    th = np.zeros(2)
    P = np.eye(2) * 0.05
    errs, cons = [], []
    n_real_used = 0
    for t in range(T):
        th = th + BIAS                                   # 예측 = 합성 생성(편향 주입)
        P = P + Q
        if method in ("cons", "ours"):                   # 일관성 관측(공짜, 매 세대)
            h = NHAT
            s = h @ P @ h + RC
            K = (P @ h) / s                              # 2-vec
            z = 0.0 + rng.normal(0, np.sqrt(RC))         # 진값 n̂·θ*=0
            th = th + K * (z - h @ th)
            P = P - np.outer(K, h @ P)
        if method in ("mix", "ours") and (t % k_real == 0):   # 실 관측(비쌈, 희소)
            H = np.eye(2)
            S = P + np.eye(2) * RR
            K = P @ np.linalg.inv(S)
            z = rng.normal(0, np.sqrt(RR), 2)            # 진값 θ*=0 + 노이즈
            th = th + K @ (z - th)
            P = P - K @ P
            n_real_used += 1
        errs.append(np.linalg.norm(th))
        cons.append(abs(NHAT @ th))                      # reprojection 잔차 대용
    return np.array(errs), np.array(cons), n_real_used


def agg(method, k_real=10, S=300):
    E = np.array([run(method, s, k_real)[0] for s in range(S)])
    C = np.array([run(method, s, k_real)[1] for s in range(S)])
    nr = run(method, 0, k_real)[2]
    return E, C, nr


print("=" * 70)
print("#43 자기교정 디지털‑스튜디오 데이터 엔진  (θ*=0, 스튜디오 편향 b, T=120, 300시드)")
print(f"  편향 b: 비일관 {B_I}·n̂ + 게이지 {B_G}·ĝ  |  일관성R={RC} 실R={RR}")
print("=" * 70)

# ── Part 1: 동일 real 예산(10세대마다=12회)에서 최종 오차·일관성잔차 ──
print("\n── Part 1. 동일 real 예산(k_real=10 → 실관측 12회) 최종 ‖오차‖ / reprojection 잔차 ──")
print(f"{'method':<16}{'실관측수':>8}{'최종‖err‖평균':>14}{'최대':>9}{'붕괴율>1':>10}{'일관성잔차':>12}")
res = {}
for mth, name in [("naive","naive(합성만)"),("cons","일관성만"),("mix","실혼합만"),("ours","ours(둘다)")]:
    E, C, nr = agg(mth)
    fin = E[:, -1]
    res[mth] = fin.mean()
    print(f"{name:<16}{nr:>8}{fin.mean():>14.3f}{fin.max():>9.3f}{(fin>1).mean()*100:>9.0f}%{C[:,-1].mean():>12.3f}")
print("  → naive 발산(붕괴). 일관성만: n̂ 공짜로 죽이나 게이지 drift 잔존. 실혼합만: 2축 다 싸워 비쌈.")
print(f"  → ours 가 최저. (일관성만 대비 게이지까지, 실혼합만 대비 {res['mix']/max(res['ours'],1e-9):.1f}× 낮음)")

# ── Part 2: 붕괴 궤적(평균 ‖err‖) 몇 지점 ──
print("\n── Part 2. 붕괴 궤적 ‖err‖ (gen 10/40/80/119) ──")
for mth, name in [("naive","naive"),("cons","일관성만"),("mix","실혼합만"),("ours","ours")]:
    E,_,_ = agg(mth)
    m = E.mean(0)
    print(f"  {name:<10} " + "  ".join(f"g{g}:{m[g]:.2f}" for g in (10,40,80,119)))
print("  → naive 선형 발산 / 일관성만 게이지 축으로 완만 증가 / ours 낮게 수렴(유계=수축).")

# ── Part 3: 샘플효율 — 실관측을 얼마나 희소하게 해도 버티나 ──
print("\n── Part 3. 샘플효율: k_real(실관측 간격)↑ 일 때 최종 ‖err‖ (mix vs ours) ──")
print(f"{'k_real':>8}{'실관측수':>9}{'mix':>10}{'ours':>10}")
for k in [5, 10, 20, 40, 120]:
    Em,_,nr = agg("mix", k); Eo,_,_ = agg("ours", k)
    print(f"{k:>8}{nr:>9}{Em[:,-1].mean():>10.3f}{Eo[:,-1].mean():>10.3f}")
print("  → ours 는 실관측을 크게 줄여도(간격↑) 일관성이 n̂ 를 떠받쳐 오차가 완만. ")
print("    같은 오차를 mix 보다 훨씬 적은 실데이터로 달성 = 디지털스튜디오의 실이득.")

print("\n" + "=" * 70)
print("종합(#43): 일관성=공짜 관측(오차 부분공간 n̂ 소거) + 소량 실=게이지 보정 + 칼만게인 적응 + 수축(#39).")
print("  (1) naive 합성 붕괴, 단순 실혼합보다 ours 가 낮은 오차(동일 예산).")
print("  (2) 일관성 단독은 게이지 drift 못 막음 — 실 앵커가 여전히 필요(정직한 한계).")
print("  (3) 일관성이 대부분을 공짜로 처리 → 실데이터 예산을 크게 절감(샘플효율).")
print("  한계: 여기선 선형·일관성방향 기지 가정. 실제는 reprojection 비선형·방향 추정 필요(EKF/학습게인).")
