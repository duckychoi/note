#!/usr/bin/env python3
"""
toy — Filtering-based synthetic-data robust training (#42)
==========================================================
아이디어: '합성데이터 재귀학습'을 칼만필터로 본다.
  - 합성 생성  = 노이즈 낀 '예측'(모델이 자기 믿음에서 샘플) → 공분산 P 가 Q 만큼 증가
  - 실데이터   = 보정 '관측' z (근접논문 2512.00757의 '오라클 라벨'을 실관측으로 대체)
  - 칼만 게인 K = P/(P+R) = '적응적 신뢰'(모델 불확실하면 실데이터 더 믿음)
  - 수축인자 (1-K) ∈ (0,1) = 소산성(#39) → 오차 유계 = 붕괴 없음 증명
  - 적응 KF   = 큰 혁신(innovation) 감지 시 Q 팽창 → 게인 급상승 → 분포이동 빠른 회복

비교(동일 real 예산 n_real):
  naive  : 자기생성만 재귀학습 (앵커 없음)        → 평균 랜덤워크 + 분산붕괴
  mix    : 합성+실데이터 고정비율 혼합 (가장 강한 기존 완화책)
  kf     : 예측–보정 (최적 고정게인)
  kf_ad  : 예측–보정 + 혁신기반 적응게인

numpy only. 다중 시드 집계(평균/최대/붕괴율) — 중앙값 마스킹 회피.
"""
import numpy as np

THETA0, SIG_D = 2.0, 1.0          # 참 평균 / 실데이터 표준편차
T, N_REAL, M = 120, 5, 50         # 세대수 / 세대당 실샘플 / 합성배치
Q = SIG_D**2 / M                  # 공정노이즈 = 한 세대 생성이 더하는 drift 분산(≈0.02)
R = SIG_D**2 / N_REAL             # 관측노이즈 = 실 n_real 평균의 분산(=0.2)


def run(method, seed, shock_gen=None, shock=3.0):
    rng = np.random.default_rng(seed)
    real0 = rng.normal(THETA0, SIG_D, N_REAL)
    theta = real0.mean()           # 평균 믿음
    var_est = real0.var()          # 분산 믿음(다양성)
    P = R                          # 믿음 공분산(KF용)
    errs, vars_, Ks = [], [], []
    for t in range(T):
        tgt = THETA0 + (shock if (shock_gen is not None and t >= shock_gen) else 0.0)
        synth = rng.normal(theta, np.sqrt(max(var_est, 1e-9)), M)   # 예측=자기생성
        if method == "naive":
            theta = synth.mean()
            var_est = synth.var()                      # ÷M ML추정 → (M-1)/M 배씩 수축(붕괴)
            K = 0.0
        elif method == "mix":
            real = rng.normal(tgt, SIG_D, N_REAL)
            data = np.concatenate([synth, real])
            theta = data.mean()
            var_est = data.var()
            K = N_REAL / (N_REAL + M)                  # 고정 비율 신뢰
        else:  # kf / kf_ad : 예측–보정
            P_pred = P + Q
            real = rng.normal(tgt, SIG_D, N_REAL)
            z = real.mean()
            innov = z - theta
            if method == "kf_ad":                      # 혁신기반 적응: 큰 innov → Q 팽창
                ratio = innov**2 / (P_pred + R)
                P_pred *= 1.0 + 8.0 * max(0.0, ratio - 1.0)
            K = P_pred / (P_pred + R)
            theta = theta + K * innov                  # 보정
            P = (1.0 - K) * P_pred                     # 수축: (1-K)<1
            s2 = real.var(ddof=1) if N_REAL > 1 else SIG_D**2
            var_est = 0.5 * var_est + 0.5 * s2          # 분산도 실관측으로 앵커
        errs.append(theta - tgt)
        vars_.append(var_est)
        Ks.append(K)
    return np.array(errs), np.array(vars_), np.array(Ks)


def agg(method, S=300, **kw):
    E = np.array([run(method, s, **kw)[0] for s in range(S)])
    V = np.array([run(method, s, **kw)[1] for s in range(S)])
    return E, V


print("="*66)
print("#42 필터링-기반 합성데이터 강건학습  (θ*=2.0, σ_d=1, n_real=5, m=50, T=120)")
print("="*66)

# ---------- Part 1: 평균 붕괴 (충격 없음) ----------
print("\n── Part 1. 평균 붕괴 (동일 real 예산, 300 시드) ──")
print(f"{'method':<8}{'final|err| 평균':>16}{'최대':>10}{'붕괴율(|err|>1)':>16}")
for mth, name in [("naive","naive"),("mix","mix-real"),("kf","KF"),("kf_ad","adaptive-KF")]:
    E, _ = agg(mth)
    fin = np.abs(E[:, -1])
    print(f"{name:<8}{fin.mean():>16.3f}{fin.max():>10.3f}{(fin>1).mean()*100:>14.1f}%")
print("  → naive 는 앵커가 없어 랜덤워크로 발산. mix<KF 순으로 안정(다음 줄 수축인자 참고).")

# 수축인자(소산성 #39) — 오차 점화식 e_{t+1}=(1-K)e_t+noise, (1-K)<1 이어야 유계
print("\n── 수축인자 (1-K) : <1 이어야 오차 유계 = 붕괴 없음 보증 (#39 소산성) ──")
for mth, name in [("naive","naive"),("mix","mix-real"),("kf","KF")]:
    _, _ = None, None
    Kbar = np.mean([run(mth, s)[2][-20:].mean() for s in range(100)])
    print(f"  {name:<10} 1-K ≈ {1-Kbar:.3f}" + ("  (=1 → 수축 없음 → 발산)" if mth=="naive" else ""))

# ---------- Part 2: 분산(다양성) 붕괴 ----------
print("\n── Part 2. 분산 붕괴 = 다양성/꼬리 손실 (최종 분산/σ_d²) ──")
for mth, name in [("naive","naive"),("mix","mix-real"),("kf","KF")]:
    _, V = agg(mth)
    print(f"  {name:<10} {V[:, -1].mean()/SIG_D**2:>6.2f}  "
          + ("← 붕괴(꼬리 소멸)" if mth=="naive" else "← 보존"))
print("  → naive 는 유한표본 자기학습으로 분산이 (M-1)/M 배씩 수축→0. 실관측 보정이 막음.")

# ---------- Part 3: 분포 이동(충격) 회복 — 적응의 이득 ----------
print("\n── Part 3. 분포이동 회복 (gen 50에서 +3.0 shift, 동일 real 예산) ──")
SG = 50
print(f"{'method':<12}{'회복세대수(|err|<0.3)':>22}{'충격후 누적오차':>16}")
for mth, name in [("mix","mix-real"),("kf","KF"),("kf_ad","adaptive-KF")]:
    E, _ = agg(mth, shock_gen=SG)
    post = np.abs(E[:, SG:])
    # 회복 세대수: 평균궤적이 0.3 아래로 처음 내려가는 지점
    m_traj = post.mean(0)
    below = np.where(m_traj < 0.3)[0]
    rec = int(below[0]) if below.size else T - SG
    cum = post.mean(0).sum()
    print(f"{name:<12}{rec:>20}{cum:>16.2f}")
print("  → 고정비율 mix 는 느린 복귀(고정 수축율). KF 적응게인은 혁신 감지→게인↑→빠른 복귀.")

print("\n" + "="*66)
print("종합: '합성=예측 / 실데이터=관측 / 칼만게인=적응신뢰 / (1-K)=수축(#39)'")
print("  (1) 동일 real 예산에서 KF 가 mix 보다 낮은 정상상태 오차 (최적게인).")
print("  (2) 분산붕괴를 실관측 보정이 차단 (다양성 보존).")
print("  (3) 적응게인이 분포이동에서 빠른 회복 — 고정혼합이 못 하는 이득.")
print("  정직한 한계: R 추정·근사 선형가우시안 가정. 딥넷 확장은 KalmanNet식")
print("  학습게인 + 대각 공분산 필요. 'mix-real'은 더 단순 — 우리 이득은 적응·보증·불확실성.")
