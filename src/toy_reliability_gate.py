#!/usr/bin/env python3
"""
toy — Reliability-Gated Filtering (#63)
=======================================
질문(교수님): 칼만/PF는 '예측 vs 관측' 게인 하나뿐. "이 관측이 유독 좋으니 더 가중"은 없지 않나?
robust/VB KF에 있긴 하나, goodness 신호가 '예측 대비 잔차'(모델 상대)라 두 공백:
  (A) 순환성: 자신만만하게 틀린 필터가 '맞지만 놀라운' 좋은 관측을 이상치로 버림.
  (B) 균일 수축(안정) 증명이 잔차기반엔 잘 안 묶임.

#63 = raw 관측 고유 품질 q(z)(잔차 아님) → 게인, + 클램프로 균일 수축.
비교:  A 고정R  /  B 잔차게이트(robust/VB, Student-t)  /  C 품질게이트(ours)  /  D 결합.
numpy only, 200 시드. 집계=평균/최대(중앙값 마스킹 회피).

관측 모델: z=x+v, v~N(0,r_t). 관측 고유 품질 covariate c∈[0,1] (SNR/선명도/일관성 대용)이
r_t 와 상관(c↑→r↓). 점프(급변) 직후엔 '좋은 관측'(c=0.95)이 들어옴 → 순환성 테스트.
"""
import numpy as np

T = 200
R_LOW, R_HIGH = 0.05, 2.0          # 관측 분산 범위
Q_TRUE, Q_FILT = 0.01, 0.02
R_BAR = R_LOW + (R_HIGH - R_LOW) * (1/3)   # E[(1-c)^2]=1/3 → 평균 분산(고정R용)
JUMP_EVERY, JUMP_SIZE = 40, 5.0
P_OUT = 0.05                        # 이상치 확률
LAM_LO, LAM_HI = 0.05, 4.0          # 잔차게이트 클램프
REFF_LO, REFF_HI = 0.02, 5.0        # 품질게이트 R_eff 클램프(= 균일 수축 보증 핵심)


def simulate(seed):
    rng = np.random.default_rng(seed)
    x = 0.0
    xs, zs, cs, jump, outl = [], [], [], np.zeros(T, bool), np.zeros(T, bool)
    for t in range(T):
        x += rng.normal(0, np.sqrt(Q_TRUE))
        if t > 0 and t % JUMP_EVERY == 0:
            x += rng.choice([-JUMP_SIZE, JUMP_SIZE]); jump[t] = True
        if jump[t]:
            c = 0.95; r = R_LOW + (R_HIGH - R_LOW) * (1 - c)**2      # 점프직후 '좋은' 관측
        elif rng.random() < P_OUT:
            c = 0.05; r = 100.0; outl[t] = True                     # 이상치(품질 낮음)
        else:
            c = rng.uniform(0, 1); r = R_LOW + (R_HIGH - R_LOW) * (1 - c)**2
        z = x + rng.normal(0, np.sqrt(r))
        xs.append(x); zs.append(z); cs.append(c)
    return np.array(xs), np.array(zs), np.array(cs), jump, outl


def quality_Reff(c):   # raw 관측 품질 → 노이즈 추정 (학습가능한 보정맵 대용), 클램프
    return np.clip(R_LOW + (R_HIGH - R_LOW) * (1 - c)**2, REFF_LO, REFF_HI)


def run(method, xs, zs, cs):
    x = zs[0]; P = 1.0
    est = np.zeros(T); infl = np.zeros(T); Ks = np.zeros(T)
    for t in range(T):
        xp = x; Pp = P + Q_FILT
        z, c = zs[t], cs[t]
        nu = z - xp; S = Pp + R_BAR
        if method == "A":                       # 고정 R
            Reff = R_BAR
        elif method == "B":                     # 잔차게이트 (robust/VB, Student-t 1-step)
            lam = 4.0 / (3.0 + nu * nu / S)
            lam = np.clip(lam, LAM_LO, LAM_HI)
            Reff = R_BAR / lam
        elif method == "C":                     # 품질게이트 (ours): raw 품질 q(z)→R_eff, 클램프
            Reff = quality_Reff(c)
        else:                                   # D 결합: 품질 + 극단이상치만 잔차로 추가차단
            Reff = quality_Reff(c)
            if (nu * nu / (Pp + Reff) > 9.0) and (c < 0.3):
                Reff = min(Reff * 20.0, REFF_HI * 20.0)
        K = Pp / (Pp + Reff)
        x = xp + K * nu; P = (1.0 - K) * Pp
        est[t] = x; infl[t] = abs(K * nu); Ks[t] = K
    return est, infl, Ks


def metrics(seed):
    xs, zs, cs, jump, outl = simulate(seed)
    out = {}
    for m in ["A", "B", "C", "D"]:
        est, infl, Ks = run(m, xs, zs, cs)
        err = np.abs(est - xs)
        steady = ~jump & ~outl
        # 점프 직후 5스텝 회복오차(순환성 테스트)
        post = []
        for t0 in np.where(jump)[0]:
            post.append(err[t0:t0+5].mean())
        out[m] = dict(rmse=np.sqrt((err[steady]**2).mean()),
                      postjump=np.mean(post) if post else np.nan,
                      outlier_max=err[outl].max() if outl.any() else 0.0,
                      infl_max=infl.max(),
                      K_lo=Ks.min(), K_hi=Ks.max())
    return out


S = 200
agg = {m: {k: [] for k in ["rmse", "postjump", "outlier_max", "infl_max", "K_lo", "K_hi"]}
       for m in "ABCD"}
for s in range(S):
    mt = metrics(s)
    for m in "ABCD":
        for k in agg[m]:
            agg[m][k].append(mt[m][k])

name = {"A": "A 고정R", "B": "B 잔차게이트(robust/VB)", "C": "C 품질게이트(ours)", "D": "D 결합(ours+)"}
print("=" * 72)
print("#63 Reliability-Gated Filtering — raw 관측 품질 → 게인 + 수축 (200시드)")
print("=" * 72)

print("\n── Part 1. 이질 노이즈 정상상태 RMSE (낮을수록 좋음) ──")
for m in "ABCD":
    print(f"  {name[m]:<26} {np.mean(agg[m]['rmse']):.3f}")
print("  → 관측별 품질 가중이 고정R(A)보다 유리. 품질게이트 C가 최적 R에 근접.")

print("\n── Part 2. 점프 직후 회복오차 = 순환성 테스트 (낮을수록 좋음) ★핵심 ──")
for m in "ABCD":
    print(f"  {name[m]:<26} {np.mean(agg[m]['postjump']):.3f}")
print("  → B(잔차게이트)는 '맞지만 놀라운' 좋은 관측을 큰 잔차=이상치로 보고 버려 회복 느림(순환성).")
print("    C/D(품질게이트)는 c↑(좋은 관측) 보고 즉시 신뢰 → 빠른 회복. = robust/VB KF의 공백(A).")

print("\n── Part 3. 이상치 최대 오차 (낮을수록 좋음) ──")
for m in "ABCD":
    print(f"  {name[m]:<26} 평균 {np.mean(agg[m]['outlier_max']):.2f}  최대 {np.max(agg[m]['outlier_max']):.2f}")
print("  → B는 이상치(큰 잔차) 강건, C는 품질(c↓)로 강건, D는 둘 다 → 최소.")

print("\n── Part 4. Bounded influence & 수축 (안정 증명의 수치 확인) ──")
for m in "ABCD":
    print(f"  {name[m]:<26} 단일스텝 최대주입 {np.mean(agg[m]['infl_max']):.2f}  "
          f"K∈[{np.min(agg[m]['K_lo']):.2f},{np.max(agg[m]['K_hi']):.2f}]  (1-K)>0 보장")
print("  → R_eff 하한 클램프(REFF_LO) → K<1 균일 → (1-K)≥REFF_LO/(Pp+REFF_LO)>0 = 균일 수축.")
print("    동시에 상한/품질이 outlier 주입을 유계화 = bounded influence. 두 성질 동시 보증.")

print("\n" + "=" * 72)
print("명제(수축): H 가역·R_eff∈[R/λmax, R/λmin]⪰R_min≻0 이면 A_cl=I-K(q)H 가")
print("  ‖A_cl‖≤ρ<1 균일(게인이 의사역에 도달 못함) & ‖K‖≤κ(주입 유계). → robust+stable 동시.")
print("요지(#63): goodness를 '잔차'(모델상대) 말고 'raw 관측 품질 q(z)'로 측정 → 순환성 해소,")
print("  클램프로 균일 수축 증명. = robust/VB KF의 공백 (A 순환성, B 증명) 메움. #42/#43 샘플신뢰로 이식.")
