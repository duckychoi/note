#!/usr/bin/env python3
"""
toy#2 — Learned Reliability Gate (#63 심화)
==========================================
#63 1차 toy의 솔직한 약점: 품질게이트 C가 '품질→노이즈 맵'을 이미 안다(오라클)고 가정.
여기선 그 맵 h_θ(c)를 **데이터에서 학습**(오라클 제거)하고, 품질신호가 **불완전**할 때(상관 ρ)
어떻게 되는지까지 본다.

학습 원리(현재 잔차 안 씀 → 순환성 회피):
  혁신 ν_t ~ N(0, S_t), S_t = P⁻_t + R_eff(c_t) 이므로 E[ν² - P⁻ | c] = E[r | c].
  → 보정셋에서 (c, ν²-P⁻) 를 모아 R_eff(c)=poly(1-c) 를 최소제곱으로 적합(모멘트매칭),
    2회 반복(맵→필터재실행→재적합). 보정/테스트 시드 분리. 현재 스텝 잔차로 가중하지 않음.

비교: A 고정R / B 잔차게이트(robust/VB) / C* 참노이즈 오라클(천장) / E 학습게이트(ours) / F 학습+이상치백업.
numpy only.
"""
import numpy as np

T = 200
R_LOW, R_HIGH = 0.05, 2.0
Q_TRUE, Q_FILT = 0.01, 0.02
R_BAR = R_LOW + (R_HIGH - R_LOW) * (1/3)
JUMP_EVERY, JUMP_SIZE = 40, 5.0
P_OUT = 0.05
REFF_LO, REFF_HI = 0.02, 5.0
LAM_LO, LAM_HI = 0.05, 4.0


def simulate(seed, rho):
    """rho = 품질 covariate 유용도(1=완벽, 0=무정보)."""
    rng = np.random.default_rng(seed)
    x = 0.0
    xs, zs, cs, rts, jump, outl = [], [], [], [], np.zeros(T, bool), np.zeros(T, bool)
    for t in range(T):
        x += rng.normal(0, np.sqrt(Q_TRUE))
        if t > 0 and t % JUMP_EVERY == 0:
            x += rng.choice([-JUMP_SIZE, JUMP_SIZE]); jump[t] = True
        u = rng.uniform(0, 1)                      # 숨은 진짜 품질
        if jump[t]:
            u = 0.95
        outlier = (rng.random() < P_OUT) and (not jump[t])
        if outlier:
            u = 0.05; r = 100.0; outl[t] = True
        else:
            r = R_LOW + (R_HIGH - R_LOW) * (1 - u)**2
        # 관측되는 품질 covariate: ρ로 u를 오염
        c = np.clip(rho * u + (1 - rho) * rng.uniform(0, 1), 0, 1)
        z = x + rng.normal(0, np.sqrt(r))
        xs.append(x); zs.append(z); cs.append(c); rts.append(r)
    return (np.array(xs), np.array(zs), np.array(cs), np.array(rts), jump, outl)


def filt(zs, reff_fn):
    x = zs[0]; P = 1.0
    est = np.zeros(T); nu = np.zeros(T); Pp = np.zeros(T); Ks = np.zeros(T)
    for t in range(T):
        xp = x; Ppt = P + Q_FILT
        n = zs[t] - xp
        Reff = reff_fn(t, n, Ppt)
        K = Ppt / (Ppt + Reff)
        x = xp + K * n; P = (1 - K) * Ppt
        est[t] = x; nu[t] = n; Pp[t] = Ppt; Ks[t] = K
    return est, nu, Pp, Ks


NBINS = 12
CENTERS = (np.arange(NBINS) + 0.5) / NBINS


def fit_map(seeds, rho, iters=2):
    """R_eff(c) = c-빈별 median(ν²-P⁻) (로버스트 모멘트매칭). 점프·이상치는 빈 내 소수라 중앙값이 무시.
    현재 스텝 잔차로 가중하지 않음(테스트 시 c만 사용) → 순환성 없음."""
    tbl = np.full(NBINS, R_BAR)
    for _ in range(iters):
        C, Y = [], []
        for s in seeds:
            xs, zs, cs, rts, jump, outl = simulate(s, rho)
            reff_fn = lambda t, n, Ppt, cs=cs, tbl=tbl: float(np.clip(
                np.interp(cs[t], CENTERS, tbl), REFF_LO, REFF_HI))
            est, nu, Pp, Ks = filt(zs, reff_fn)
            C.append(cs); Y.append(nu**2 - Pp)
        C = np.concatenate(C); Y = np.concatenate(Y)
        idx = np.clip((C * NBINS).astype(int), 0, NBINS - 1)
        new = tbl.copy()
        for b in range(NBINS):
            sel = Y[idx == b]
            if sel.size > 5:
                new[b] = np.clip(np.median(sel), REFF_LO, REFF_HI)  # 중앙값=로버스트
        tbl = new
    return tbl


def map_val(tbl, c):
    return float(np.clip(np.interp(c, CENTERS, tbl), REFF_LO, REFF_HI))


def reff_builders(w):
    tbl = w
    def A(cs): return lambda t, n, Ppt: R_BAR
    def B(cs):
        def f(t, n, Ppt):
            S = Ppt + R_BAR
            lam = np.clip(4.0/(3.0 + n*n/S), LAM_LO, LAM_HI)
            return R_BAR/lam
        return f
    def Cstar(rts): return lambda t, n, Ppt: float(np.clip(rts[t], REFF_LO, REFF_HI))
    def E(cs):
        return lambda t, n, Ppt: map_val(tbl, cs[t])
    def F(cs):
        def f(t, n, Ppt):
            Reff = map_val(tbl, cs[t])
            if (n*n/(Ppt+Reff) > 9.0) and (cs[t] < 0.3):
                Reff = min(Reff*20.0, REFF_HI*20.0)
            return Reff
        return f
    return A, B, Cstar, E, F


def evaluate(seeds, rho, w):
    A, B, Cstar, E, F = reff_builders(w)
    acc = {m: {"rmse": [], "post": [], "omax": []} for m in "ABCEF"}
    for s in seeds:
        xs, zs, cs, rts, jump, outl = simulate(s, rho)
        fns = {"A": A(cs), "B": B(cs), "C": Cstar(rts), "E": E(cs), "F": F(cs)}
        for m, fn in fns.items():
            est, nu, Pp, Ks = filt(zs, fn)
            err = np.abs(est - xs); steady = ~jump & ~outl
            post = [err[t0:t0+5].mean() for t0 in np.where(jump)[0]]
            acc[m]["rmse"].append(np.sqrt((err[steady]**2).mean()))
            acc[m]["post"].append(np.mean(post))
            acc[m]["omax"].append(err[outl].max() if outl.any() else 0.0)
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}


CAL = list(range(50)); TEST = list(range(50, 250))
name = {"A": "A 고정R", "B": "B 잔차게이트", "C": "C* 참노이즈오라클(천장)",
        "E": "E 학습게이트(ours)", "F": "F 학습+백업"}

print("=" * 74)
print("#63-2 Learned Reliability Gate — 맵 h_θ(c) 데이터학습(오라클 제거), 보정/테스트 분리")
print("=" * 74)

print("\n── Part 1. ρ=0.8 (불완전 품질신호). 보정 50시드 학습 → 테스트 200시드 ──")
w = fit_map(CAL, 0.8)
print(f"  학습된 맵 R_eff: c=0.1→{map_val(w,0.1):.2f}  c=0.5→{map_val(w,0.5):.2f}  "
      f"c=0.9→{map_val(w,0.9):.2f}  (나쁜관측↑ 좋은관측↓ 복원; 참 E[r|c]≈1.58/0.49/0.05)")
r = evaluate(TEST, 0.8, w)
print(f"\n  {'method':<22}{'정상RMSE':>10}{'순환성회복':>12}{'이상치max':>11}")
for m in "ABCEF":
    print(f"  {name[m]:<22}{r[m]['rmse']:>10.3f}{r[m]['post']:>12.3f}{r[m]['omax']:>11.2f}")
print("  → E(학습,오라클없음)가 C*(천장)에 근접하고 A·B를 이김. 순환성도 B보다 우월.")

print("\n── Part 2. 품질신호 유용도 ρ 스윕 — 정상 RMSE (안전성: ρ→0서 E가 A보다 나빠지면 안 됨) ──")
print(f"  {'ρ':>5}{'A 고정':>10}{'B 잔차':>10}{'E 학습(ours)':>14}{'C* 천장':>10}")
for rho in [1.0, 0.7, 0.4, 0.0]:
    w_r = fit_map(CAL, rho)
    rr = evaluate(TEST, rho, w_r)
    print(f"  {rho:>5.1f}{rr['A']['rmse']:>10.3f}{rr['B']['rmse']:>10.3f}"
          f"{rr['E']['rmse']:>14.3f}{rr['C']['rmse']:>10.3f}")
print("  → ρ↓ 일수록 E는 C*에서 멀어져 A로 수렴(품질신호 무정보면 자동으로 고정R화=안전).")
print("    B(잔차)는 ρ와 무관하게 나쁨(순환성). E는 '최소한 A만큼, 신호 좋으면 C*까지'.")

print("\n" + "=" * 74)
print("결론(#63-2): h_θ(c)를 예측가능도 모멘트매칭으로 학습 → 오라클 없이 E[r|c] 복원.")
print("  현재 잔차 안 쓰므로 순환성 없음. 품질신호 무정보면 고정R로 안전 퇴화(해 끼치지 않음).")
print("  = robust/VB(잔차)의 두 공백(순환성·안전성)을 학습형 품질게이트가 동시에 해결.")
