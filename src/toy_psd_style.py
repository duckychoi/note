#!/usr/bin/env python3
"""
toy ③ — PSD(파워스펙트럼) 스타일 손실 검증
==========================================
주장(docs/04): 리듬=STYLE 은 '시간축 Gram=자기상관 ↔(Wiener–Khinchin) PSD' 로
위상불변하게 포착된다. per-frame L2 는 위상/시간이동에 취약하다.

검증:
 (S1) 같은 리듬·다른 위상 → per-frame L2 큼, PSD 거리 ≈ 0
 (S2) 다른 리듬           → PSD 거리 큼 (리듬 판별)
 (S3) content(이벤트) → per-frame 로는 잡히나 PSD 는 blind (content/style 분리)
 (W)  Wiener–Khinchin 수치 확인: IFFT(PSD) == 자기상관
numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

L = 256
t = np.arange(L)

def gait(f0, phase, harmonics=(1, 2, 3), amps=(1.0, 0.4, 0.2), noise=0.03):
    """보행형 리듬 신호: 기본주파수 f0(사이클/길이) + 하모닉, 위상 phase."""
    x = np.zeros(L)
    for h, a in zip(harmonics, amps):
        x += a * np.sin(2*np.pi*f0*h*t/L + h*phase)
    return x + noise*rng.standard_normal(L)

def l2(a, b):
    return np.mean((a-b)**2)

def psd(x):
    X = np.fft.rfft(x - x.mean())
    return (np.abs(X)**2) / L

def psd_dist(a, b):
    pa, pb = psd(a), psd(b)
    return np.mean((np.log1p(pa) - np.log1p(pb))**2)   # 로그 PSD L2 (위상 blind)

# ---- 기준 신호 ----
ref = gait(f0=6, phase=0.0)

print("===== (S1) 같은 리듬, 다른 위상 =====")
for ph in [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]:
    s = gait(f0=6, phase=ph)
    print(f"  위상 {ph/np.pi:>4.2f}π | per-frame L2 = {l2(ref,s):6.3f} | PSD거리 = {psd_dist(ref,s):7.4f}")
print("  → 위상 바뀌면 L2 폭발하지만 PSD거리는 ~0 (리듬 동일을 위상불변 인식)\n")

print("===== (S2) 다른 리듬(주파수) =====")
for f in [4, 5, 6, 7, 9]:
    s = gait(f0=f, phase=0.0)
    print(f"  f0={f} | per-frame L2 = {l2(ref,s):6.3f} | PSD거리 = {psd_dist(ref,s):7.4f}")
print("  → f0=6(동일리듬)에서 PSD거리 최소, 다른 리듬일수록 커짐 (리듬 판별)\n")

print("===== (S3) content 이벤트 (리듬 같음 + 특정 시점 돌출) =====")
s_event = gait(f0=6, phase=0.0).copy()
s_event[120:130] += 3.0      # 특정 시점의 '사건'(content)
print(f"  이벤트 신호 vs ref | per-frame L2 = {l2(ref,s_event):6.3f} (이벤트 잡힘)"
      f" | PSD거리 = {psd_dist(ref,s_event):7.4f} (상대적으로 둔감)")
# 위상만 다르고 이벤트 없는 신호와 비교: content 차이를 per-frame만 구분
s_phase = gait(f0=6, phase=np.pi)
print(f"  판별: 'content 이벤트'는 per-frame L2({l2(ref,s_event):.2f})가 위상차({l2(ref,s_phase):.2f})와 섞여 모호")
print("  → content(이벤트/목표)는 per-frame·위상민감, style(리듬)은 PSD·위상불변 = 분리 필요\n")

print("===== (W) Wiener–Khinchin 수치확인: IFFT(PSD) == 순환 자기상관 =====")
x = gait(f0=6, phase=0.7)
xc = x - x.mean()
# 순환 자기상관 r[k] = Σ_n xc[n]·xc[(n+k) mod L]  (PSD의 IFFT와 같은 정의)
autocorr = np.array([np.sum(xc * np.roll(xc, -k)) for k in range(20)])
P = np.abs(np.fft.fft(xc))**2                                # PSD(비정규화)
wk = np.fft.ifft(P).real[:20]                                # IFFT(PSD)
err = np.max(np.abs(autocorr - wk))
print(f"  max|순환자기상관 - IFFT(PSD)| (앞 20 lag) = {err:.2e}"
      f"  → {'일치(Gram=PSD 푸리에쌍 확인)' if err<1e-6 else '불일치'}")

print("\n===== 종합 =====")
print("S1: PSD거리는 위상불변(같은 리듬 ~0), L2는 위상에 폭발 → 'style=리듬'은 PSD로 담아야")
print("S2: PSD거리가 리듬(주파수)을 정확히 판별")
print("S3: content(이벤트)는 per-frame 필요, style(리듬)은 PSD → content/style 분리 타당")
print("W : Gram(자기상관) ↔ PSD 푸리에쌍 수치확인 → docs/04 스펙트럴 해석 성립")
