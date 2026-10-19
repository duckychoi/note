#!/usr/bin/env python3
"""
Spatial Super-Twisting Aperture — 1-bit 개구 양자화 로브 셰이핑 시뮬레이션 (rev2, 2026-10-19).

rev1(첫 커밋) 수치가 이론과 어긋난 원인은 NOTES.md 참조. 요약:
  - θ0=30° 는 4ω0∈2πℤ (ω0=π/2) 라 3차 고조파가 미러빔 위로 정확히 접힘 → -9.5 dB 로브가 guard 에 가려짐.
  - (1 - z^-1)^p 는 영점이 broadside(ω=0). 조향빔에는 영점 ±ω0 의 대역통과 NTF (1 - 2cosω0 z^-1 + z^-2)^p.
  - rev1 dsm2_linear 는 선형화 극점 z=-1.618 (단위원 밖) 인 불안정 구조.
  - rev1 STA 이산화는 z 부호 오류(양의 피드백) → 한 부호로 고착.
  - rev1 피크 지표는 분모가 전체 최댓값이라 0 dB 에서 잘려 실패를 가림 → 조향점 |AF(±ω0)| 대비로 변경.
  - 기울기 검증은 r = c − x (STF=1) 로, 백색 근사는 디더로. 균일 개구 누설이 고차 기울기를 깎음(N 의존).

변조기 (실수 1-bit 코드 {±1}, 목표 x(n) = amp·cos(ω0 n + φ)):
  naive     : sign(x)                                   (p=0, 톤성 오차)
  efb(B,A)  : 오차귀환 ΔΣ, NTF = B(z)/A(z)  (u = x + (NTF-1)e,  c = sign(u),  e = c - u)
              ntf_lowpass(p)          → (1 - z^-1)^p          영점 ω=0 (broadside)
              ntf_bandpass(ω0, p, r)  → 영점 ±ω0, 극점 반경 r (r>0 이면 |NTF|max 억제)
  dsm2_rev1 : rev1 의 dsm2_linear 그대로 (선형화 불안정, E0)
  stw_rev1  : rev1 의 슈퍼트위스팅 루프 그대로 (z 부호 오류 → 양의 피드백, 한 부호로 고착)
  stw_signfix: z 부호만 연속 STA 와 일치시킨 루프 (σ 유계. 셰이핑 성능은 미검증 가설, 저역형 구조)

의존성: numpy (matplotlib 있으면 그림 저장).
사용:   python3 spatial_supertwisting.py      (약 20초, 시드 고정; Windows 콘솔은 python -X utf8)
문서:   ../docs/03_spatial_supertwisting_aperture.md
"""
import numpy as np

# ---------- 공통 ----------
def ideal_weights(N, th0_deg, d=0.5):
    n = np.arange(N)
    return np.exp(-1j * 2*np.pi * d * n * np.sin(np.deg2rad(th0_deg)))

def sgn(v):
    """1-bit {±1}. 0은 +1로."""
    return np.where(np.asarray(v) >= 0, 1.0, -1.0)

# ---------- 변조기들 ----------
def naive(x):
    return sgn(x)

def ntf_lowpass(p):
    B = np.array([1.0])
    for _ in range(p):
        B = np.convolve(B, [1.0, -1.0])
    return B, np.array([1.0])

def ntf_bandpass(w0, p, r=0.0):
    b = np.array([1.0, -2*np.cos(w0), 1.0])
    a = np.array([1.0, -2*r*np.cos(w0), r*r])
    B = np.array([1.0]); A = np.array([1.0])
    for _ in range(p):
        B = np.convolve(B, b); A = np.convolve(A, a)
    return B, A

def ntf_maxgain(B, A, M=4096):
    zi = np.exp(-1j*np.linspace(0, np.pi, M))          # z^-1
    return np.max(np.abs(np.polyval(B[::-1], zi) / np.polyval(A[::-1], zi)))

def efb(x, B, A, dither=None, elog=None):
    """1-bit 오차귀환 ΔΣ. NTF=B/A (둘 다 monic). c = x + NTF·e.  반환: c, max|u|(과부하 지표).
    elog 에 list 를 주면 내부 오차 e(n) 을 기록한다(백색성 확인용)."""
    N = len(x); K = max(len(B), len(A)) - 1
    B = np.pad(B, (0, K+1-len(B))); A = np.pad(A, (0, K+1-len(A))); D = B - A
    e = np.zeros(K); y = np.zeros(K); c = np.zeros(N); umax = 0.0
    dv = np.zeros(N) if dither is None else dither
    for n in range(N):
        yn = D[1:] @ e - A[1:] @ y if K else 0.0      # (NTF-1)·e, 인과
        u = x[n] + yn
        cn = 1.0 if u + dv[n] >= 0 else -1.0
        if K:
            e = np.roll(e, 1); e[0] = cn - u
            y = np.roll(y, 1); y[0] = yn
        c[n] = cn; umax = max(umax, abs(u))
        if elog is not None:
            elog.append(cn - u)
    return c, umax

def stw_rev1(x, k1=1.5, k2=1.1):
    """rev1 슈퍼트위스팅 루프(변경 없음, 실수 입력으로 정리). 미검증 가설 — NOTES.md §STW."""
    N = len(x); c = np.zeros(N); sig = 0.0; z = 0.0
    for n in range(N):
        u = x[n] - k1*np.sqrt(abs(sig))*np.sign(sig) - z
        c[n] = 1.0 if u >= 0 else -1.0
        sig += c[n] - x[n]
        z -= k2*np.sign(sig)
    return c

def stw_signfix(x, k1=1.5, k2=1.1):
    """rev1 루프의 z 부호만 연속 STA(σ̇ = -k1|σ|^½sgnσ + z, ż = -k2 sgnσ)와 일치시킴: u = x - k1|σ|^½sgnσ + z.
    rev1 은 '- z' 라 양의 피드백(σ→∞, 한 부호로 고착). 이 수정은 σ 를 유계로 만들 뿐, 셰이핑 성능은 미검증."""
    N = len(x); c = np.zeros(N); sig = 0.0; z = 0.0; smax = 0.0
    for n in range(N):
        u = x[n] - k1*np.sqrt(abs(sig))*np.sign(sig) + z
        c[n] = 1.0 if u >= 0 else -1.0
        sig += c[n] - x[n]
        z -= k2*np.sign(sig)
        smax = max(smax, abs(sig))
    return c, smax

def dsm2_rev1(x, b1=2.0, b2=1.0):
    """rev1 dsm2_linear 그대로(실수 입력). 선형화: NTF=(1-q)²/(1+q-q²), STF=(3-2q)/(1+q-q²), q=z⁻¹
    → 극점 z = 0.618, −1.618 (단위원 밖): 선형화 루프 자체가 불안정."""
    N = len(x); c = np.zeros(N); i1 = 0.0; i2 = 0.0; umax = 0.0
    for n in range(N):
        i1 += x[n] - (c[n-1] if n > 0 else 0.0)
        i2 += i1
        u = b1*i1 + b2*i2
        c[n] = 1.0 if u >= 0 else -1.0
        umax = max(umax, abs(u))
    return c, umax

def lfilter(B, A, x):
    """y = (B/A)·x, A monic (scipy 없이)."""
    K = max(len(B), len(A)); B = np.pad(B, (0, K-len(B))); A = np.pad(A, (0, K-len(A)))
    y = np.zeros(len(x))
    for n in range(len(x)):
        acc = 0.0
        for k in range(min(K, n+1)):
            acc += B[k]*x[n-k] - (A[k]*y[n-k] if k else 0.0)
        y[n] = acc
    return y

# ---------- 분석 ----------
def _grid(Npad):
    return 2*np.pi*np.fft.fftfreq(Npad)                 # ω ∈ [-π, π)

def _dist(w, center):
    return np.abs(np.angle(np.exp(1j*(w - center))))

def peak_quant_lobe_db(c, w0, d=0.5, guard_bins=3, Npad=1 << 14):
    """가시영역 |ω|≤2πd 에서 메인(+ω0)·미러(-ω0) ±guard_bins·(2π/N) 밖 피크를 **정확한 조향점** |AF(ω0)| 대비 dB 로.
    분모 = |Σ c(n) e^{-jω0 n}| (직접 합; 실수 코드는 ±ω0 에서 같은 크기). 외부 로브가 메인보다 크면 > 0 dB."""
    N = len(c); w = _grid(Npad); C = np.abs(np.fft.fft(c, Npad))
    g = guard_bins*2*np.pi/N
    main = np.abs(c @ np.exp(-1j*w0*np.arange(N)))
    m = (np.abs(w) <= 2*np.pi*d) & (_dist(w, w0) >= g) & (_dist(w, -w0) >= g)
    return 20*np.log10(C[m].max()/main)

def band_psd(r, w0, Obs, window=None, Npad=None):
    """잔차 r 의 |ω-ω0|<Ob 대역 평균 PSD. 정규화: 분산 σ² 인 백색 r 이면 기대값 σ² (Npad·창과 무관)."""
    N = len(r); Npad = Npad or 4*N
    W = np.ones(N) if window is None else window
    R = np.abs(np.fft.fft(r*W, Npad))**2/(W @ W)
    w = _grid(Npad)
    return np.array([R[_dist(w, w0) < o].mean() for o in np.atleast_1d(Obs)])

def ntf_band_slope(B, A, w0, Obs, M=200001):
    """백색 오차를 NTF 에 통과시켰을 때(무한 개구) 대역 전력의 정확한 로그-로그 기울기 — 점근값 2p+1 과 구분."""
    P = []
    for o in Obs:
        zi = np.exp(-1j*(w0 + np.linspace(-o, o, M)))
        P.append(np.mean(np.abs(np.polyval(B[::-1], zi)/np.polyval(A[::-1], zi))**2)*2*o)
    return np.polyfit(np.log(Obs), np.log(P), 1)[0]

# ---------- 실험 ----------
def tone(N, w0, amp, rng):
    return amp*np.cos(w0*np.arange(N) + rng.uniform(0, 2*np.pi))

def e0_rev1_dsm2(rng):
    print("[E0] rev1 dsm2_linear 구조 — 선형 가산오차 모델 NTF=(1-q)²/(1+q-q²), q=z⁻¹ (비선형 루프의 안정성 판정은 아님)")
    print(f"  극점 z (z² + z − 1 = 0): {np.sort(np.roots([1, 1, -1]))}  → |z|=1.618 이 단위원 밖")
    N = 64
    for th in [10.0, 23.7, 45.0]:
        w0 = np.pi*np.sin(np.deg2rad(th))
        res = [dsm2_rev1(tone(N, w0, 1.0, rng)) for _ in range(50)]
        print(f"  θ0={th:4.1f}°  피크 로브(메인 대비) {np.mean([peak_quant_lobe_db(c, w0) for c, _ in res]):+6.2f} dB"
              f"   관측 max|u| {max(u for _, u in res):.3g} (50회, 일반 상한 아님)")

def e1_naive_lobe(rng):
    print("\n[E1] 순진 1-bit 피크 양자화 로브 (N=64, d=λ/2, 메인 대비) — §2 예측 -9.54 dB")
    N = 64
    for th in [20.0, 23.7, 30.0, 35.0, 45.0]:
        w0 = np.pi*np.sin(np.deg2rad(th))
        h3 = np.angle(np.exp(3j*w0))
        pk_phase0 = peak_quant_lobe_db(naive(np.cos(w0*np.arange(N))), w0)
        pk_ens = np.mean([peak_quant_lobe_db(naive(tone(N, w0, 1.0, rng)), w0) for _ in range(100)])
        tag = "  ← 4ω0∈2πℤ: 3ω0 가 미러(-ω0) 위로 접힘(퇴화 각도)" if _dist(h3, -w0) < 1e-9 else ""
        print(f"  θ0={th:5.1f}°  φ=0: {pk_phase0:6.2f} dB   위상앙상블: {pk_ens:6.2f} dB"
              f"   (3ω0→sin={h3/np.pi:+.3f}){tag}")

def e2_ntf_location(rng):
    print("\n[E2] 메인빔 근방(|ω-ω0|<0.05) 잔차 r=c−x 의 대역 PSD (N=1024, amp=0.5, 40회)")
    print("     p0 = sign(x+U(±1)) (E[c|x]=x, 이득 1) / LP1·BP1 = 루프 내 디더 U(±0.5), STF=1 (c = x + NTF·e)")
    print("     |NTF_LP1(e^{jω0})|² = (2 sin(ω0/2))² : *같은 내부 오차*에 대한 이득 (sinθ0 > 1/3 이면 > 1)")
    print("     백색 예측 LP1/p0 = |NTF|²·σ²_e,LP1/σ²_e,p0 (내부 오차가 백색일 때만 성립하는 조건부 근사)")
    print("     백색지수 = 내부 오차 e 의 대역 PSD / var(e) (백색이면 0 dB; p0: e=c−x, LP1: e=cumsum(c−x))")
    print("     '목표톤 대비' = 대역 평균 |R|² / 명목 목표 톤 전력 (amp·N/2)² — §2 의 피크 로브와는 다른 지표")
    N = 1024; Ob = 0.05; amp = 0.5; main2 = (amp*N/2)**2
    for th in [5.0, 10.0, 19.0, float(np.rad2deg(np.arcsin(1/3))), 20.0, 25.0, 45.0, 60.0,
               19.47, 19.48, 19.4, 19.55, 19.3, 19.2, 19.7]:
        # 뒤 7개: asin(1/3) 근방 스윕(앞 각도 수치 불변). |6ω0−2π| 열은 '(6k+1)ω0 고조파가 대역(<Ob)에 접힘' 가설 검사용
        # — 19.2°·19.7°(|6ω0−2π|>Ob)도 유색이라 'k=1 조건이 필요조건' 인 단순 가설은 반증됨.
        #   고차 k 의 (6k+1)ω0 접힘 기여 여부와 유색 원인은 미확인.
        w0 = np.pi*np.sin(np.deg2rad(th)); acc = {"p0": 0.0, "LP1": 0.0, "BP1": 0.0}
        ve = {"p0": 0.0, "LP1": 0.0}; wi = {"p0": 0.0, "LP1": 0.0}
        for _ in range(40):
            x = tone(N, w0, amp, rng); dv = rng.uniform(-0.5, 0.5, N)
            c = {"p0": sgn(x + rng.uniform(-1, 1, N)),
                 "LP1": efb(x, *ntf_lowpass(1), dither=dv)[0],
                 "BP1": efb(x, *ntf_bandpass(w0, 1), dither=dv)[0]}
            for k in acc:
                acc[k] += band_psd(c[k] - x, w0, Ob)[0]/40
            ei = {"p0": c["p0"] - x,                           # p0: e = c − x
                  "LP1": np.cumsum(c["LP1"] - x)}              # LP1: c − x = e(n) − e(n−1), e(−1)=0 ⇒ e = cumsum
            for k, e in ei.items():
                ve[k] += np.var(e)/40; wi[k] += band_psd(e, w0, Ob)[0]/40
        rel = lambda k: 10*np.log10(acc[k]/acc["p0"])
        ab = lambda k: 10*np.log10(acc[k]*N/main2)       # 대역 평균 |R|² = PSD·N
        pred = 20*np.log10(2*np.sin(w0/2)) + 10*np.log10(ve["LP1"]/ve["p0"])
        wix = lambda k: 10*np.log10(wi[k]/ve[k])
        print(f"  θ0={th:6.3f}° [|6ω0−2π|={abs(6*w0-2*np.pi):.3f}{'<Ob' if abs(6*w0-2*np.pi) < Ob else '   '}]  |NTF_LP1|² {20*np.log10(2*np.sin(w0/2)):+5.1f} | 백색예측 {pred:+5.1f} / 측정 LP1 {rel('LP1'):+5.1f}"
              f" (백색지수 p0 {wix('p0'):+4.1f}, LP1 {wix('LP1'):+4.1f}) | p0 대비 BP1 {rel('BP1'):+6.1f}"
              f" | 목표톤 대비: p0 {ab('p0'):6.1f}, LP1 {ab('LP1'):6.1f}, BP1 {ab('BP1'):6.1f} dB")

def e3_slope(rng):
    Obs = np.array([0.02, 0.04, 0.08, 0.16]); w0 = np.pi*np.sin(np.deg2rad(23.7)); amp = 0.5
    print("\n[E3] 보호대역 전력 ∝ Ω_b^(2p+1) 기울기 (θ0=23.7°, amp=0.5, Ω_b∈{0.02..0.16}) — §5 점근 예측 1/3/5")
    print("     잔차 r=c−x (STF=1, 이득 추정 없음), 같은 표본에 직사각·Hann 둘 다 적용. BP 루프는 디더 U(±0.5).")
    print("     선형모델 = 독립 백색 e 를 NTF 로 필터한 대조군(같은 N·창) / 정확 = 무한 개구에서 |NTF|² 적분")
    cases = [("p=0 디더 sign", None), ("p=1 BP r=0.6", ntf_bandpass(w0, 1, 0.6)),
             ("p=2 BP r=0.7", ntf_bandpass(w0, 2, 0.7))]
    sl = lambda v: np.polyfit(np.log(Obs), np.log(v*Obs), 1)[0]
    for N, M in [(4096, 10), (65536, 3)]:
        print(f"  N={N}, {M}회")
        W = {"직사각": None, "Hann": np.hanning(N)}
        for name, BA in cases:
            P = {k: 0.0 for k in W}; L = {k: 0.0 for k in W}; umax = 0.0; ac = np.zeros(2)
            for _ in range(M):
                x = tone(N, w0, amp, rng)
                if BA is None:
                    c = sgn(x + rng.uniform(-1, 1, N)); el = rng.standard_normal(N)
                else:
                    eg = []
                    c, um = efb(x, *BA, dither=rng.uniform(-0.5, 0.5, N), elog=eg); umax = max(umax, um)
                    eg = np.array(eg) - np.mean(eg)
                    ac += np.array([eg[:-k] @ eg[k:] for k in (1, 2)])/(eg @ eg)/M
                    el = lfilter(*BA, rng.standard_normal(N + 2000))[2000:]
                for k in W:
                    P[k] = P[k] + band_psd(c - x, w0, Obs, W[k])
                    L[k] = L[k] + band_psd(el, w0, Obs, W[k])
            ex = 1.0 if BA is None else ntf_band_slope(*BA, w0, Obs)
            extra = "" if BA is None else (f"  |NTF|max={ntf_maxgain(*BA):4.2f} max|u|={umax:.3g}"
                                           f" 내부e 자기상관 lag1 {ac[0]:+.3f} lag2 {ac[1]:+.3f}")
            print(f"    {name:14s}: 측정 직사각 {sl(P['직사각']):4.2f} / Hann {sl(P['Hann']):4.2f}"
                  f" | 선형모델 직사각 {sl(L['직사각']):4.2f} / Hann {sl(L['Hann']):4.2f} | 정확 {ex:4.2f}{extra}")

def e4_stw(rng):
    print("\n[E4] 슈퍼트위스팅 루프 — 미검증 가설 (N=64, 피크 로브는 메인 대비, 100회 평균, 실수 목표 x)")
    N = 64; w0 = np.pi*np.sin(np.deg2rad(23.7)); x0 = np.cos(w0*np.arange(N))
    for s in (+1, -1):
        c0 = stw_rev1(s*x0)
        print(f"  rev1 (θ0=23.7°, φ=0, 입력 부호 {s:+d}): +1 비율 {int(np.sum(c0 > 0))}/{N},"
              f" 피크 로브 {peak_quant_lobe_db(c0, w0):+6.2f} dB")
    for th in [10.0, 23.7, 45.0]:
        w0 = np.pi*np.sin(np.deg2rad(th))
        nv = np.mean([peak_quant_lobe_db(naive(tone(N, w0, 1.0, rng)), w0) for _ in range(100)])
        r1 = np.mean([peak_quant_lobe_db(stw_rev1(tone(N, w0, 1.0, rng)), w0) for _ in range(100)])
        row = [f"naive {nv:+6.2f}", f"rev1 {r1:+6.2f}"]
        for k1, k2 in [(1.5, 1.1), (0.2, 0.02)]:
            res = [stw_signfix(tone(N, w0, 0.5, rng), k1, k2) for _ in range(100)]
            row.append(f"부호수정 k1={k1},k2={k2} {np.mean([peak_quant_lobe_db(c, w0) for c, _ in res]):+6.2f}"
                       f" (max|σ|={max(s for _, s in res):.2f})")
        print(f"  θ0={th:5.1f}°  " + " | ".join(row) + " dB")

def e5_peak_vs_band(rng):
    print("\n[E5] 보호대역 밖 피크 로브(d=λ/2, 메인·미러 ±3빔폭 제외, 메인 대비, 50회)")
    for N in [64, 256]:
        row = []
        for th in [10.0, 23.7, 45.0]:
            w0 = np.pi*np.sin(np.deg2rad(th))
            nv = np.mean([peak_quant_lobe_db(naive(tone(N, w0, 1.0, rng)), w0) for _ in range(50)])
            bp = [np.mean([peak_quant_lobe_db(efb(tone(N, w0, amp, rng), *ntf_bandpass(w0, 1))[0], w0)
                           for _ in range(50)]) for amp in (1.0, 0.5)]
            row.append(f"θ0={th:4.1f}° naive {nv:+6.2f} / BP1(amp1) {bp[0]:+6.2f} / BP1(amp.5) {bp[1]:+6.2f}")
        print(f"  N={N:3d}  " + " | ".join(row) + " dB")

def main():
    for k, f in enumerate([e0_rev1_dsm2, e1_naive_lobe, e2_ntf_location, e3_slope, e4_stw, e5_peak_vs_band]):
        f(np.random.default_rng(20261019 + k))     # 실험별 독립 시드: 한 실험을 고쳐도 다른 실험 수치가 안 바뀜

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        N = 256; th0 = 23.7; w0 = np.pi*np.sin(np.deg2rad(th0)); Npad = 1 << 14
        x = np.cos(w0*np.arange(N)); w = np.fft.fftshift(_grid(Npad))
        th = np.rad2deg(np.arcsin(w/np.pi))
        plt.figure(figsize=(9, 5))
        for name, c in [("naive", naive(x)), ("ΔΣ1 lowpass (zero @ broadside)", efb(0.5*x, *ntf_lowpass(1))[0]),
                        ("ΔΣ1 bandpass (zeros @ ±ω0)", efb(0.5*x, *ntf_bandpass(w0, 1))[0])]:
            C = np.abs(np.fft.fftshift(np.fft.fft(c, Npad)))
            plt.plot(th, 20*np.log10(C/C.max()+1e-12), label=name, lw=1.0)
        plt.xlabel("θ (deg)"); plt.ylabel("normalized |AF| (dB)")
        plt.ylim(-50, 2); plt.legend(); plt.grid(alpha=.3)
        plt.title(f"1-bit aperture, N={N}, θ0={th0}°, d=λ/2")
        plt.tight_layout(); plt.savefig("spatial_supertwisting_patterns.png", dpi=130)
        print("\n[그림] spatial_supertwisting_patterns.png 저장됨")
    except Exception as ex:
        print(f"\n(matplotlib 없음/오류 — 수치만 출력: {ex})")

if __name__ == "__main__":
    main()
