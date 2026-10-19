#!/usr/bin/env python3
"""
Spatial Super-Twisting Aperture — 1-bit 개구 양자화 로브 셰이핑 시뮬레이션 (rev2, 2026-10-19).

rev1(첫 커밋) 수치가 이론과 어긋난 원인은 NOTES.md 참조. 요약:
  - θ0=30° 는 ω0=π/2 라 3차 고조파가 미러빔 위로 정확히 접힘 → -9.5 dB 로브가 guard 에 가려짐(퇴화 각도).
  - (1 - z^-1)^p 는 영점이 broadside(ω=0)에 있다. 조향빔(ω0≠0)에는 영점을 ±ω0 에 둔
    대역통과 NTF (1 - 2cosω0 z^-1 + z^-2)^p 가 필요하다.
  - 순수 (1 - z^-1)^2, (BP)^2 오차귀환은 |NTF|max 가 1-bit 안정 한계(~1.5)를 크게 넘어 과부하.
  - 단일 정현 입력의 양자화 오차는 백색이 아니라 톤 → §5 의 2p+1 기울기는 디더/위상 앙상블 전제.
  - 균일(직사각) 개구 자체의 누설이 대역 밖 셰이핑 노이즈를 대역 안으로 새게 해 기울기를 깎는다.

변조기 (실수 1-bit 코드 {±1}, 목표 x(n) = amp·cos(ω0 n + φ)):
  naive     : sign(x)                                   (p=0, 톤성 오차)
  efb(B,A)  : 오차귀환 ΔΣ, NTF = B(z)/A(z)  (u = x + (NTF-1)e,  c = sign(u),  e = c - u)
              ntf_lowpass(p)          → (1 - z^-1)^p          영점 ω=0 (broadside)
              ntf_bandpass(ω0, p, r)  → 영점 ±ω0, 극점 반경 r (r>0 이면 |NTF|max 억제)
  stw_rev1  : rev1 의 슈퍼트위스팅 루프 그대로 (z 부호 오류 → 양의 피드백, c≡+1 고착)
  stw_signfix: z 부호만 연속 STA 와 일치시킨 루프 (σ 유계. 셰이핑 성능은 미검증 가설, 저역형 구조)

의존성: numpy (matplotlib 있으면 그림 저장).
사용:   python3 spatial_supertwisting.py      (수 분 소요, 시드 고정)
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

def efb(x, B, A, dither=None):
    """1-bit 오차귀환 ΔΣ. NTF=B/A (둘 다 monic). c = x + NTF·e.  반환: c, max|u|(과부하 지표)."""
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
    rev1 은 '- z' 라 양의 피드백(σ→∞, c≡+1 고착). 이 수정은 σ 를 유계로 만들 뿐, 셰이핑 성능은 미검증."""
    N = len(x); c = np.zeros(N); sig = 0.0; z = 0.0; smax = 0.0
    for n in range(N):
        u = x[n] - k1*np.sqrt(abs(sig))*np.sign(sig) + z
        c[n] = 1.0 if u >= 0 else -1.0
        sig += c[n] - x[n]
        z -= k2*np.sign(sig)
        smax = max(smax, abs(sig))
    return c, smax

# ---------- 분석 ----------
def _grid(Npad):
    return 2*np.pi*np.fft.fftfreq(Npad)                 # ω ∈ [-π, π)

def _dist(w, center):
    return np.abs(np.angle(np.exp(1j*(w - center))))

def peak_quant_lobe_db(c, w0, d=0.5, guard_bins=3, Npad=1 << 14):
    """메인(+ω0)·미러(-ω0) 빔 ±guard_bins·(2π/N) 제외, 가시영역 |ω|≤2πd 안 피크(dB, 메인 대비).
    guard 를 각도가 아닌 빔폭(bin) 단위로 둬서 N·각도에 따른 왜곡을 없앤다."""
    N = len(c); w = _grid(Npad); C = np.abs(np.fft.fft(c, Npad))
    g = guard_bins*2*np.pi/N
    m = (np.abs(w) <= 2*np.pi*d) & (_dist(w, w0) >= g) & (_dist(w, -w0) >= g)
    return 20*np.log10(C[m].max()/C.max())

def inband_error_power(c, x, w0, Ob, window=None, Npad=1 << 16):
    """잔차 r = c - α̂x (α̂: 최소제곱 이득) 의 |ω-ω0|<Ob 적분 전력. 메인로브 자체는 빠진다."""
    al = (c @ x)/(x @ x); r = c - al*x
    if window is not None:
        r = r*window
    R = np.abs(np.fft.fft(r, Npad))**2
    w = _grid(Npad)
    Ob = np.atleast_1d(Ob)
    return np.array([R[_dist(w, w0) < o].sum() for o in Ob])

# ---------- 실험 ----------
def tone(N, w0, amp, rng):
    return amp*np.cos(w0*np.arange(N) + rng.uniform(0, 2*np.pi))

def e1_naive_lobe(rng):
    print("[E1] 순진 1-bit 피크 양자화 로브 (N=64, d=λ/2) — §2 예측 -9.54 dB")
    N = 64
    for th in [20.0, 23.7, 30.0, 35.0, 45.0]:
        w0 = np.pi*np.sin(np.deg2rad(th))
        h3 = np.angle(np.exp(3j*w0))
        pk_phase0 = peak_quant_lobe_db(naive(np.cos(w0*np.arange(N))), w0)
        pk_ens = np.mean([peak_quant_lobe_db(naive(tone(N, w0, 1.0, rng)), w0) for _ in range(100)])
        tag = "  ← 3ω0 가 미러(-ω0) 위로 접힘: 퇴화 각도" if _dist(h3, -w0) < 1e-9 else ""
        print(f"  θ0={th:5.1f}°  φ=0: {pk_phase0:6.2f} dB   위상앙상블: {pk_ens:6.2f} dB"
              f"   (3ω0→sin={h3/np.pi:+.3f}){tag}")

def e2_ntf_location(rng):
    print("\n[E2] 메인빔 근방(|ω-ω0|<0.05) 잔차전력, p=0 대비 (N=1024, amp=0.5, 디더 U(±0.5), 40회)")
    print("     예측: 저역 LP1 은 |NTF(e^{jω0})|² = (2 sin(ω0/2))², sinθ0>1/3(θ0>19.47°) 에서 >0 dB(악화)")
    N = 1024; Ob = 0.05
    for th in [5.0, 10.0, 19.47, 25.0, 45.0, 60.0]:
        w0 = np.pi*np.sin(np.deg2rad(th)); acc = {"p0": 0.0, "LP1": 0.0, "BP1": 0.0}
        mods = {"p0": (np.array([1.0]), np.array([1.0])), "LP1": ntf_lowpass(1),
                "BP1": ntf_bandpass(w0, 1)}
        for _ in range(40):
            x = tone(N, w0, 0.5, rng); dv = rng.uniform(-0.5, 0.5, N)
            for k, (B, A) in mods.items():
                acc[k] += inband_error_power(efb(x, B, A, dv)[0], x, w0, Ob, Npad=1 << 15)[0]
        pred = 20*np.log10(2*np.sin(w0/2))
        print(f"  θ0={th:5.2f}°  예측 LP1 {pred:+5.1f} dB | 측정 LP1 {10*np.log10(acc['LP1']/acc['p0']):+5.1f} dB"
              f" | 측정 BP1 {10*np.log10(acc['BP1']/acc['p0']):+6.1f} dB")

def e3_slope(rng):
    print("\n[E3] 보호대역 전력 ∝ Ω_b^(2p+1) 기울기 (N=4096, θ0=23.7°, amp=0.5, 10회) — §5 예측 1/3/5")
    N = 4096; w0 = np.pi*np.sin(np.deg2rad(23.7)); Obs = np.array([0.02, 0.04, 0.08, 0.16])
    cases = [("p=0 디더 sign", None, 0),
             ("p=0 순진(톤성)", None, -1),
             ("p=1 BP r=0.6", ntf_bandpass(w0, 1, 0.6), 0),
             ("p=2 BP r=0.7", ntf_bandpass(w0, 2, 0.7), 0),
             ("p=2 BP r=0 (순수)", ntf_bandpass(w0, 2, 0.0), 0)]
    for name, BA, mode in cases:
        out = []
        for wname, W in [("직사각", None), ("Hann", np.hanning(N))]:
            P = np.zeros(len(Obs)); umax = 0.0
            for _ in range(10):
                x = tone(N, w0, 0.5 if mode == 0 else 1.0, rng)
                if BA is None:
                    c = sgn(x + rng.uniform(-1, 1, N)) if mode == 0 else naive(x)
                else:
                    c, um = efb(x, *BA); umax = max(umax, um)
                P += inband_error_power(c, x, w0, Obs, window=W)
            out.append(f"{wname} {np.polyfit(np.log(Obs), np.log(P), 1)[0]:5.2f}")
        extra = "" if BA is None else f"   |NTF|max={ntf_maxgain(*BA):4.2f} max|u|={umax:.3g}"
        print(f"  {name:18s}: 기울기 " + " / ".join(out) + extra)

def e4_stw(rng):
    print("\n[E4] 슈퍼트위스팅 루프 — 미검증 가설, 참고 수치 (N=64, 피크 로브 100회 평균)")
    N = 64
    x0 = np.cos(np.pi*np.sin(np.deg2rad(23.7))*np.arange(N))
    c0 = stw_rev1(x0)
    print(f"  rev1 고착 확인(θ0=23.7°, φ=0): c 중 +1 비율 {np.mean(c0 > 0):.2f}")
    for th in [10.0, 23.7, 45.0]:
        w0 = np.pi*np.sin(np.deg2rad(th))
        nv = np.mean([peak_quant_lobe_db(naive(tone(N, w0, 1.0, rng)), w0) for _ in range(100)])
        r1 = np.mean([peak_quant_lobe_db(stw_rev1(tone(N, w0, 1.0, rng)), w0) for _ in range(100)])
        row = [f"naive {nv:6.2f}", f"rev1 {r1:6.2f}"]
        for k1, k2 in [(1.5, 1.1), (0.2, 0.02)]:
            res = [stw_signfix(tone(N, w0, 0.5, rng), k1, k2) for _ in range(100)]
            row.append(f"부호수정 k1={k1},k2={k2} {np.mean([peak_quant_lobe_db(c, w0) for c, _ in res]):6.2f}"
                       f" (max|σ|={max(s for _, s in res):.2f})")
        print(f"  θ0={th:5.1f}°  " + " | ".join(row) + " dB")

def e5_peak_vs_band(rng):
    print("\n[E5] 보호대역 밖 피크 로브(d=λ/2, 메인·미러 ±3빔폭 제외, 50회) — 대역 내 셰이핑이 피크로 이어지나")
    for N in [64, 256]:
        row = []
        for th in [10.0, 23.7, 45.0]:
            w0 = np.pi*np.sin(np.deg2rad(th))
            nv = np.mean([peak_quant_lobe_db(naive(tone(N, w0, 1.0, rng)), w0) for _ in range(50)])
            bp = [np.mean([peak_quant_lobe_db(efb(tone(N, w0, amp, rng), *ntf_bandpass(w0, 1))[0], w0)
                           for _ in range(50)]) for amp in (1.0, 0.5)]
            row.append(f"θ0={th:4.1f}° naive {nv:6.2f} / BP1(amp1) {bp[0]:6.2f} / BP1(amp.5) {bp[1]:6.2f}")
        print(f"  N={N:3d}  " + " | ".join(row) + " dB")

def main():
    rng = np.random.default_rng(20261019)
    e1_naive_lobe(rng)
    e2_ntf_location(rng)
    e3_slope(rng)
    e4_stw(rng)
    e5_peak_vs_band(rng)

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
