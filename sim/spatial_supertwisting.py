#!/usr/bin/env python3
"""
Spatial Super-Twisting Aperture — 1-bit 개구 양자화 로브 셰이핑 시뮬레이션.

비교군:
  (a) ideal   : 이상 연속위상 (상한)
  (b) naive   : 메모리 없는 1-bit  sign(Re a)
  (c) dsm1    : 1차 공간 ΔΣ  (= 1-sliding), NTF (1 - z^-1)
  (d) dsm2lin : 선형 2차 ΔΣ  (대조군, 불안정/톤 위험)
  (e) stw     : 슈퍼트위스팅 2차 (제안), NTF ~ (1 - z^-1)^2 + |σ|^{1/2} 강인항

지표: 피크 양자화 로브(dB), 메인/미러 제외 ; 보호대역 적분전력 ∝ Ω_b^{2p+1} 검증.

의존성: numpy (matplotlib 있으면 그림 저장).
사용:   python3 spatial_supertwisting.py
문서:   ../docs/03_spatial_supertwisting_aperture.md
"""
import numpy as np

# ---------- 공통 ----------
def ideal_weights(N, th0_deg, d=0.5):
    n = np.arange(N)
    return np.exp(-1j * 2*np.pi * d * n * np.sin(np.deg2rad(th0_deg)))

def Q(x):
    """1-bit {±1}: Re 부호. 0은 +1로."""
    s = np.sign(np.real(x))
    s[s == 0] = 1.0
    return s

# ---------- 변조기들 ----------
def naive(a):
    return Q(a)

def dsm1(a):                       # 1차 ΔΣ (= 1-sliding)
    N = len(a); c = np.zeros(N); e = 0.0
    for n in range(N):
        u = a[n] - e
        cn = Q(np.array([u]))[0]
        e = cn - np.real(u)
        c[n] = cn
    return c

def dsm2_linear(a, b1=2.0, b2=1.0): # 선형 2차 ΔΣ (대조군)
    N = len(a); c = np.zeros(N); i1 = 0.0; i2 = 0.0
    for n in range(N):
        i1 += np.real(a[n]) - (c[n-1] if n > 0 else 0.0)
        i2 += i1
        u = b1*i1 + b2*i2
        cn = Q(np.array([u]))[0]
        c[n] = cn
    return c

def stw(a, k1=1.5, k2=1.1):        # 슈퍼트위스팅 2차 (제안)
    N = len(a); c = np.zeros(N); sig = 0.0; z = 0.0
    settle = None
    for n in range(N):
        u = a[n] - k1*np.sqrt(abs(sig))*np.sign(sig) - z
        cn = Q(np.array([u]))[0]
        c[n] = cn
        sig += cn - np.real(a[n])     # 첫 적분기(누적 코드오차)
        z   -= k2*np.sign(sig)        # 둘째 상태
        if settle is None and abs(sig) < 0.5:
            settle = n
    return c, (settle if settle is not None else N)

# ---------- 분석 ----------
def array_factor_db(c, Npad=8192):
    S = np.fft.fftshift(np.fft.fft(c, Npad))
    mag = np.abs(S)
    return 20*np.log10(mag/ (mag.max()+1e-12) + 1e-12)

def peak_quant_lobe_db(c, th0_deg, d=0.5, guard_deg=8.0, Npad=8192):
    """메인·미러 빔 주변 guard 제외한 영역의 피크(dB)."""
    afdb = array_factor_db(c, Npad)
    # FFT bin → sinθ 매핑: ω = 2π d sinθ ; fftshift 후 ω∈[-π,π) → sinθ = ω/(2π d)
    omega = np.linspace(-np.pi, np.pi, Npad, endpoint=False)
    sin_t = omega/(2*np.pi*d)
    vis = np.abs(sin_t) <= 1.0
    th = np.full(Npad, np.nan)
    th[vis] = np.rad2deg(np.arcsin(sin_t[vis]))
    mask = vis.copy()
    for center in (th0_deg, -th0_deg):
        mask &= ~(np.abs(th - center) < guard_deg)
    return np.nanmax(afdb[mask])

def inband_power(c, th0_deg, d=0.5, band_deg=6.0, Npad=8192):
    afdb = array_factor_db(c, Npad)
    lin = 10**(afdb/10)
    omega = np.linspace(-np.pi, np.pi, Npad, endpoint=False)
    sin_t = omega/(2*np.pi*d); vis = np.abs(sin_t) <= 1.0
    th = np.full(Npad, np.nan); th[vis] = np.rad2deg(np.arcsin(sin_t[vis]))
    band = vis & (np.abs(th - th0_deg) < band_deg) & (np.abs(th - th0_deg) > 0.5)
    return lin[band].sum()

# ---------- 메인 ----------
def main():
    N = 64; th0 = 30.0
    a = ideal_weights(N, th0)
    c_naive = naive(a)
    c_dsm1  = dsm1(a)
    c_dsm2  = dsm2_linear(a)
    c_stw, settle = stw(a)

    print(f"[N={N}, θ0={th0}°]  피크 양자화 로브(dB), 메인·미러 guard 제외")
    for name, c in [("naive 1-bit", c_naive), ("ΔΣ 1차", c_dsm1),
                    ("ΔΣ 2차(선형)", c_dsm2), ("슈퍼트위스팅", c_stw)]:
        print(f"  {name:14s}: {peak_quant_lobe_db(c, th0):7.2f} dB"
              f"   보호대역전력={inband_power(c, th0):.3e}")
    print(f"  슈퍼트위스팅 유한정착 소자수 ≈ {settle}")

    print("\n[예측검증] 보호대역 전력 ∝ Ω_b^(2p+1) — band_deg 스윕 로그-로그 기울기 확인:")
    for name, c in [("naive(p=0)", c_naive), ("ΔΣ1(p=1)", c_dsm1), ("STW(p=2)", c_stw)]:
        bands = [2,4,6,8,10]
        ps = [inband_power(c, th0, band_deg=b) for b in bands]
        slope = np.polyfit(np.log(bands), np.log(np.array(ps)+1e-18), 1)[0]
        print(f"  {name:11s}: log-log slope ≈ {slope:.2f}  (이론 2p+1)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Npad=8192
        omega = np.linspace(-np.pi, np.pi, Npad, endpoint=False)
        sin_t = omega/(2*np.pi*0.5); vis = np.abs(sin_t)<=1
        th = np.rad2deg(np.arcsin(np.clip(sin_t,-1,1)))
        plt.figure(figsize=(9,5))
        for name,c in [("ideal",Q(a)*0+1 and a),("naive",c_naive),
                       ("DSM-1",c_dsm1),("STW(ours)",c_stw)]:
            af = array_factor_db(c if not np.iscomplexobj(c) or name!="ideal" else a, Npad)
            plt.plot(th[vis], af[vis], label=name, lw=1.3)
        plt.xlabel("θ (deg)"); plt.ylabel("normalized |AF| (dB)")
        plt.ylim(-50,2); plt.legend(); plt.grid(alpha=.3)
        plt.title(f"Spatial Super-Twisting Aperture (N={N}, θ0={th0}°)")
        plt.tight_layout(); plt.savefig("spatial_supertwisting_patterns.png", dpi=130)
        print("\n[그림] spatial_supertwisting_patterns.png 저장됨")
    except Exception as ex:
        print(f"\n(matplotlib 없음/오류 — 수치만 출력: {ex})")

if __name__ == "__main__":
    main()
