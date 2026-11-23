#!/usr/bin/env python3
"""
toy — Quantum-inspired KF (#41): 복소 진폭 상태 vs 실수 KF (샘플프리)
====================================================================
논제: PF의 다봉성을 '샘플 없이' 얻자. 복소 진폭 상태(파동함수형)를 가진 샘플프리 필터는
 (1) 다봉 사후를 단일 복소벡터로 표현(실수 KF=단일 가우시안→갭 붕괴),
 (2) 위상 간섭(cross term)으로 고전 확률 합을 넘는 표현력.
저차원은 그리드로 샘플프리, 고차원은 텐서네트워크(MPS=#40)로 확장.
numpy only.
"""
import numpy as np
rng=np.random.default_rng(0)

# ---------- Part 1: 다봉 사후를 샘플프리로 ----------
# 모호한 관측 y=cos(x) → 주어진 y0에 대해 x=±acos(y0) (두 모드). 사후가 bimodal.
N=512; x=np.linspace(-np.pi,np.pi,N)
x_true=1.2; y0=np.cos(x_true); sig=0.12
L=np.exp(-(np.cos(x)-y0)**2/(2*sig**2)); p=L/L.sum()     # 정확 사후(그리드) = |psi|^2
modes=np.array([1.2,-1.2])

def mass_near(prob, centers, tol=0.3):
    return sum(prob[np.abs(x-c)<tol].sum() for c in centers)

# 복소/그리드(샘플프리) 필터: |psi|^2 = p (그리드 상 bimodal 그대로 표현)
cov_complex = mass_near(p, modes)
# 실수 KF: 사후를 단일 가우시안으로 모멘트매칭 → μ,σ. 점추정 μ.
mu=(x*p).sum(); var=((x-mu)**2*p).sum()
gauss=np.exp(-(x-mu)**2/(2*var)); gauss/=gauss.sum()
cov_realkf = mass_near(gauss, modes)
print("===== Part 1: 다봉 사후 (샘플프리) =====")
print(f"참 모드 x=±1.2 (관측 y=cos x 모호)")
print(f"  복소/그리드 필터  : 모드질량 {cov_complex:.2f} (양 모드 보존)")
print(f"  실수 KF(가우시안) : 점추정 μ={mu:+.2f} (갭, 최근접모드까지 {min(abs(mu-1.2),abs(mu+1.2)):.2f}), 모드질량 {cov_realkf:.2f}")
print(f"  → 복소/그리드는 샘플 없이 두 모드 보존, 실수 KF는 가운데 갭으로 붕괴.\n")

# ---------- Part 2: 위상 간섭 = 고전 확률 넘는 표현력 ----------
# 같은 영역에 두 진폭 lump가 겹칠 때: 복소는 간섭(cross term), 고전은 위상-블라인드.
c1,c2,s=-0.4,0.4,0.5
A=np.exp(-(x-c1)**2/(2*s**2)); B=np.exp(-(x-c2)**2/(2*s**2))   # 진폭 크기(실수부분)
ov=(np.abs(x)<0.6)                                            # 겹침 영역
print("===== Part 2: 위상 간섭 (복소) vs 확률 합 (고전) =====")
print(f"{'Δφ':>8} {'복소 |a+b|² 겹침질량':>20} {'고전 |a|²+|b|² 겹침질량':>24}")
for dphi in [0.0, np.pi/2, np.pi]:
    a=A.astype(complex); b=B*np.exp(1j*dphi)
    pc=np.abs(a+b)**2; pc/=pc.sum()                          # 복소: 간섭
    pcl=(np.abs(a)**2+np.abs(b)**2); pcl/=pcl.sum()          # 고전: 합(위상 무시)
    print(f"{dphi/np.pi:>6.1f}π {pc[ov].sum():>20.3f} {pcl[ov].sum():>24.3f}")
print("  → 복소: Δφ=0 보강(겹침질량↑), Δφ=π 상쇄(겹침질량↓) — 위상이 분포를 바꿈.")
print("    고전: Δφ 무관 동일(위상 블라인드). = 간섭은 고전 확률이 못 하는 추가 표현력.\n")

print("===== 종합 (#41) =====")
print("(1) 복소/그리드 상태 = 샘플 없이 다봉(파동함수 |psi|²) — PF 강점을 KF 효율로.")
print("(2) 위상 간섭 = 고전 확률 합 넘는 표현력(#28·#40 그 수학).")
print("→ 'quantum-inspired KF': 샘플프리 + 간섭 + (소산성 결합 시)안정. 고차원은 MPS(#40)로.")
