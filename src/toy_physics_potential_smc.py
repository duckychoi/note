#!/usr/bin/env python3
"""
toy — 물리-potential SMC (#36): 통계 prior × 물리 법칙 융합
=========================================================
사후 ∝ p_θ(궤적)[통계 생성 prior] × exp(-λ·E_phys)[물리 potential].
생성모델=제안, 물리=가중치, 리샘플+물리 rejuvenation=곱 강제. 재학습 없이 추론때 물리 주입.
toy 물리: 1D 수직운동(중력). 물리궤적 y_t=y0+v0 t-½g t². 물리 ⟺ 2차차분=-g.
prior: 물리궤적 + 잡음(비물리 jitter). 비교: raw prior vs 물리가이드 vs 물리-SMC.
지표: 물리위반에너지 E_phys ↓, 그리고 다양성(초기속도 분산) 보존 여부(붕괴 아님).
numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

T = 24          # 시점
G = 0.05        # 중력(인덱스 단위 가속)
SIG = 0.22      # prior 잡음(비물리 jitter)
tt = np.arange(T)

# 2차차분 연산자 S: (Sy)_i = y_i - 2y_{i+1} + y_{i+2}, 물리면 = -G
S = np.zeros((T-2, T))
for i in range(T-2): S[i,i]=1; S[i,i+1]=-2; S[i,i+2]=1
gvec = -G*np.ones(T-2)

def ballistic(y0,v0): return y0 + v0*tt - 0.5*G*tt**2
def E_phys(Y):                 # Y:(K,T) → 물리위반에너지(낮을수록 물리적)
    r = Y@S.T - gvec[None,:]
    return (r**2).sum(1)
def gradE(Y):                  # ∇E = 2 (SY - g) S  (각 행)
    return 2*((Y@S.T - gvec[None,:])@S)
def est_v0(Y): return Y[:,1]-Y[:,0]   # 초기속도 추정(다양성 지표)

# prior 샘플: 물리궤적(랜덤 y0,v0) + jitter
def prior_sample(K):
    y0 = rng.uniform(-1,1,K); v0 = rng.uniform(0.3,1.2,K)
    base = y0[:,None] + v0[:,None]*tt[None,:] - 0.5*G*tt[None,:]**2
    return base + SIG*rng.standard_normal((K,T)), v0

K = 2000
Y0, v0_true = prior_sample(K)

# --- (A) 물리 가이드 (샘플링 guidance = Langevin on E_phys) ---
def phys_guide(Y, steps=60, eta=0.02, tau=0.0):
    Y=Y.copy()
    for _ in range(steps):
        Y = Y - eta*gradE(Y) + tau*rng.standard_normal(Y.shape)
    return Y

# --- (B) 물리-SMC (리샘플 + 물리 rejuvenation) ---
def phys_smc(Y, rounds=6, lam=2.0, rej=8, eta=0.02):
    Y=Y.copy()
    for _ in range(rounds):
        w = np.exp(-lam*(E_phys(Y)-E_phys(Y).min())); w/=w.sum()
        idx = rng.choice(len(Y), len(Y), p=w); Y=Y[idx]           # 리샘플(저E 선호)
        for _ in range(rej):                                       # 물리 rejuvenation
            Y = Y - eta*gradE(Y) + 0.01*rng.standard_normal(Y.shape)
    return Y

Yg = phys_guide(Y0)
Ys = phys_smc(Y0)

def report(name, Y, v0ref=None):
    e = E_phys(Y)
    div = est_v0(Y).std()
    print(f"  {name:16s} | 평균 E_phys = {e.mean():8.3f} | 초기속도 다양성(std) = {div:.3f}")

print(f"T={T}, 중력G={G}, prior잡음σ={SIG}, 입자K={K}\n")
print(f"  {'참 물리궤적':16s} | 평균 E_phys = {E_phys(np.stack([ballistic(0,0.7)])).mean():8.3f} | (기준: 완전물리≈0)")
print()
report("raw prior(통계)", Y0)
report("물리가이드(A)", Yg)
report("물리-SMC(B)", Ys)
print(f"\n  (prior 초기속도 다양성 원본 std = {v0_true.std():.3f})")

print("\n해석:")
print(" - raw prior: E_phys 큼(잡음=비물리 jitter).")
print(" - 물리가이드/SMC: E_phys 급감(물리 법칙 주입) + 초기속도 다양성 보존(한 궤적 붕괴 아님)")
print("   → 통계 prior=다양성/그럴듯함, 물리 potential=법칙 → 재학습 없이 추론때 융합(#36).")
print(" - SMC(B)는 리샘플로 분포 교정까지(제안×potential 곱). 물리엔진처럼 딱딱하지 않음.")
