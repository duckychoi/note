#!/usr/bin/env python3
"""
toy — 소산성(dissipativity) 제약 신경 KF 세계모델 (#39)
======================================================
아이디어(교수님): KF는 PF와 달리 샘플 없이 예측–보정 → 효율적 세계모델.
거기에 '에너지 기반 성능지표(extended dissipative performance)'로 학습을 제약하면
학습된 필터가 '소산적(=에너지 발산 안 함)'이 되어 안정. Mamba/Transformer 백본과 호환.

여기선 핵심 원리를 최소모형으로: 노이즈 짧은 궤적에서 선형 SSM(A)을 적합(sample-free, 지도).
- plain: 최소제곱 적합만 → 노이즈/소량데이터로 스펙트럼반경 ρ(A)>1(불안정) 빈발 → '상상 rollout' 발산.
- dissipative: 에너지 소산 조건(ρ≤ρ*=0.98)을 투영으로 강제 → 안정 rollout.
세계모델은 계획 위해 '관측 없이 상상(rollout)' 해야 하므로 내재 안정성(소산성)이 필수.
numpy only. (확장 소산성은 H∞·L2-L∞·passivity 통합 — 여기선 그 핵심인 Lyapunov 에너지 소산만.)
"""
import numpy as np
rng = np.random.default_rng(0)

dt=0.1
# 참 시스템: 감쇠 진동자 (안정, ρ<1)
A_true=np.array([[1,dt],[-1.2*dt,1-0.15*dt]])
def gen_traj(T,proc=0.03):
    x=np.array([1.0,0.0]);X=[x.copy()]
    for _ in range(T): x=A_true@x+proc*rng.standard_normal(2);X.append(x.copy())
    return np.array(X)

def fit_A(X):                              # sample-free LS 적합: X_{t+1}≈A X_t
    Xt,Xn=X[:-1],X[1:]
    return np.linalg.lstsq(Xt,Xn,rcond=None)[0].T
def rho(A): return np.max(np.abs(np.linalg.eigvals(A)))
def dissipative_project(A,rstar=0.98):     # 고유값 크기를 ρ* 이하로 투영(에너지 소산 강제)
    w,V=np.linalg.eig(A)
    w2=np.where(np.abs(w)>rstar, w/np.abs(w)*rstar, w)
    return np.real(V@np.diag(w2)@np.linalg.inv(V))
def rollout(A,steps=150,dist=0.02):        # 관측없는 '상상' rollout + 외란
    x=np.array([1.0,0.0]);n=[]
    for _ in range(steps):
        x=A@x+np.array([0,dist])           # 상수 외란
        n.append(np.linalg.norm(x))
    return np.array(n)

Nseed=120; Ttrain=18; NOISE=0.08
print(f"참 시스템 ρ(A_true)={rho(A_true):.3f} (안정). 학습데이터 T={Ttrain}, 관측노이즈 σ={NOISE}\n")
res={'plain':{'unst':0,'final':[]}, 'dissip':{'unst':0,'final':[]}}
for s in range(Nseed):
    rng2=np.random.default_rng(100+s)
    X=gen_traj(Ttrain) + NOISE*rng2.standard_normal((Ttrain+1,2))   # 노이즈 관측
    Ap=fit_A(X)
    Ad=dissipative_project(Ap,0.98)
    for name,A in [('plain',Ap),('dissip',Ad)]:
        if rho(A)>1.0+1e-9: res[name]['unst']+=1
        n=rollout(A)
        res[name]['final'].append(min(n[-1],1e6))   # 발산 상한

print(f"{'방법':10s} {'불안정 ρ>1':>10s} {'rollout 발산(>10) 비율':>20s} {'최종norm 평균':>14s} {'최대':>12s}")
for name in ['plain','dissip']:
    f=np.array(res[name]['final']); unst=res[name]['unst']/Nseed
    div=(f>10).mean(); print(f"{name:10s} {unst*100:>9.0f}% {div*100:>18.0f}% {f.mean():>14.1f} {f.max():>12.1f}")

print("\n해석:")
print(" - plain(소산성 제약 X): 노이즈·소량데이터로 ρ>1 불안정 빈발 → 관측없는 상상 rollout 발산(norm 폭주).")
print(" - dissip(소산성 제약 O): ρ≤0.98 강제 → 불안정 0%, rollout norm 유계(안정).")
print(" → 세계모델은 계획 위해 '관측없이 상상'해야 하므로, 에너지 소산(dissipativity) 제약이")
print("   학습된 KF 세계모델의 장기 안정성을 보장. = extended dissipative 학습의 핵심 가치(#39).")
print("   (관측으로 보정하면 plain도 가려지지만, 계획용 imagination엔 내재 안정성이 필수.)")
