#!/usr/bin/env python3
"""
toy — LeWAM (#38): 행동조건부 다봉 세계모델 예측
================================================
LeWM 예측기는 결정론 MSE(ẑ'=pred(z,a)) → 다봉 미래를 평균(붕괴). 행동이 미래 분포를 바꾸는데도.
flow matching 예측기 p(z'|z,a) 로 바꾸면: 행동이 '분포를 조절'(중립=다봉, 편향행동=단봉)하는 걸 담음.
toy: 상태 s, 행동 a∈{-1,0,+1}. a=0 중립→z=s±Δ(좌우 분기), a=+1→s+Δ(우), a=-1→s-Δ(좌).
 - 결정론 pred = 커널회귀 E[z|s,a].
 - 생성형 pred = 비모수 조건부 flow p(z|s,a).
numpy only. (#37 생성형 JEPA를 '행동조건부'로 확장.)
"""
import numpy as np
rng = np.random.default_rng(0)
D = 2.0
def gen(n):
    s = rng.uniform(-2,2,n); a = rng.integers(-1,2,n)
    br = np.where(a==0, rng.integers(0,2,n)*2-1, a)     # a=0이면 랜덤 분기, 아니면 a가 분기 결정
    z = s + br*D + 0.18*rng.standard_normal(n)
    return s,a,z
N=4000; S,A,Z = gen(N)
Z0 = rng.standard_normal(N); U = Z-Z0
TS=np.linspace(0.1,0.9,6); dt=1/len(TS)

def kreg(sq,aq,h=0.3):                                   # 결정론 예측 E[z|s,a]
    w = np.exp(-0.5*((S[None,:]-sq[:,None])/h)**2)*(A[None,:]==aq[:,None])
    w/=w.sum(1,keepdims=True)+1e-12; return w@Z
def flow(sq,aq,hs=0.3,hy=0.4):                           # 생성형 예측 p(z|s,a)
    B=len(sq); y=rng.standard_normal(B)
    for t in TS:
        Zt=(1-t)*Z0+t*Z
        w=np.exp(-0.5*((S[None,:]-sq[:,None])/hs)**2)*(A[None,:]==aq[:,None])*np.exp(-0.5*((Zt[None,:]-y[:,None])/hy)**2)
        w/=w.sum(1,keepdims=True)+1e-12; y=y+dt*(w@U)
    return y

print(f"행동조건부 분기: a=0 중립→s±{D}(다봉), a=±1→편향(단봉). 테스트 s=0\n")
print("결정론 세계모델(회귀):")
for a in [-1,0,1]:
    d=kreg(np.array([0.0]),np.array([a]))[0]
    tag="← 갭(다봉 평균붕괴)" if a==0 else ""
    print(f"  a={a:+d} → ẑ={d:+.2f} {tag}")
print("\n생성형 세계모델(flow) 샘플분포 @ s=0:")
for a in [-1,0,1]:
    y=flow(np.zeros(400),np.full(400,a))
    hi=(np.abs(y-D)<0.6).mean(); lo=(np.abs(y+D)<0.6).mean(); gap=(np.abs(y)<0.6).mean()
    print(f"  a={a:+d} → 우(+{D}) {hi:.2f} / 좌(-{D}) {lo:.2f} / 갭(0) {gap:.2f}")
print("\n해석: 결정론은 중립(a=0) 다봉을 평균→갭(무효). flow는 a=0서 좌우 둘 다, a=±1서")
print("  해당쪽 단봉으로 정확히 이동 → '행동이 미래 분포를 조절'하는 세계모델(#38 LeWAM).")
print("  LeWM의 SIGReg가 latent을 N(0,I)로 만들어줌 = flow의 base 분포가 공짜 ⇒ 궁합 최적.")
