#!/usr/bin/env python3
"""
toy — 생성형 JEPA (#37): 결정론 예측 vs flow-matching 조건부 예측
================================================================
논제: 순수 JEPA는 다음 latent를 '하나로' 예측(회귀) → 다봉 미래를 평균내 '갭'에 떨어짐(붕괴).
flow matching으로 '조건부 미래분포'를 생성하면 다봉성 보존.
toy: 미래가 분기(bimodal)하는 최소 환경. context c → future z = c ± Δ (좌/우 분기) + 잡음.
 - 결정론 JEPA = 커널회귀 평균 E[z|c] ≈ c (모드 사이 갭 = 무효).
 - 생성형 JEPA = 비모수 조건부 flow p(z|c) 샘플 → 양쪽 모드 복원.
지표: 예측이 유효 모드(c±Δ) 근처인가.
numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

DELTA = 2.0
def gen_data(n):
    c = rng.uniform(-2, 2, n)
    s = rng.integers(0, 2, n)*2 - 1          # ±1 분기
    z = c + s*DELTA + 0.18*rng.standard_normal(n)
    return c, z
N = 3000
C, Z = gen_data(N)

def kreg_mean(c_query, h=0.3):               # 결정론 JEPA: Nadaraya-Watson 평균
    w = np.exp(-0.5*((C[None,:]-c_query[:,None])/h)**2)
    w /= w.sum(1, keepdims=True)+1e-12
    return w @ Z

# 생성형 JEPA: 조건부 flow. z_t=(1-t)z0+t z, z0~N(0,1). v*(y,t;c)=E[z-z0|y_t,c].
Z0 = rng.standard_normal(N)
U = Z - Z0
TS = np.linspace(0.1, 0.9, 6); dt = 1.0/len(TS)
def flow_sample(c_query, hc=0.25, hy=0.4):
    B = len(c_query); y = rng.standard_normal(B)
    for t in TS:
        Zt = (1-t)*Z0 + t*Z
        wc = np.exp(-0.5*((C[None,:]-c_query[:,None])/hc)**2)      # (B,N) context 유사
        wy = np.exp(-0.5*((Zt[None,:]-y[:,None])/hy)**2)           # 상태 유사
        w = wc*wy; w /= w.sum(1, keepdims=True)+1e-12
        v = w @ U
        y = y + dt*v
    return y

# 평가
cq = np.array([-1.0, 0.0, 1.0])
print(f"분기 미래: z = c ± {DELTA} (좌/우). context c = {cq}\n")
det = kreg_mean(cq)
print("결정론 JEPA(회귀 평균) 예측 ẑ:")
for c,d in zip(cq,det):
    dist = min(abs(d-(c+DELTA)), abs(d-(c-DELTA)))
    print(f"  c={c:+.1f} → ẑ={d:+.2f} (최근접 모드까지 {dist:.2f}) {'← 갭(무효)' if dist>0.8 else ''}")

print("\n생성형 JEPA(flow) 샘플 분포:")
for c in cq:
    s = flow_sample(np.full(400, c))
    near_hi = (np.abs(s-(c+DELTA))<0.6).mean()
    near_lo = (np.abs(s-(c-DELTA))<0.6).mean()
    gap = (np.abs(s-c)<0.6).mean()
    print(f"  c={c:+.1f} → 우모드(c+2) {near_hi:.2f} / 좌모드(c-2) {near_lo:.2f} / 갭(c) {gap:.2f}")

print("\n해석:")
print(" - 결정론 JEPA: 양 모드를 평균 → c 근처(모드 사이 '갭')에 떨어짐 = 무효 예측(붕괴).")
print(" - 생성형 JEPA(flow): 좌·우 모드를 둘 다 복원(각 ~0.5), 갭엔 거의 안 감 → 다봉 미래 보존.")
print(" → '표현공간에서 flow로 조건부 미래분포 예측' 이 JEPA 모드평균 약점을 해결(#37).")
