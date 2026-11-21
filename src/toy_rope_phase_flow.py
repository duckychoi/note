#!/usr/bin/env python3
"""
toy ① — RoPE-위상 조건 Flow Matching 생성 (비모수)
=================================================
주장(#34): RoPE(=회전 위상 임베딩)로 flow를 조건화하면 '명령한 위상'의 모션을 생성.
링(2D 원) = 순환 위상 다양체 = 리듬 위상의 최소모형. 각도 = 위상 φ.

- 데이터: x1 = R·[cosφ, sinφ] + noise, φ~U(0,2π).
- RoPE 조건화: 목표위상 φ* 를 회전임베딩 e(φ*)=(cosφ*,sinφ*)로, 데이터 가중
  K_phase = exp(κ·e(φ_i)·e(φ*)) = exp(κ·cos(φ_i−φ*))  (= RoPE 내적 = von Mises).
- 비모수 조건 FM 필드: v*(y,t;φ*) = Σ_i w_i (x1_i−x0_i),
  w_i ∝ K_space(y−x_t_i)·K_phase.  x0~N, x_t=(1−t)x0+t·x1.
- 생성: y0~N(0,I) → Euler 적분 dy/dt=v* → y1. 생성각도 vs 명령 φ* 오차.
- 비교: 조건(κ=8) vs 무조건(κ=0). numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

N = 1500
R = 2.0
phi_data = rng.uniform(0, 2*np.pi, N)
X1 = R*np.stack([np.cos(phi_data), np.sin(phi_data)], 1) + 0.08*rng.standard_normal((N, 2))
X0 = rng.standard_normal((N, 2))
U = X1 - X0
TS = np.linspace(0.05, 0.95, 20)
dt = 1.0/len(TS)
HS = 0.6   # 공간 커널 폭

def field(Y, t, phi_t, kappa):
    """Y:(B,2), t scalar, phi_t:(B,) 목표위상, kappa. → v*(Y,t;phi_t):(B,2)."""
    Xt = (1-t)*X0 + t*X1                                  # (N,2)
    d2 = ((Y[:, None, :] - Xt[None, :, :])**2).sum(-1)    # (B,N)
    Ks = np.exp(-0.5*d2/HS**2)
    Kp = np.exp(kappa*np.cos(phi_data[None, :] - phi_t[:, None]))  # (B,N) RoPE 위상 유사도
    w = Ks*Kp
    w = w/(w.sum(1, keepdims=True)+1e-12)
    return w @ U

def generate(phi_cmd, kappa):
    B = len(phi_cmd)
    Y = rng.standard_normal((B, 2))
    for t in TS:
        Y = Y + dt*field(Y, t, phi_cmd, kappa)
    return Y

def ang_err(Y, phi_cmd):
    a = np.arctan2(Y[:, 1], Y[:, 0])
    e = np.abs(np.angle(np.exp(1j*(a - phi_cmd))))   # 순환 각도오차 [0,π]
    return e

B = 300
phi_cmd = rng.uniform(0, 2*np.pi, B)

print("===== RoPE-위상 조건 flow 생성 (링=순환 위상) =====")
print(f"데이터 N={N}, 링반경 R={R}, 생성 B={B}, Euler {len(TS)}스텝\n")
print(f"{'설정':16s} {'평균 각도오차(rad)':>18s} {'<0.3rad 비율':>14s} {'평균 반경':>10s}")
for label, kappa in [("무조건 κ=0", 0.0), ("RoPE조건 κ=4", 4.0), ("RoPE조건 κ=8", 8.0)]:
    Y = generate(phi_cmd, kappa)
    e = ang_err(Y, phi_cmd)
    radius = np.sqrt((Y**2).sum(1)).mean()
    frac = (e < 0.3).mean()
    print(f"{label:16s} {e.mean():>18.3f} {frac:>13.2f} {radius:>10.2f}")

# 무조건 기준선: 각도가 명령과 무관하면 평균오차 ~ π/2 ≈ 1.57
print(f"\n참고: 각도가 명령과 완전 무관(무조건 이상)이면 평균 각도오차 ≈ π/2 = {np.pi/2:.2f} rad")
print("해석: κ↑(RoPE 위상조건 강화) → 생성각도가 명령 φ* 추종(오차↓, <0.3rad 비율↑),")
print("      무조건(κ=0)은 명령 무시(오차≈π/2). = RoPE 위상조건이 '리듬 위상'을 제어.")

# 추가: 특정 위상 4개 명령 → 생성 위치 확인
print("\n===== 특정 위상 명령 → 생성 위치 (κ=8) =====")
cmds = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
Y = generate(cmds, 8.0)
for c, y in zip(cmds, Y):
    print(f"  명령 {c/np.pi:.2f}π → 생성 ({y[0]:+.2f},{y[1]:+.2f}), 각도 {np.arctan2(y[1],y[0])/np.pi:+.2f}π")
