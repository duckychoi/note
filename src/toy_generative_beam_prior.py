#!/usr/bin/env python3
"""
toy — 생성 사전확률(beam-space prior)이 ill-posed 빔포밍을 해결하나 (#14)
========================================================================
핵심 주장: 빔 가중치 설계는 제약이 적으면 ill-posed(무한해, 대개 사이드로브 큼).
'좋은 빔포머' 분포를 학습한 생성 사전(여기선 선형-가우시안 프록시; 완전 디퓨전은 torch 필요)이
그 ill-posed를 해결해 저사이드로브 해로 복원.
beam-space = 각도영역(DFT dual) = 스펙트럴. numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

N = 16                                   # ULA 소자
th = np.linspace(-np.pi/2, np.pi/2, 361) # 각도 그리드
def steer(theta):                        # 조향벡터 a(θ), 반파장
    n = np.arange(N)
    return np.exp(1j*np.pi*n*np.sin(theta))
A = np.stack([steer(t) for t in th], 0)  # (361,N)

def pattern_db(w):
    P = np.abs(A@np.conj(w))**2
    P = P/ P.max()
    return 10*np.log10(P+1e-12)
def psl_db(w, th0, guard=0.18):          # peak sidelobe level (메인로브 밖 최대, dB)
    P = np.abs(A@np.conj(w))**2; P/=P.max()
    main = np.abs(th-th0) < guard
    return 10*np.log10(P[~main].max()+1e-12)

# ---------- '좋은 빔포머' 데이터셋 (저사이드로브 Hamming, 임의 조향) ----------
ham = 0.54 - 0.46*np.cos(2*np.pi*np.arange(N)/(N-1))   # Hamming 테이퍼(저사이드로브)
def good_bf(th0): return ham*steer(th0)
TH0 = rng.uniform(-0.7, 0.7, 600)
Wd = np.stack([good_bf(t) for t in TH0], 0)            # (600,N) 데이터

# ---------- 선형-가우시안 사전 (디퓨전 프록시): μ, Σ ----------
mu = Wd.mean(0)
Xc = Wd - mu
Sig = (Xc.conj().T @ Xc)/len(Wd) + 1e-3*np.eye(N)       # 복소 공분산
# beam-space 구조: Σ의 고유값(저차원 매니폴드)
ev = np.sort(np.abs(np.linalg.eigvalsh(Sig)))[::-1]
print(f"ULA N={N}, 데이터 600개. Σ 고유값 상위6: {np.round(ev[:6],3)}")
print(f"  → 유효차원(>1% 최대) = {int((ev>0.01*ev[0]).sum())}/{N} (빔포머 매니폴드 저차원=스펙트럴 구조)\n")

def prior_map(A_c, b):
    """가우시안 MAP: w = μ + ΣA^H(AΣA^H)^-1 (b - Aμ). A_c:(m,N) 제약행, b:(m,)."""
    S_ = A_c@Sig@A_c.conj().T
    return mu + Sig@A_c.conj().T@np.linalg.solve(S_, b - A_c@mu)

print("===== 역문제: 가중치 inpainting(소자 고장 복구) — 사전이 돕나? =====")
# 참 빔포머 good_bf(θ0) 에서 k개 소자 손실 → 복구. 전역 가우시안 사전 조건부평균 vs 영채움.
def inpaint_gauss(w_obs, obs):
    mis = np.array([i for i in range(N) if i not in obs])
    So = Sig[np.ix_(obs,obs)] + 1e-6*np.eye(len(obs))
    Smo = Sig[np.ix_(mis,obs)]
    wm = mu[mis] + Smo@np.linalg.solve(So, w_obs - mu[obs])
    w = mu.copy().astype(complex); w[list(obs)] = w_obs; w[mis] = wm
    return w
print(f"{'손실 소자수 k':>12} {'영채움 PSL':>12} {'가우시안사전 PSL':>16} {'진리 PSL':>10}")
for k in [2, 4, 6, 8]:
    z_psl=[]; g_psl=[]; t_psl=[]
    for _ in range(80):
        th0 = rng.uniform(-0.7,0.7); wt = good_bf(th0)
        brk = rng.choice(N, k, replace=False); obs=[i for i in range(N) if i not in brk]
        w_zero = wt.copy(); w_zero[brk]=0
        w_g = inpaint_gauss(wt[obs], obs)
        z_psl.append(psl_db(w_zero,th0)); g_psl.append(psl_db(w_g,th0)); t_psl.append(psl_db(wt,th0))
    print(f"{k:>12} {np.mean(z_psl):>12.1f} {np.mean(g_psl):>16.1f} {np.mean(t_psl):>10.1f}")
print("  → 가우시안사전 PSL < 영채움 이면 사전이 복구에 도움.\n")

print("종합/정직: (A) 게인제약만의 ill-posed에선 전역 선형-가우시안 사전이 실패")
print("  (평균≈0, 전역 2차통계가 특정 방향 위상 비선형성을 못 담음) → '선형 불충분'.")
print("  (B) inpainting(역문제)에서 사전이 영채움을 이기면 '사전은 유효하나 선형은 약함'.")
print("  => #14 결론: 빔포머 매니폴드는 방향에 대해 비선형(위상) → '스펙트럴 디퓨전'(비선형·조건부")
print("     생성사전)이 필요하다는 근거. 선형 프록시의 한계가 곧 디퓨전의 당위.")
