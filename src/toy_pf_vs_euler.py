#!/usr/bin/env python3
"""
toy — 저 NFE에서 PF(SMC)가 Euler ODE보다 정확한가?
===================================================
무조건 생성에서 확률플로우를 PF로 '대체'. 폐형 선형보간 플로우(GMM) 사용.
 p_0=N(0,I), p_1=GMM.  x_t=(1-t)x0+t x1 → p_t, score, v*(x,t) 모두 폐형.
 - Euler: dx=v*·h, 결정론 N스텝.
 - SMC  : dx=v*·h+노이즈, 매 스텝 '알려진 p_t' 로 가중(bootstrap)+리샘플 → 바이어스 교정.
동일 NFE-depth N 에서 분포오차(모드질량·평균·에너지거리) 비교.
numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

# ---------- GMM 타깃 ----------
MU = np.array([[-2.5,0.0],[2.5,0.0],[0.0,3.2]]); S0 = 0.5; W = np.array([0.6,0.3,0.1])
def st2(t): return (1-t)**2 + (t*S0)**2
def p_t(X, t):
    s2 = st2(t)
    d = X[:,None,:]-t*MU[None,:,:]
    comp = W[None,:]*np.exp(-0.5*(d**2).sum(-1)/s2)/(2*np.pi*s2)
    return comp.sum(1), comp            # 밀도, 성분별
def vstar(X, t):
    s2 = st2(t); _, comp = p_t(X,t); r = comp/(comp.sum(1,keepdims=True)+1e-30)  # 책임도 (B,3)
    coef = (t*S0**2 - (1-t))/s2
    # v*_k = μ_k + coef*(x - t μ_k)
    xk = X[:,None,:] - t*MU[None,:,:]
    vk = MU[None,:,:] + coef*xk          # (B,3,2)
    return (r[:,:,None]*vk).sum(1)

def score_t(X, t):                        # ∇log p_t (폐형 GMM score) — rejuvenation용
    s2 = st2(t); _, comp = p_t(X,t); r = comp/(comp.sum(1,keepdims=True)+1e-30)
    xk = X[:,None,:] - t*MU[None,:,:]
    return (r[:,:,None]*(-xk/s2)).sum(1)

TS = None  # 스텝은 아래에서

def euler(N):
    global_ts = np.linspace(0,1,N+1); h = 1.0/N
    X = rng.standard_normal((4000,2))
    for k in range(N):
        X = X + h*vstar(X, global_ts[k])
    return X, np.ones(len(X))

def smc(N, K=1500, tau=0.15, rejuv=0, eps=0.012):
    ts = np.linspace(0,1,N+1); h = 1.0/N
    X = rng.standard_normal((K,2))
    for k in range(N):
        t0, t1 = ts[k], ts[k+1]
        drift = vstar(X, t0)
        Xprev = X.copy()
        X = X + h*drift + tau*np.sqrt(h)*rng.standard_normal((K,2))   # 노이즈 탐색
        # bootstrap 가중: 타깃 p_{t1} / 제안(이전입자의 전이커널 KDE)
        mean = Xprev + h*drift                                        # (K,2) 전이 평균
        var = (tau**2)*h + 1e-9
        d2 = ((X[:,None,:]-mean[None,:,:])**2).sum(-1)               # (K,K)
        q = np.exp(-0.5*d2/var).mean(1)/(2*np.pi*var)                # 제안밀도
        pt,_ = p_t(X, t1)
        w = pt/(q+1e-30); w /= w.sum()
        ess = 1.0/(w**2).sum()
        if ess < 0.5*K:                                              # ESS 낮으면 리샘플
            idx = rng.choice(K, K, p=w); X = X[idx]; w = np.ones(K)/K
            # rejuvenation: 알려진 p_{t1} 을 타깃으로 ULA move (중복입자 다양성 복원)
            for _ in range(rejuv):
                X = X + eps*score_t(X,t1) + np.sqrt(2*eps)*rng.standard_normal((K,2))
    return X, np.ones(len(X))

# ---------- 지표 ----------
TRUE = MU[rng.choice(3,3000,p=W)] + S0*rng.standard_normal((3000,2))
def mode_mass(X):
    m = np.zeros(3)
    for i,mu in enumerate(MU): m[i]=(((X-mu)**2).sum(1)<1.2**2).mean()
    return m/(m.sum()+1e-12)
def energy_dist(A,B,n=800):
    a=A[rng.choice(len(A),min(n,len(A)),replace=False)]
    b=B[rng.choice(len(B),min(n,len(B)),replace=False)]
    def pd(X,Y): return np.sqrt(((X[:,None,:]-Y[None,:,:])**2).sum(-1)+1e-12).mean()
    return 2*pd(a,b)-pd(a,a)-pd(b,b)

print("저 NFE-depth 에서 Euler ODE vs SMC(PF) — 무조건 GMM 생성\n")
print(f"목표 모드질량 W={W}\n")
print(f"{'N(스텝)':>7} | {'방법':5s} | {'모드질량':>22s} | {'평균오차':>8s} | {'에너지거리':>9s}")
for N in [3,5,8,12,20]:
    for name, fn in [("Euler", lambda: euler(N)),
                     ("SMC", lambda: smc(N)),
                     ("SMC+rej", lambda: smc(N, rejuv=4))]:
        X,_ = fn()
        mm = mode_mass(X); me = np.abs(mm-W).sum()
        ed = energy_dist(X, TRUE)
        print(f"{N:>7} | {name:5s} | {str(np.round(mm,2)):>22s} | {me:>8.3f} | {ed:>9.4f}")
    print()
print("해석:")
print(" - 저 NFE(N=3,5): SMC/SMC+rej 가 Euler보다 에너지거리 작으면 '리샘플이 이산화 바이어스 교정' 확인.")
print(" - 고 NFE(N>=8): 순수 SMC는 경로퇴화로 열화. SMC+rej(알려진 p_t로 ULA move)가 소수모드를")
print("   보존해 열화를 완화하면 → 'rejuvenation이 enabling 기법' 실증(저 NFE 우위 유지 + 고 NFE 안정).")
