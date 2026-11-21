#!/usr/bin/env python3
"""
toy — §4.1: rectified/OT 결합이 C(q)(기약 결합분산)를 줄여 divergence 신호를 개선하나
=====================================================================================
독립결합 vs OT(근사:greedy NN)결합으로 v* 구성 후:
  (A) 데이터 CFM 잔차(직선성 지표, 낮을수록 C↓/path 직선)
  (B) corr(FM잔차 R, KL) 가 OT에서 더 높은가 (divergence 신호 선명도)
numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

MU0 = np.array([[-2.5,0.0],[2.5,0.0],[0.0,3.2]]); SIG0=0.55; W0=np.array([0.6,0.3,0.1])
def gmm_logpdf(X,MU,SIG,W):
    d=X[:,None,:]-MU[None,:,:]; e=-0.5*(d**2).sum(-1)/SIG**2-np.log(2*np.pi*SIG**2)
    a=e+np.log(W)[None,:]; m=a.max(1,keepdims=True); return m.squeeze(1)+np.log(np.exp(a-m).sum(1))
def gmm_sample(n,MU,SIG,W):
    k=rng.choice(len(W),n,p=W); return MU[k]+SIG*rng.standard_normal((n,2))

N=500
X1=gmm_sample(N,MU0,SIG0,W0); X0=rng.standard_normal((N,2))
TS=np.linspace(0.1,0.9,5); HK=0.4

def greedy_ot(A,B):
    """A[i] 를 서로 다른 B[j] 에 greedy 최근접 매칭 → match[i]=j."""
    n=len(A); D=((A[:,None,:]-B[None,:,:])**2).sum(-1)
    order=np.argsort(D,axis=None); match=-np.ones(n,int); usedB=np.zeros(n,bool); cnt=0
    for idx in order:
        i=idx//n; j=idx%n
        if match[i]<0 and not usedB[j]:
            match[i]=j; usedB[j]=True; cnt+=1
            if cnt==n: break
    return match

# 두 결합
X0_indep = X0                         # 독립
match = greedy_ot(X1, X0)
X0_ot = X0[match]                     # OT(근사)

def make_vstar(X0p):
    U=X1-X0p
    def v(Y,t):
        Xt=(1-t)*X0p+t*X1
        d2=((Y[:,None,:]-Xt[None,:,:])**2).sum(-1); w=np.exp(-0.5*d2/HK**2)
        w/= (w.sum(1,keepdims=True)+1e-12); return w@U
    return v, U

def data_cfm_residual(X0p):
    """데이터쌍에서 ||v*(x_t,t)-(x1-x0)||^2 평균 = 직선성(낮을수록 C↓)."""
    v,U=make_vstar(X0p); acc=0.0
    for t in TS:
        Xt=(1-t)*X0p+t*X1; acc+=((v(Xt,t)-U)**2).sum(1).mean()
    return acc/len(TS)

def fm_residual(q_samples, X0p):
    v,_=make_vstar(X0p); Y=q_samples; acc=np.zeros(len(Y)); x0s=rng.standard_normal((4,2))
    for t in TS:
        for x0 in x0s:
            Yt=(1-t)*x0+t*Y; acc+=((v(Yt,t)-(Y-x0))**2).sum(1)
    return acc.mean()/(len(TS)*4)

def kl_mc(MU,SIG,W):
    Xq=gmm_sample(3000,MU,SIG,W); return (gmm_logpdf(Xq,MU,SIG,W)-gmm_logpdf(Xq,MU0,SIG0,W0)).mean()
def spearman(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra-=ra.mean(); rb-=rb.mean(); return (ra*rb).sum()/(np.sqrt((ra**2).sum()*(rb**2).sum())+1e-12)

fam=[("base",MU0,SIG0,W0)]
for s in [0.5,1.0,2.0]: fam.append((f"shift{s}",MU0+np.array([s,0.]),SIG0,W0))
for c in [1.3,1.7,2.5]: fam.append((f"var{c}",MU0,SIG0*c,W0))
for w in [[0.34,0.33,0.33],[0.8,0.15,0.05],[0.1,0.1,0.8]]: fam.append(("wtilt",MU0,SIG0,np.array(w)))

print("§4.1 검증: 독립결합 vs OT(근사)결합\n")
print(f"{'결합':10s} {'(A)데이터CFM잔차(직선성)':>24s} {'base R(≈C floor)':>18s} {'(B)corr(R,KL)':>14s}")
for label,X0p in [("독립",X0_indep),("OT근사",X0_ot)]:
    A=data_cfm_residual(X0p)
    KL=[]; R=[]
    for tag,mu,sig,w in fam:
        KL.append(kl_mc(mu,sig,w)); R.append(fm_residual(gmm_sample(400,mu,sig,w),X0p))
    KL=np.array(KL); R=np.array(R)
    print(f"{label:10s} {A:>24.3f} {R[0]:>18.3f} {spearman(R,KL):>14.3f}")
print("\n해석: OT결합이 데이터CFM잔차(직선성)↓ + base R(≈C floor)↓ 이면 C 축소 성공,")
print("      그 결과 corr(R,KL)↑ 이면 'C 줄이면 divergence 신호 선명' (§4.1) 실증.")
