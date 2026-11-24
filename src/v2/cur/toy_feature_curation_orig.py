#!/usr/bin/env python3
"""
toy — RL 자기개선 피처 큐레이션 (#33)
=====================================
스트리밍 회귀에서 '유용한 피처'가 시간에 따라 바뀜(비정상). 에이전트가 피처를 능동 선택(큐레이션)
해 다운스트림 예측을 돕고, 전환 후 재적응하는가.
비교: all(전부) / fixed(고정랜덤k) / oracle(참 관련피처) / RL-curated(피처밴딧).
RL 큐레이터: 피처별 유용도 q 추적(다운스트림 가중치 EMA), top-k 선택+ε탐색, q 감쇠로 적응.
numpy only.
"""
import numpy as np
rng = np.random.default_rng(0)

P = 50            # 전체 피처(고차원일수록 all이 전환에 느림)
S = 3             # 관련 피처 수
K = 6             # 큐레이터가 고르는 수
T = 720           # 라운드
SWITCH = 120      # 비정상: SWITCH마다 관련 피처 교체(짧게=적응속도 중요)
NOISE = 0.3
lr = 0.05

def relevant_at(t):
    seg = t//SWITCH
    r = np.random.default_rng(1000+seg)
    return r.choice(P, S, replace=False), r.standard_normal(S)

def run(mode, k=K, eps=0.1, beta=0.2, decay=0.995):
    W = np.zeros(P)                    # 다운스트림 LMS 가중(선택된 것만 갱신)
    q = np.zeros(P)                    # 큐레이터 유용도
    fixedA = rng.choice(P, k, replace=False)
    errs = np.empty(T); rel_recall=np.empty(T)
    for t in range(T):
        rel, coef = relevant_at(t)
        x = rng.standard_normal(P)
        y = x[rel]@coef + NOISE*rng.standard_normal()
        # --- 피처 선택 ---
        if mode=="all":     A = np.arange(P)
        elif mode=="fixed": A = fixedA
        elif mode=="oracle":A = rel
        elif mode=="rl":
            top = np.argsort(-q)[:k]
            if rng.random()<eps:      # ε 탐색: 일부를 랜덤 교체
                top = top.copy(); top[rng.integers(k)] = rng.integers(P)
            A = np.unique(top)
        # --- 다운스트림 예측·갱신 (선택수로 스텝 정규화=공정) ---
        yhat = x[A]@W[A]; e = y-yhat
        W[A] += (lr/np.sqrt(len(A)))*e*x[A]
        errs[t] = e*e
        rel_recall[t] = len(set(A)&set(rel))/S
        # --- 큐레이터 업데이트(RL): 모든 피처의 잔차상관 |x_f·e| = 유용도(선택 안해도 계산) ---
        if mode=="rl":
            q = decay*q + beta*np.abs(x*e)              # 잔차와 상관 큰 피처 = 유용 → top-k
    return errs, rel_recall

print(f"P={P} 피처, 관련 S={S}, 큐레이터 k={K}, T={T}, {SWITCH}마다 관련피처 교체(비정상)\n")
res={}
for mode in ["all","fixed","oracle","rl"]:
    errs,rr = run(mode); res[mode]=(errs,rr)

# 구간별(전환 후 적응 포함) 평균 MSE
print(f"{'방법':8s} | 전체MSE | 각 구간(300)별 평균 MSE | 관련피처 recall(RL/oracle)")
seg_bounds=[(0,240),(240,480),(480,720)]
for mode in ["all","fixed","oracle","rl"]:
    errs,rr=res[mode]
    segs=[errs[a:b].mean() for a,b in seg_bounds]
    rrm = rr.mean()
    print(f"{mode:8s} | {errs.mean():7.3f} | "+" ".join(f"{s:6.3f}" for s in segs)+f" | {rrm:.2f}")

# 전환 직후 적응 속도: 각 전환 후 50라운드 평균 MSE
print(f"\n전환 직후(40라운드) 평균 MSE — 적응속도:")
for mode in ["all","fixed","oracle","rl"]:
    errs,_=res[mode]
    adapt=[errs[s:s+40].mean() for s in [120,240,360,480,600]]
    print(f"  {mode:8s}: "+" ".join(f"{a:.2f}" for a in adapt))

print("\n해석: RL-curated 가 all/fixed 보다 MSE 낮고 oracle 에 근접하며, 관련피처 recall↑,")
print("      전환 직후 재적응(MSE 회복)하면 → '능동 피처 큐레이션+비정상 적응' 실증(#33).")
