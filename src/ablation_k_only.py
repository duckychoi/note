#!/usr/bin/env python3
"""절제: ε=0.6 고정, k 만 다름. (a) k=0 고정  (b) 적분제어 k, H*=0.9 (제어기가 실제로 개입하도록 현재 엔트로피보다 높게).
질문: 상한 완화 k 가 '유효 답이 여럿일 때' 엔트로피를 실제로 끌어올리는가. 다중정답·필터 끔 무대(demo Part 2와 동일)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mai_grpo as M
import demo_entropy_control as D

_orig = M.MAIConfig
def run_variant(variant, seed):
    def cfg_factory(*a, **kw):
        c = _orig()
        if variant == "k0":
            c.k_max = 0.0; c.delta = 0.0
        else:
            c.target_entropy = 0.9
        return c
    D.M.MAIConfig = cfg_factory
    try:
        return D.run("adaptive", seed, multi=True, filters=False, iters=300)
    finally:
        D.M.MAIConfig = _orig

for v in ["k0", "kctrl_H0.9"]:
    L = [run_variant(v, s) for s in range(5)]
    H = np.array([[x[1] for x in l] for l in L]); A = np.array([[x[2] for x in l] for l in L]); K = np.array([[x[3] for x in l] for l in L])
    print(f"[{v}] 최종 H 평균 {H[:,-1].mean():.3f} (시드별 {np.round(H[:,-1],3)})  둘다유지 {A[:,-1].mean():.3f}  최종 k 평균 {K[:,-1].mean():.2f}  최대 k {K.max():.2f}")
    print(f"      iter50 H {H[:,49].mean():.3f}, iter100 H {H[:,99].mean():.3f}")
d = None

# --- 상한 클립이 실제로 걸리는 조건: 학습률 10배, 클립 발동률 기록 ---
_orig_loss = M.grpo_loss
stats_log = []
def logging_loss(*a, **kw):
    out = _orig_loss(*a, **kw); stats_log.append(out[2]); return out
M.grpo_loss = logging_loss
D.LR = 5.0
print("\n[학습률 10배]")
for v in ["k0", "kctrl_H0.9"]:
    stats_log.clear()
    L = [run_variant(v, s) for s in range(5)]
    H = np.array([[x[1] for x in l] for l in L]); A = np.array([[x[2] for x in l] for l in L])
    ch = np.mean([s["clipfrac_high"] for s in stats_log]); oc = np.mean([s["outer_clipfrac"] for s in stats_log])
    print(f"[{v}] 최종 H {H[:,-1].mean():.3f} (시드별 {np.round(H[:,-1],3)})  둘다유지 {A[:,-1].mean():.3f}  상한클립률 {ch:.4f}  외부클립률 {oc:.5f}")
M.grpo_loss = _orig_loss
