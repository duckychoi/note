#!/usr/bin/env python3
"""
적응게인 STW — "깨끗하면 1/t-최적, 외란 있으면 STW-강인" 하이브리드 검증
=======================================================================
고정 STW 는 정상환경에서 노이즈 floor(≈0.16) 때문에 1/t 평균에 짐.
step α_t 를 Robbins-Monro 스케줄로 감쇠시키면 floor 가 줄어 정상환경도 잡히는지,
그러면서 상수외란 제거(강인성)는 유지되는지 본다.

비교 대상:
  sa_dec    : θ += (1/t^pw)(z-θ)                     # 정상환경 최적 베이스라인
  stw_fixed : 고정 게인 슈퍼트위스팅                   # 강인하지만 floor
  stw_adapt : α_t = α0/(t+1)^q 로 감쇠하는 STW        # 제안 하이브리드
시나리오 A(정상) / D(상수 외란) 에서 후반 RMSE.
"""
import numpy as np

T = 3000
N_SEED = 400
SIGMA = 1.0

def run(update, target_fn, seed, dist_fn=None, **p):
    rng = np.random.default_rng(seed)
    th = 0.0; u = 0.0
    err2 = np.empty(T)
    for t in range(T):
        tgt = target_fn(t)
        dist = 0.0 if dist_fn is None else dist_fn(t)
        z = tgt + rng.normal(0, SIGMA)
        d = z - th
        if update == "sa_dec":
            a = 1.0 / (t + 1) ** p["pw"]
            th += a * d + dist
        elif update == "stw_fixed":
            u += p["a"] * p["k2"] * np.sign(d)
            th += p["a"] * (p["k1"] * np.sqrt(abs(d)) * np.sign(d) + u) + dist
        elif update == "stw_adapt":
            a = p["a0"] / (t + 1) ** p["q"]
            u += a * p["k2"] * np.sign(d)
            th += a * (p["k1"] * np.sqrt(abs(d)) * np.sign(d) + u) + dist
        err2[t] = (th - tgt) ** 2
    return err2

def sweep(update, target_fn, grid, dist_fn=None):
    best = None
    for p in grid:
        acc = np.zeros(T)
        for s in range(N_SEED):
            acc += run(update, target_fn, seed=2000 + s, dist_fn=dist_fn, **p)
        rmse = np.sqrt(acc / N_SEED)
        tail = rmse[int(0.8 * T):].mean()
        if best is None or tail < best[0]:
            best = (tail, p, rmse)
    return best

scen = {
    "A_stationary": (lambda t: 2.0, None),
    "D_const_dist": (lambda t: 2.0, lambda t: 0.05),
}

grids = {
    "sa_dec":    [{"pw": pw} for pw in (0.6, 0.8, 1.0)],
    "stw_fixed": [{"a": a, "k1": k1, "k2": k2}
                  for a in (0.05, 0.1) for k1 in (0.5, 1.0) for k2 in (0.2, 0.5)],
    "stw_adapt": [{"a0": a0, "q": q, "k1": k1, "k2": k2}
                  for a0 in (0.2, 0.4) for q in (0.3, 0.5)
                  for k1 in (0.5, 1.0) for k2 in (0.2, 0.5)],
}
labels = {"sa_dec": "SA(1/t 감쇠)", "stw_fixed": "STW(고정)", "stw_adapt": "STW(적응 α_t)"}

print(f"T={T}, seeds={N_SEED}, σ={SIGMA}\n")
summary = {}
for sname, (tfn, dfn) in scen.items():
    print(f"===== {sname} =====")
    print(f"{'method':16s} {'best gains':34s} | tail RMSE")
    summary[sname] = {}
    for upd in ("sa_dec", "stw_fixed", "stw_adapt"):
        tail, p, _ = sweep(upd, tfn, grids[upd], dist_fn=dfn)
        gains = ",".join(f"{k}={v}" for k, v in p.items())
        print(f"{labels[upd]:16s} {gains:34s} | {tail:5.3f}")
        summary[sname][labels[upd]] = tail
    print()

print("===== 요약 (후반 RMSE, 낮을수록 좋음) =====")
print(f"{'method':16s} {'A_정상':>8s} {'D_외란':>8s}   비고")
for upd in ("sa_dec", "stw_fixed", "stw_adapt"):
    lb = labels[upd]
    a = summary['A_stationary'][lb]; d = summary['D_const_dist'][lb]
    note = ""
    if lb == "SA(1/t 감쇠)": note = "정상 최강, 외란 취약"
    elif lb == "STW(고정)":   note = "외란 강인, 정상 약함"
    else:                     note = "<- 둘 다 잡는가?"
    print(f"{lb:16s} {a:8.3f} {d:8.3f}   {note}")
