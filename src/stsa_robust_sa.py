#!/usr/bin/env python3
"""
STSA toy 검증: 표준 1차 확률근사(SA) vs 슈퍼트위스팅 SA(STW)
scalar value/mean tracking. 세 시나리오에서 RMSE 비교.
  A. 정상(stationary)      — target 고정, 제로평균 노이즈
  B. 램프 드리프트(ramp)   — target가 일정 속도로 이동(비정상)
  C. 급변 스텝(step)       — target가 중간에 점프
결론은 숫자로 말한다. 공정 비교 위해 여러 게인 설정 + 다수 시드 평균.
"""
import numpy as np

rng_master = np.random.default_rng(0)
T = 3000
N_SEED = 400
SIGMA = 1.0

def run(update, target_fn, seed, dist_fn=None, **p):
    rng = np.random.default_rng(seed)
    th = 0.0
    u = 0.0
    err2 = np.empty(T)
    for t in range(T):
        tgt = target_fn(t)
        dist = 0.0 if dist_fn is None else dist_fn(t)   # 동역학(파라미터) 외란
        z = tgt + rng.normal(0, SIGMA)          # 노이즈 섞인 관측
        d = z - th                               # innovation
        if update == "sa_const":
            th += p["a"] * d + dist
        elif update == "sa_dec":
            a = 1.0 / (t + 1) ** p.get("pw", 1.0)
            th += a * d + dist
        elif update == "stw":
            u += p["a"] * p["k2"] * np.sign(d)
            th += p["a"] * (p["k1"] * np.sqrt(abs(d)) * np.sign(d) + u) + dist
        err2[t] = (th - tgt) ** 2
    return err2

def sweep(update, target_fn, grid, dist_fn=None):
    """게인 그리드에서 최종구간 RMSE 최소 설정 선택(공정 튜닝)."""
    best = None
    for p in grid:
        acc = np.zeros(T)
        for s in range(N_SEED):
            acc += run(update, target_fn, seed=1000 + s, dist_fn=dist_fn, **p)
        rmse = np.sqrt(acc / N_SEED)
        tail = rmse[int(0.8 * T):].mean()         # 후반 20% 평균 RMSE
        if best is None or tail < best[0]:
            best = (tail, p, rmse)
    return best  # (tail_rmse, params, rmse_curve)

# 시나리오 정의: (target_fn, dist_fn)
scen = {
    "A_stationary":  (lambda t: 2.0,                               None),
    "B_ramp":        (lambda t: 2.0 + 0.004 * t,                   None),
    "C_step":        (lambda t: 2.0 if t < T // 2 else 6.0,        None),
    "D_const_dist":  (lambda t: 2.0,          lambda t: 0.05),               # 상수 외란(동역학)
    "E_sine_dist":   (lambda t: 2.0,          lambda t: 0.05 * np.sin(0.01 * t)),  # 느린 정현 외란
}

# 게인 그리드
grid_sa_const = [{"a": a} for a in (0.02, 0.05, 0.1, 0.2)]
grid_sa_dec   = [{"pw": pw} for pw in (0.6, 0.8, 1.0)]
grid_stw      = [{"a": a, "k1": k1, "k2": k2}
                 for a in (0.05, 0.1, 0.2)
                 for k1 in (0.5, 1.0, 1.5)
                 for k2 in (0.2, 0.5, 1.0)]

methods = {
    "SA(const α)": ("sa_const", grid_sa_const),
    "SA(1/t 감쇠)": ("sa_dec",   grid_sa_dec),
    "STW(제안)":    ("stw",      grid_stw),
}

def ckpt_rmse(curve):
    idx = [49, 199, 499, 999, 1999, 2999]
    return [curve[i] for i in idx]

print(f"T={T}, seeds={N_SEED}, σ={SIGMA}\n")
summary = {}
for sname, (tfn, dfn) in scen.items():
    print(f"===== 시나리오 {sname} =====")
    print(f"{'method':14s} {'best gains':28s} | RMSE@50  200   500  1000  2000  3000 | tail")
    summary[sname] = {}
    for mname, (upd, grid) in methods.items():
        tail, p, curve = sweep(upd, tfn, grid, dist_fn=dfn)
        c = ckpt_rmse(curve)
        gains = ",".join(f"{k}={v}" for k, v in p.items())
        print(f"{mname:14s} {gains:28s} | " +
              " ".join(f"{x:5.3f}" for x in c) + f" | {tail:5.3f}")
        summary[sname][mname] = tail
    print()

print("===== 요약: 후반 RMSE (낮을수록 좋음) =====")
print(f"{'scenario':16s} {'SA(const)':>10s} {'SA(1/t)':>10s} {'STW':>10s}  승자")
for sname in scen:
    s = summary[sname]
    win = min(s, key=s.get)
    print(f"{sname:16s} {s['SA(const α)']:10.3f} {s['SA(1/t 감쇠)']:10.3f} {s['STW(제안)']:10.3f}  -> {win}")
