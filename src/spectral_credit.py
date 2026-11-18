#!/usr/bin/env python3
"""
스펙트럴 신용할당 (Spectral Credit Assignment)
================================================
리턴 추정기 = 보상열에 걸리는 LTI 필터. 그 '임펄스응답' h_k 를 직접 설계한다.

핵심 관계:
  복합리턴  G^w = Σ_n w_n G^(n)   (Σ w_n = 1, G^(n)=n-step 리턴)
  는 보상 r_{t+k} 에 가중치   h_k = Σ_{n>=k} w_n   를 거는 것과 동치.
  → h_k 가 '적격흔적 임펄스응답'.  DTFT(h) = 신용할당의 주파수응답.

특수경우:
  n-step(w_N=1)      → h_k = 1 (k<=N), 0 (k>N)   = 사각 FIR
  TD(λ)(w_n=(1-λ)λ^{n-1}) → h_k = λ^{k-1}         = 기하 IIR (= 표준 eligibility)
  설계형 FIR         → h 를 원하는 주파수응답으로 직접 설계 (Hann 저역 등)

실험: 19-state random walk 정책평가. 서로 다른 h 설계의 RMSE(참값 대비) 비교.
가치함수가 '매끈한(저역)' 구조라 저역통과형 신용할당이 바이어스-분산을 더 잘 잡는지 본다.
stdlib + numpy.
"""
import numpy as np

# ---------- 환경: 19-state random walk ----------
N = 19                      # 비말단 상태 1..19, 말단 0/20
GAMMA = 1.0
V_TRUE = np.array([(s - (N + 1) / 2) / ((N + 1) / 2) for s in range(1, N + 1)])  # 선형 -0.9..0.9
KMAX = 24                   # 리턴 탭 최대 길이

def episode(rng):
    """경로 생성: (상태열, 보상열). 좌말단 보상 -1, 우말단 +1, 그 외 0."""
    s = (N + 1) // 2
    states, rewards = [], []
    while True:
        states.append(s)
        s2 = s + (1 if rng.random() < 0.5 else -1)
        if s2 == 0:
            rewards.append(-1.0); break
        elif s2 == N + 1:
            rewards.append(1.0); break
        else:
            rewards.append(0.0); s = s2
    return states, rewards

# ---------- 가중치 설계 w_n (n=1..KMAX), Σ=1 ----------
def w_nstep(n):
    w = np.zeros(KMAX); w[min(n, KMAX) - 1] = 1.0; return w

def w_tdlambda(lam):
    n = np.arange(1, KMAX + 1)
    w = (1 - lam) * lam ** (n - 1); w[-1] += lam ** KMAX  # 꼬리 보정
    return w / w.sum()

def w_fir_rect(K):
    w = np.zeros(KMAX); w[:K] = 1.0 / K; return w

def w_fir_hann(K):
    # Hann 창 저역통과형 FIR (중간 n 강조 → 매끈한 신용)
    n = np.arange(1, K + 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * (n - 0.5) / K)
    w = np.zeros(KMAX); w[:K] = win / win.sum(); return w

def h_from_w(w):
    """보상 임펄스응답 h_k = Σ_{n>=k} w_n (누적 역합)."""
    return np.cumsum(w[::-1])[::-1]

def freq_response(h, freqs):
    """DTFT 크기 |H(e^{jω})|."""
    k = np.arange(1, len(h) + 1)
    return np.array([abs((h * np.exp(-1j * f * k)).sum()) for f in freqs])

# ---------- 복합리턴 기반 정책평가 ----------
def evaluate(w, episodes, seed, alpha=0.02):
    rng = np.random.default_rng(seed)
    V = np.zeros(N + 2)        # index 0..N+1, 말단=0
    rmse = np.empty(episodes)
    for e in range(episodes):
        states, rewards = episode(rng)
        T = len(states)
        # 각 시점 n-step 리턴 G^(n)_t, 그리고 복합리턴 G^w_t
        for t in range(T):
            Gw = 0.0
            for n in range(1, KMAX + 1):
                if w[n - 1] == 0:
                    continue
                # n-step 리턴 (부트스트랩)
                g = 0.0; disc = 1.0; boot = 0.0
                for i in range(n):
                    if t + i < T:
                        g += disc * rewards[t + i]; disc *= GAMMA
                        last = states[t + i + 1] if (t + i + 1) < T else 0  # 다음상태(말단=0)
                    else:
                        break
                # 부트스트랩: n 스텝 뒤 상태 가치 (경로 안이면)
                if t + n < T:
                    boot = disc * V[states[t + n]]
                g += boot
                Gw += w[n - 1] * g
            s = states[t]
            V[s] += alpha * (Gw - V[s])
        rmse[e] = np.sqrt(np.mean((V[1:N + 1] - V_TRUE) ** 2))
    return rmse

# ---------- 실행 ----------
designs = {
    "TD(0)      FIR len1": w_nstep(1),
    "n-step n=4 FIR len4": w_nstep(4),
    "TD(λ=0.8)  IIR기하":  w_tdlambda(0.8),
    "FIR 사각 K=8":        w_fir_rect(8),
    "FIR Hann  K=12(설계)": w_fir_hann(12),
}

EPISODES, SEEDS = 200, 200
print(f"19-state random walk 정책평가 | episodes={EPISODES}, seeds={SEEDS}, α=0.02\n")

# 임펄스응답/주파수응답 요약
freqs = np.array([0.0, np.pi/8, np.pi/4, np.pi/2, np.pi])
print("설계별 임펄스응답 h_k (앞 8탭) 과 |H| @ ω=0,π/8,π/4,π/2,π")
for name, w in designs.items():
    h = h_from_w(w)
    H = freq_response(h, freqs)
    print(f"  {name:20s} h[:8]=" + " ".join(f"{x:4.2f}" for x in h[:8]) +
          " | |H|=" + " ".join(f"{x:5.2f}" for x in H))
print()

# RMSE 비교 (학습곡선 체크포인트 + 최종)
print(f"{'design':22s} | RMSE@10  50  100  200(final) | AUC(평균)")
results = {}
for name, w in designs.items():
    acc = np.zeros(EPISODES)
    for s in range(SEEDS):
        acc += evaluate(w, EPISODES, seed=100 + s)
    rmse = acc / SEEDS
    ck = [rmse[9], rmse[49], rmse[99], rmse[199]]
    auc = rmse.mean()
    results[name] = (rmse, auc)
    print(f"{name:22s} | " + " ".join(f"{x:5.3f}" for x in ck) + f"   | {auc:5.3f}")

best = min(results, key=lambda k: results[k][1])
print(f"\n최저 AUC(가장 빠르고 낮은 오차): {best}")
print("\n해석: h_k 의 주파수응답이 다름 = '신용을 어느 시간척도에 배분'이 다름.")
print("가치가 매끈(저역)할수록 저역통과형(Hann) 설계가 바이어스-분산 균형에서 유리할 수 있음.")
