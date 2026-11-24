#!/usr/bin/env python3
"""
toy — "창의력 = 긴 기억력 + 위상 보강간섭" 조작화 (#28)
=====================================================
FHRR(푸리에 홀로그래픽 기억) + 복소 SSM(감쇠 복소진동자)로:
 T1 용량(capacity): 중첩 N쌍에서 role로 filler 회상 정확도 vs N → '누화 천장' 측정.
 T2 창의=위상 재조합: 알려진 조각으로 '새 아이템' 합성 후 읽어내기(coherent recombination)
    정확도 vs 복잡도(roles) → '창의력'을 조작화 + 천장.
 T3 복소SSM 시간기억: 스트림을 x_t=λ x_{t-1}+TIME⊛item 로 누적. 회상 vs lag,
    기억길이(1/α)가 길수록 더 먼 과거 재조합 가능하나 누화↑ → 용량-창의 트레이드오프.
numpy only (복소).
"""
import numpy as np
rng = np.random.default_rng(0)
D = 1024  # 차원

def phasor(n=1):
    return np.exp(1j*rng.uniform(-np.pi, np.pi, (n, D))) if n>1 else np.exp(1j*rng.uniform(-np.pi,np.pi,D))
def bind(a, b): return a*b                    # 결합 = 성분별 복소곱 (=순환합성곱의 FFT판)
def unbind(a, b): return np.conj(a)*b         # 해제
def sim(a, B):                                # a:(D,), B:(M,D) → 코사인유사도(실부)
    return (np.real(np.conj(B)@a))/D
def cleanup(v, book):                         # 코드북 최근접
    return int(np.argmax(sim(v, book)))

# ================= T1: 용량(누화 천장) =================
print("===== T1: FHRR 용량 — 중첩 N쌍에서 회상 정확도 vs N =====")
M = 100
Roles = phasor(M); Fillers = phasor(M)
print(f"D={D}, 코드북 M={M}")
print(f"{'N(중첩쌍)':>9} {'회상정확도':>10}")
for N in [5, 15, 30, 60, 100, 150]:
    acc = []
    for _ in range(20):
        idx = rng.choice(M, min(N, M), replace=False)
        extra = rng.integers(0, M, max(0, N-M))        # N>M이면 중복 허용
        ids = np.concatenate([idx, extra]) if N>M else idx
        mem = np.zeros(D, complex)
        for i in ids: mem += bind(Roles[i], Fillers[i])
        # 저장된 것 중 무작위 질의
        q = rng.choice(ids)
        fhat = unbind(Roles[q], mem)
        acc.append(cleanup(fhat, Fillers) == q)
    print(f"{N:>9} {np.mean(acc):>10.2f}")
print("  → N 커질수록 누화로 정확도↓ = 중첩 용량 천장(무한주기≠무한용량).\n")

# ================= T2: 창의 = 위상 재조합 =================
print("===== T2: 창의적 재조합(아이템-ID 인덱싱, 부하 L 아래) — 판독 정확도 vs L =====")
# 아이템 = ID_i ⊛ (R개 role⊛value 번들). 창의 새 아이템을 L-1개와 함께 중첩, ID로 되찾아 판독.
R = 3; V = 20; MAXID = 200
RoleV = phasor(R); Vocab = [phasor(V) for _ in range(R)]; IDs = phasor(MAXID)
print(f"roles R={R}, 어휘 V={V}, D={D} (아이템마다 고유 ID 키)")
print(f"{'L(중첩 아이템수)':>14} {'재조합 판독 완전정확률':>18}")
def item_bundle(vals):
    it=np.zeros(D,complex)
    for r in range(R): it+=bind(RoleV[r],Vocab[r][vals[r]])
    return it
for L in [1, 10, 30, 60, 100, 150]:
    ok=0; T=120
    for _ in range(T):
        ids=rng.choice(MAXID, L, replace=False)
        want=[rng.integers(V) for _ in range(R)]                 # 창의 아이템(새 조합)
        mem=bind(IDs[ids[0]], item_bundle(want))                 # id0 = 창의 아이템
        for j in range(1,L):
            mem+=bind(IDs[ids[j]], item_bundle([rng.integers(V) for _ in range(R)]))
        rec=unbind(IDs[ids[0]], mem)                             # ID로 창의 아이템 복원
        good=all(cleanup(unbind(RoleV[r],rec),Vocab[r])==want[r] for r in range(R))
        ok+=good
    print(f"{L:>14} {ok/T:>18.2f}")
print("  → ID 인덱싱 시 부하 L 수십까지 새 조합 coherent 판독(창의 성립), L↑ 용량천장서 붕괴.\n")

# ================= T3: 복소 SSM 시간기억 =================
print("===== T3: 복소 SSM 시간기억 — 회상 vs lag, 기억길이(α) 효과 =====")
L = 40
Times = phasor(L)                              # 시간 태그 (홀로그래픽)
Items = phasor(L)                              # 스트림 아이템(코드북=Items)
print(f"스트림 길이 L={L}")
print(f"{'α(감쇠)':>8} {'기억길이~1/α':>12}  회상정확도 @ lag= " + " ".join(f"{k:>3}" for k in [1,3,6,10,15,25]))
for alpha in [0.30, 0.12, 0.04]:
    lam = np.exp(-alpha)
    x = np.zeros(D, complex)
    recall = {k: [] for k in [1,3,6,10,15,25]}
    for t in range(L):
        x = lam*x + bind(Times[t], Items[t])    # 복소 SSM 누적(홀로그래픽)
        for k in recall:
            if t-k >= 0:
                ihat = unbind(Times[t-k], x)
                recall[k].append(cleanup(ihat, Items) == t-k)
    accs = [np.mean(recall[k]) if recall[k] else np.nan for k in [1,3,6,10,15,25]]
    print(f"{alpha:>8.2f} {1/alpha:>12.1f}  " + "                " + " ".join(f"{a:>3.2f}" if not np.isnan(a) else "  -" for a in accs))
print("  → α작을수록(기억길이↑) 먼 과거(lag↑)까지 회상 = '긴 기억'. 단 많이 중첩→누화.")
print("    즉 긴 기억 = 더 많은 과거를 동시보유 = 재조합(창의) 재료↑, 천장은 용량(T1).")

print("\n===== 종합 =====")
print("창의력=긴기억력+위상보강간섭 을 조작화: (T2) 위상 번들대수로 coherent 새조합 생성,")
print("(T1) 그 능력의 천장=중첩 용량, (T3) 긴 기억(복소SSM 작은 α)=더 많은 재조합 재료.")
print("→ '생성형 홀로그래픽 기억'의 핵심 수치근거 확보. 미해결: 천장 상향(sparsity/clean-up net).")
