# FHRR + 복소 SSM — 수식과 논제

## 1. FHRR (Fourier Holographic Reduced Representations)
아이템 = 단위 위상자 복소벡터 $a\in\mathbb C^D$, $a_k=e^{j\theta_k}$, $\theta_k\sim U(-\pi,\pi)$.
- **결합(bind)**: $a\circledast b = a\odot b$ (성분별 복소곱). 시간영역 순환합성곱의 FFT판. 가역: $\text{unbind}(a,c)=\bar a\odot c$.
- **번들(bundle)**: $m=\sum_i x_i$ (중첩). 비가역, 유사도로 복원.
- **유사도**: $\mathrm{sim}(a,b)=\tfrac1D\Re\langle a,b\rangle\in[-1,1]$. cleanup = 코드북 최근접.

연상쌍 저장: $m=\sum_i R_i\circledast F_i$. 질의: $\bar R_j\odot m = F_j + \underbrace{\sum_{i\neq j}\bar R_j R_i F_i}_{\text{누화}}$. 누화는 평균0·분산 $\propto (N-1)/D$ → **용량 $\sim O(D)$ 천장**.

## 2. 복소 SSM (감쇠 복소진동자)
$$x_t=\Lambda\,x_{t-1}+\text{bind}(TIME_t,item_t),\quad \Lambda=\mathrm{diag}(e^{-\alpha_k+j\omega_k}).$$
- **크기 $e^{-\alpha}$ = 기억길이**(1/α 시정수), **위상 $\omega$ = 공명 주파수**. S4D·S5·LRU의 대각 복소 점화와 동형.
- lag $k$ 회상: $\text{unbind}(TIME_{t-k},x_t)\approx e^{-\alpha k}\,item_{t-k}+\text{누화}$. → **α 작을수록 먼 과거까지**.

## 3. 논제: 창의력 = 긴 기억력 + 위상 보강간섭
- **위상 보강간섭(binding/bundle 대수)** → 저장 안 된 **새 조합**을 coherent하게 구성/판독 (= 창의의 메커니즘).
- **긴 기억(작은 α)** → 더 많은 과거를 동시보유 → 재조합 재료↑ → 창의 잠재력↑.
- **천장 = 중첩 용량**(누화 $\propto N/D$). "무한주기≠무한용량".
- 르장드르(LegS) = 충실 장기저장(비주기) vs 푸리에/복소 = 재조합·위상(binding) — 둘의 분업.

## 4. toy 실증 (results/)
- T1: 용량 천장(N=150→0.85).
- T2: 창의 재조합, 부하 L≤30 완벽 → L=150 0.28 (천장=용량).
- T3: α=0.04(길이25) → lag25 완벽 ("긴 기억" 정량).

## 5. 빈 땅
생성형 홀로그래픽 기억(복소SSM+FHRR 보강간섭) · 천장 상향(sparsity/clean-up net/VSA설계) · 위상=창의 조절손잡이(보강↔상쇄로 창의↔충실 연속조절).
