# PF 생성 플로우 — 방법 정식화 + 오차바운드 스케치

## 1. 세팅
생성 플로우: 확률플로우 ODE $\dot x = v(x,t)$ (또는 역SDE $dx=v\,dt+\sigma dW$), $p_0\to p_1=p_{data}$.
중간 주변 $p_t$ 는 학습된 score로 (원칙적으로) 접근 가능.

## 2. Euler 적분 (기준)
스텝 $h=1/N$: $x_{k+1}=x_k+h\,v(x_k,t_k)$. 국소오차 $O(h^2)$, 전역 바이어스 $O(h)=O(1/N)$. **결정론 → 바이어스 교정 수단 없음.**

## 3. SMC 생성기 (제안)
$K$ 입자 $\{x_k^i,w_k^i\}$. 각 스텝:
1. **예측(조악)**: $\tilde x^i = x_k^i + h\,v(x_k^i,t_k) + \tau_k\sqrt h\,\xi^i$ (탐색노이즈).
2. **보정(가중)**: 알려진 주변 $p_{t_{k+1}}$ 을 potential로, 제안밀도 $q_k$(전이커널 혼합) 대비
   $$w^i \propto \frac{p_{t_{k+1}}(\tilde x^i)}{q_k(\tilde x^i)},\qquad q_k(x)=\tfrac1K\sum_j \mathcal N(x;\,x_k^j+hv(x_k^j,t_k),\,\tau_k^2 h I).$$
3. **리샘플**: $\mathrm{ESS}=1/\sum_i (w^i)^2 < \rho K$ 이면 $w$ 로 재추출.

→ 매 스텝 입자군이 **참 주변 $p_t$ 로 되끌려감**. $K\to\infty$ 에서 self-normalized IS 일관 → **임의 $h$ 에서 점근 무바이어스**.

## 4. 오차바운드 스케치 (핵심 기여 타깃)
- **Euler**: $\mathcal W_2(\hat p_1^{Euler}, p_1) = O(h)$ (이산화 바이어스, $K$ 무관).
- **SMC**: $\hat p_1^{SMC}$ 의 바이어스 $= O(1/K)$(SMC 일관) $+\ \epsilon_{score}$(score 근사) $+\ \epsilon_{res}$(리샘플 분산). **이산화 $h$ 에 대한 1차 바이어스 없음** — 가중이 각 스텝 주변을 교정하므로.
  $$\boxed{\ \mathcal W_2(\hat p_1^{SMC},p_1)\ \le\ \frac{C_1}{\sqrt K}+C_2\,\epsilon_{score}\quad(\text{vs Euler } O(h))\ }$$
  → 고정 소수 depth $N$(큰 $h$)에서 SMC(충분한 $K$)가 Euler보다 정확. **depth(순차 NFE)를 width(병렬 $K$)로 교환.**
- 증명도구: SMC 전파오차의 Banach/수축 논법(Del Moral), self-normalized IS 분산, 전이커널 혼합의 밀도비 유계.
- ⚠️ **경로퇴화**: 리샘플 반복이 입자다양성 소실→소수모드 소멸(toy: N≥8 열화). 바운드에 $\epsilon_{res}$ 가 $N$ 에 누적. **rejuvenation(MCMC move)·stratified 리샘플**로 $\epsilon_{res}$ 억제가 필수 보조정리.

## 5. toy 실측과의 부합
- N=3·5: SMC 에너지거리 ≪ Euler (바이어스 교정 우세) → §4 "저 NFE 우월" 확인.
- N≥8: Euler $O(h)$ 작아지고 SMC $\epsilon_{res}$ 누적 → 크로스오버. §4의 $\epsilon_{res}$ 항이 지배.
- naive 어닐 실패: $\tau_k\to0$ 이 $q_k\to\delta$ → 밀도비 폭주(가중퇴화). → §3 제안분산 하한 필요.

## 6. 남은 과제
rejuvenation 커널 설계($\epsilon_{res}$ 바운드) · score 근사오차 전파($\epsilon_{score}$) · 고차원 가중분산 · ESS-적응 depth 스케줄(compute-optimal).
