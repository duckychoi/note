# Particle-Filter Generative Flow — PF로 확산/플로우 ODE를 대체하기

생성모델(확산·flow matching)의 샘플링은 **확률플로우 ODE/역SDE 적분**으로 진행된다.
이 적분기를 **파티클 필터(SMC)** 로 대체하면? — 예측(조악한 스텝)–보정(가중)–리샘플 구조로
**저 NFE(few-step)에서 이산화 바이어스를 교정**할 수 있다는 가설.

> 상태: **연구 프로그램 / proof-of-concept.** 핵심 가설은 toy로 확인(저 NFE PF 압승), 고 NFE 퇴화는 정직한 한계.

## 문헌 확인 — 빈 공간 확정
- **Feynman-Kac Correctors** (Skreta et al., ICML 2025, arXiv:2503.02819): SMC를 시간축에 쓰나 **조건부/템퍼링**(annealing·CFG·product) 용. **무조건(β=1)이면 보정=0 → 기저 샘플러 대체 안 함.** **NFE 가속 주장 없음**(오히려 compute↑), **유한표본 오차한계 없음**.
- **SURGE**(arXiv:2605.18745, ICML 2026)·**Wu et al. 2023**: 전부 조건부(data assimilation/posterior).
- "Particle Filters and Flows" 정확한 제목 논문 없음 (EnFF·DiffPF 등은 PF 안에 flow를 쓰는 **반대방향**).
- → **무조건 생성에서 확률플로우 ODE를 PF로 '대체'해 NFE를 줄이고 오차한계를 세운 연구는 아직 없음.**

## 핵심 가설 & toy 검증 (src/toy_pf_vs_euler.py)
폐형(closed-form) 선형보간 플로우(GMM): $p_0{=}N(0,I)$, $p_1{=}$GMM, $x_t{=}(1{-}t)x_0{+}t x_1$ → $p_t$·score·$v^*$ 폐형.
- **Euler**: $dx=v^*h$ 결정론 N스텝.
- **SMC**: $dx=v^*h+\text{noise}$, 매 스텝 **알려진 $p_t$ 로 bootstrap 가중 + ESS 리샘플** → 바이어스 교정.

**결과(에너지거리, 낮을수록↑):**
- **N=3**: SMC **0.022** vs Euler 0.108 (**≈5배↓**) / N=5: SMC **0.012** vs 0.055 → **저 NFE PF 압승(가설 확인)**.
- **N≥8**: 경로퇴화(리샘플 반복→소수모드 소멸)로 SMC 열화, Euler 우월. **크로스오버 N≈6~8**.
- naive 노이즈 어닐링(SMC-an): 제안분산→0 이 **가중치 퇴화** 유발 → 악화(실패, 정직한 음성).

**결론:** PF-flow 대체는 **few-step 영역의 방법**. 고 NFE 퇴화를 막는 **rejuvenation(MCMC-move)·stratified 리샘플링**이 핵심 enabling 기법 = 연구 to-do.

## 연구 각도 (3)
1. **ESS-적응 SMC 생성기** — ESS 떨어질 때만 리샘플·세분화로 NFE↓ + "조악한 이산화서 SMC 한계오차 < Euler" **오차 바운드 정리**(Banach 고정점/수축). ← 메인.
2. **Wasserstein 공간 SMC** — 연속성방정식 해를 입자경험분포로, 리샘플을 Wasserstein 기준으로.
3. **ESS 기반 compute 배분** — 망가진 궤적 조기 제거→살아있는 입자에 연산 집중 (compute-optimal 생성/inference-time scaling).

## 정직한 리스크
- 무조건 생성에선 twist/가중 계산이 score 평가를 요구(공유 가능하나) → "FKC 무조건판" 딱지 위험. **델타 = 오차바운드 + few-step 가속**으로 명확히.
- 고차원 가중 분산·경로퇴화(toy가 보여줌) → rejuvenation 필수.

## 구조
```
src/toy_pf_vs_euler.py   # 저 NFE PF vs Euler (GMM 폐형 플로우)
docs/                    # survey·gap, method·오차바운드 스케치
results/
```
요구사항: Python 3 + numpy.
