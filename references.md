# References

## 오차보정 골격 · 칼만 기반 RL
- Geist & Pietquin, *Kalman Temporal Differences*, JAIR 2010. arXiv:1406.3270
- Geist & Pietquin, *Managing Uncertainty within Value Function Approximation in RL*, ADPRL 2011 (KTDQ)
- *MM-KTD: Multiple Model Kalman Temporal Differences*, 2020. arXiv:2006.00195
- *Probabilistic Successor Representations with KTD*, 2019. arXiv:1910.02532
- *Generalized Kalman Filter based TD RL*, 2026. arXiv:2607.20010

## 스펙트럴 / 필터 관점 신용할당
- *Demystifying the Recency Heuristic in Temporal-Difference Learning*, arXiv:2406.12284 (리턴=LTI 임펄스응답)
- *Discerning Temporal Difference Learning*, AAAI (emphasis-weighted n-step)
- van Seijen & Sutton, *True Online TD(λ)*, JMLR 17 (2016)

## 슈퍼트위스팅 / 슬라이딩 모드
- Moreno & Osorio, *A Lyapunov approach to second-order sliding mode controllers and observers* / *Strict Lyapunov functions for the super-twisting algorithm*, Automatica 2013
- *Super-twisting Adaptive Sliding Mode Control: a Lyapunov Design*
- *Robust Constrained Optimization via Sliding Mode Control*, 2026. arXiv:2604.27587

## 후속 RL 알고리즘 (맥락)
- DreamerV3 (world model, 범용) · TD-MPC2 (모델기반+TD) · MuZero/EfficientZero · GRPO/DPO(LLM)

## 주의
toy 실험(scalar·19-state MRP) 기준 proof-of-concept. 수렴 정리는 미완(연구 프로그램).
arXiv 번호 일부는 검색 스니펫 기반 — 인용 전 원문 확인 권장.
