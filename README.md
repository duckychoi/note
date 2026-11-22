# 생성형 JEPA (#37) — 시공간 latent에서 flow matching 조건부 미래예측

교수님 아이디어: **JEPA(표현예측)에 동의하되, latent이 시공간에서 자유롭지 않아야 하고, JEPA와 flow matching을 결합**하자.

> 상태: proof-of-concept. 핵심 주장(결정론 JEPA=모드평균 붕괴 vs flow=다봉 보존)을 toy로 실증.

## 논제
- **JEPA 약점**: 다음 latent를 **하나로** 예측(회귀) → 예측불가 미래를 **평균내 뭉갬**(다봉성·불확실성 소실). 르쿤 JEPA의 알려진 한계.
- **해법 = 생성형 JEPA**: flow matching을 **JEPA 예측기**로 → "다음 latent 하나" 대신 **조건부 미래 latent 분포** $p(z_{t+1}\mid \text{context})$ 를 flow로 생성. 다봉·불확실성 보존 = **"표현공간에서의 생성"**.
- **시공간 구조 보존**: 시간 RoPE / 복소 SSM 위상(#28·#34)을 latent에 넣어 latent이 시공간에서 "자유롭지 않게"(구조 유지).

## toy (src/toy_generative_jepa.py)
분기 미래: context $c$ → future $z=c\pm\Delta$(좌/우 분기)+잡음.
- **결정론 JEPA(회귀 평균)**: 전부 $\hat z\approx c$ → **모드 사이 '갭'에 떨어짐**(최근접 모드까지 ~1.9 = 무효, 붕괴).
- **생성형 JEPA(flow)**: 좌·우 모드 **둘 다 복원**(우 ~0.35 / 좌 ~0.40 / 갭 0.01). → **다봉 미래 보존**.

→ "표현공간에서 flow로 조건부 미래분포 예측"이 JEPA 모드평균 약점을 해결.

## 선행 & 빈 땅
- **Stochastic Latent Variable JEPA** (arXiv 2611.09302): JEPA에 확률적 잠재변수 — 같은 방향. **flow matching 버전은 빈 땅.**
- **LeJEPA/LeWorldModel/V-JEPA 2**: 결정론·에너지 예측. 생성형(분포)으로의 확장이 열림.
- SOTA 참고: **Wan(알리바바) = DiT + Flow Matching** — flow가 latent 생성에 쓰인 실제 SOTA(여기선 픽셀 latent, 우린 표현 latent).

## 🟢 각도
① 시공간 RoPE latent + 조건부 flow 예측기 ② 행동조건부(월드모델: $p(z'|z,a)$ 다봉) → 계획 ③ 다봉 보존이 접촉·분기 동역학(모션 #34) 뭉갬 방지.

## 연결
원장 #37. [[신규]] JEPA/월드모델 #25 · 스펙트럼 위상 #28 · [[phase-rope-flowmatching-motion-reward]] #34 · [[pf-generative-flow]] #35 · [[physics-potential-smc]] #36.

## 구조
`src/toy_generative_jepa.py` · `results/`. 요구: Python3 + numpy.
