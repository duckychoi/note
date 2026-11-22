# LeWAM (#38) — LeWorldModel + Action + Flow Matching + RoPE

교수님 아이디어: **LeWM(르쿤 월드모델)에 행동 + flow matching + RoPE가 들어갈 자리**를 분석.

> 상태: 분석 + proof-of-concept toy. LeWM 원문 구조 기반으로 flow·RoPE 삽입점 규명 + 행동조건부 다봉 예측 실증.

## LeWorldModel 구조 (arXiv 2603.19312 원문 기반)
- **인코더** `enc(o)=z`: ViT-tiny, 프레임당 **단일 CLS 벡터(192d)** (패치그리드 아님) + projector(BN).
- **예측기** `ẑ'=pred(z,a)`: 인과 트랜스포머(6층), **행동은 AdaLN(zero-init)** 로 주입, **프레임 히스토리 자기회귀**.
- **손실** `ℒ = ‖ẑ'−z'‖² + λ·SIGReg(Z)`: 예측=**결정론 MSE**, SIGReg=임베딩을 **등방 가우시안 N(0,I)** 로 밀어 붕괴방지(EMA·stop-grad 불필요).
- **위치인코딩 = 학습형**(RoPE/정현파 아님). **계획 = CEM/MPC**(terminal latent goal match).
- **한계**: 결정론→다봉 미래를 평균(논문 미논의), 롤아웃 오차누적→짧은 지평, TwoRoom 열세.

## 🟢 flow·RoPE 삽입점 분석
**① Flow matching 자리 = 예측기 MSE 헤드.**
`ẑ'=pred(z,a)` (점예측)를 **조건부 flow** $v_\theta(z'^\tau,\tau\mid z,a)$ 로 교체 → $p(z'|z,a)$ 생성. 행동은 기존 **AdaLN 훅** 재사용, flow 시각 $\tau$ 도 AdaLN으로.
- 💎 **궁합 최적(핵심 인사이트)**: SIGReg가 latent 주변을 **N(0,I)** 로 만듦 = **flow matching의 base 분포가 공짜**. SIGReg(붕괴방지→가우시안) × flow(가우시안 base→데이터)가 딱 맞물림.
- 이득: **행동조건부 다봉 미래** 보존(toy 확인). 계획은 CEM을 **샘플 롤아웃 기대비용**으로.
- 주의: 확률 예측기면 SIGReg 붕괴분석 재정의 필요.

**② RoPE 자리 = 예측기 시간축 위치인코딩(현재 학습형).**
인과 히스토리 트랜스포머의 **시간축에 RoPE**. 
- 이득: 상대시간·주기(보행·순환 동역학) + **지평 외삽(RoPE가 긴 위치 일반화)** → LeWM의 **짧은 지평/오차누적 한계 직격**.
- 변형: CLS 벡터 대신 **패치토큰 latent + 공간 RoPE** → "latent이 시공간에서 자유롭지 않게"(교수님 포인트).

## toy (src/toy_action_world_model.py)
행동조건부 분기: a=0 중립→$s\pm2$(다봉), a=±1→편향(단봉). s=0:
- 결정론: a=±1 정확, **a=0 → ẑ≈0(갭, 다봉 평균붕괴)**.
- 생성형 flow: a=−1→좌 0.82, a=+1→우 0.81, **a=0→양쪽(0.32/0.38)**. → 행동이 미래 분포를 조절.

## 두 기여 포인트
1. **가우시안 정규화 latent = native flow base** (SIGReg × flow 시너지) — 표현공간 생성 월드모델.
2. **RoPE로 latent 월드모델 지평 외삽** — 롤아웃 오차누적 완화.

## 연결
원장 #38. [[generative-jepa]] #37(생성형 예측기) · JEPA/월드모델 #25 · [[pf-generative-flow]] #35 · [[physics-potential-smc]] #36 · 스펙트럼 위상 #28 · 모션 flow #34.

## 구조
`src/toy_action_world_model.py` · `results/`. 요구: Python3 + numpy.
