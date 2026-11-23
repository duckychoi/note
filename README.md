# 소산성(Dissipativity) 신경 KF 세계모델 (#39)

교수님 아이디어: **KF는 PF와 달리 샘플 없이 예측–보정 → 효율적 세계모델**. 거기에 **extended dissipative performance(제어이론 에너지 지표)로 학습**을 제약해 "소산적=안정"한 필터를 만들고, Mamba/Transformer 백본으로 world model 구현.

> 상태: 아이디어 + 핵심원리 toy. 나침반(에너지기반·샘플효율·물리세계) 전부 적중.

## 선행 (조각은 있음, novel 조각 분리)
- **KF=샘플프리 신경 세계모델**: KalmanNet(NN이 칼만게인 학습, 노이즈모델 가정X)·DynaNet·2026 DL-KF·KalmanTNet(Transformer). → "효율적 예측–보정 세계모델"은 실재.
- **Mamba×KF**: **KalMamba**(Mamba로 latent 선형가우시안 SSM 학습→KF+스무딩, POMDP 세계모델)·Mamba-2+EKF. → "Mamba 궁합"은 검증·일부 구현됨. (Mamba=선택적 SSM=학습된 상태공간 점화=KF 예측단계의 친척.)
- 🟢 **빈 땅 = extended dissipativity로 학습**: dissipativity(H∞·L2-L∞·passivity 통합 에너지 지표)를 신경 KF/SSM 세계모델 **학습 목적**으로 쓴 건 **검색상 없음**. 이게 novel 기여.

## 왜 소산성인가 (핵심)
세계모델은 **계획 위해 '관측 없이 상상(imagination rollout)'** 해야 함. 이때 학습된 필터가 불안정(ρ>1)하면 **rollout이 발산**(LeWM #38의 "롤아웃 오차누적·짧은 지평" 한계의 근본 원인). **소산성 = 에너지 발산 안 함 = Lyapunov 안정** → 장기 rollout 유계 보장.

## toy (src/toy_dissipative_kf.py)
노이즈 짧은 궤적에서 선형 SSM(A) 적합(sample-free). plain(제약X) vs dissipative(ρ≤0.98 투영=에너지 소산 강제).
- **plain: 불안정 ρ>1 11% · rollout 발산 1% · 최종norm 평균 0.6·최대 16.0**
- **dissip: 불안정 0% · 발산 0% · 평균 0.2·최대 0.5** → 안정 보장.
(관측 보정하면 plain도 가려지나, 계획용 상상엔 내재 안정성 필수.)

## 설계 (제안)
- 백본: **Mamba(선택적 SSM)** = 학습된 상태전이 Ā,B̄,C → KF 예측단계. + **칼만 보정(게인 K·공분산)** 헤드(KalmanNet식).
- 목적: 예측오차 + **extended dissipativity LMI를 미분가능 패널티**(또는 구조적 만족 파라미터화)로. → 안정·강인(H∞/passivity) 보장 세계모델.
- 계획: 상상 rollout이 안정하므로 MPC/latent 계획 신뢰도↑.

## 연결 & 나침반
원장 #39. [[super-twisting-stochastic-approximation]] #32(칼만≈RL·강인) · [[le-wam]] #38(세계모델 rollout 안정화) · [[pf-generative-flow]] #35(PF는 샘플·다봉, KF는 샘플프리·단봉 — 상보) · 🧭 나침반(에너지기반·샘플효율·물리·학계 소규모 적합) 전적 부합.

## 구조
`src/toy_dissipative_kf.py` · `results/`. 요구: Python3 + numpy.
