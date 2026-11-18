# 라인 2 — 강인·비정상 확률근사 STSA (Super-Twisting Stochastic Approximation)

## 통일 골격
$$\text{새 추정} = \text{옛 추정} + (\text{게인})\times(\text{관측}-\text{예측})$$
= 1차 확률근사(Robbins–Monro). 칼만 $x{+}K(z{-}Hx)$ ↔ TD $V{+}\alpha\delta$.
차이: 게인을 **계산**(칼만, 공분산 전파) vs **정함**(RL, $\alpha$). 간극 메움 = KTD(Geist–Pietquin 2010 등).

## 제안: STSA
"1차·무기억" 가정을 깨 혁신 $\delta$ 를 **분수차 + 적분기억**으로:
$$\theta_{t+1}=\theta_t+\alpha\big(-k_1|\delta_t|^{1/2}\mathrm{sign}(\delta_t)+u_t\big),\quad
u_{t+1}=u_t-\alpha k_2\,\mathrm{sign}(\delta_t)$$

## 결정론 백본 (증명됨)
Moreno–Osorio 엄격 Lyapunov $V=\zeta^\top P\zeta,\ \zeta=[|\delta|^{1/2}\mathrm{sign}\delta,\ u]^\top$,
$\dot V\le-\kappa V^{1/2}\Rightarrow$ 유한시간 수렴 (Automatica 2013; constrained-opt via SMC arXiv 2604.27587).

## 실험 1 — 고정 STW (src/stsa_robust_sa.py)
scalar tracking, 400 seeds, 후반 RMSE:

| 시나리오 | SA(const) | SA(1/t) | STW | 승자 |
|---|---|---|---|---|
| A 정상 | 0.101 | **0.019** | 0.164 | SA(1/t) |
| B 램프 | 0.177 | 0.446 | **0.164** | STW |
| C 스텝 | 0.101 | **0.067** | 0.164 | SA(1/t) |
| D 상수외란 | 0.416 | 5.58 | **0.164** | STW |
| E 정현외란 | 0.376 | 2.36 | **0.308** | STW |

**🔑 STW tail ≈ 0.164 로 외란 불변** = 강인성 정리 실증. 깨끗한 환경에선 noise floor로 패배(정직).

## 실험 2 — 적응게인 STW (src/stsa_adaptive.py)
$\alpha_t=\alpha_0/(t+1)^q$ 로 감쇠 → floor 축소 + 강인성 유지 가설.

| 방법 | A 정상 | D 외란 |
|---|---|---|
| SA(1/t) | **0.019** | 5.58 |
| STW(고정) | 0.163 | 0.163 |
| **STW(적응)** | **0.051** | **0.132** |

→ 적응 STW가 고정 STW를 **양 축에서 지배**(A 3배↓, D 유지). 1/t 완전 추격은 미달 = 잔여 floor.

## 🟢 남은 이론 (핵심 기여 타깃)
1. **확률 슈퍼트위스팅 유한시간 정리**: 마팅게일 차분 노이즈 + 감쇠 step 하에서 "유한시간 노이즈볼 도달 + 가속 수렴률".
   - 도구: 확률 Lyapunov(초마팅게일) + ODE 방법(비평활·유한시간 안정 극한) + Filippov/확률평균화.
2. **noise floor 특성화**: floor $\approx f(\sigma,k_1,k_2,\alpha)$ 닫힌형 + 적응 스케줄로 소거.

## 응용 니치
비정상/지속학습(continual) RL, 강인·적대적 RL, **시변 채널 빔포밍**(원장 #7/#8/#30), KTD식 불확실성.
