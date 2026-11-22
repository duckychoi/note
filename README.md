# 물리-potential SMC 생성 (#36) — 통계 prior × 물리 법칙 융합

생성모델(디퓨전/flow)은 물리를 **통계적으로 암묵** 흡수 → 불완전(물컵 사라짐 등). 재학습 없이 **추론 때 물리를 주입**하자:
$$p(\text{궤적})\ \propto\ \underbrace{p_\theta(\text{궤적})}_{\text{생성=통계 prior}}\ \times\ \underbrace{\exp(-\lambda\,E_{\text{phys}})}_{\text{물리 potential}}$$
**생성=제안(proposal), 물리=가중치(potential), 리샘플+물리 rejuvenation=곱 강제.** = #35 PF머신 + 물리 potential.

> 상태: proof-of-concept. 통계+물리 융합이 재학습 없이 물리를 주입하고 다양성을 보존함을 실증.

## toy (src/toy_physics_potential_smc.py, 1D 수직운동·중력)
물리궤적 $y_t=y_0+v_0t-\tfrac12 g t^2$ ⟺ 2차차분 $=-g$. $E_{phys}=\|Sy+g\|^2$. prior=물리궤적+잡음(비물리 jitter).

| 방법 | 평균 E_phys (낮을수록 물리적) | 초기속도 다양성(std) |
|---|---|---|
| 참 물리궤적 | 0.000 | — |
| raw prior(통계) | **6.448** | 0.414(잡음포함) |
| 물리가이드(Langevin on $E$) | **0.012** | 0.266 |
| 물리-SMC(리샘플+rejuv) | **0.041** | 0.289 |

→ **물리 potential이 E를 6.45→0.01~0.04로 급감(법칙 주입) + 다양성 보존**(참 0.257 수준, 한 궤적 붕괴 아님). **재학습 없이 추론때 통계+물리 융합** 실증.

## 두 노선
- **(A) 샘플링 가이드**: 매 스텝 $-\eta\nabla E_{phys}$ 로 밀기(classifier/energy guidance). 간단·강력.
- **(B) 물리-SMC**: 입자 reweight $\propto\exp(-\lambda E)$ + 리샘플 + 물리 rejuvenation. **분포 수준 교정(제안×potential 곱)**, #35 rejuvenation으로 경로퇴화 방지.

## 🟢 연구 각도 / 난관
- 비디오로 확장: 물리는 **깨끗한 영상**에만 정의 → 중간 노이즈 latent에서 $x_0$-예측으로 디코드 후 $E_{phys}$ 계산→가중(score/x0-pred 활용).
- 물리 potential 설계: 광류 평활·운동량/질량 보존·강체·접촉. 미분가능해야.
- 비용: 다수 입자+경로퇴화 → #35 rejuvenation.
- vs 르쿤 노선: JEPA(생성 X, 표현예측)와 달리 **생성 유지 + 물리 교정** 하이브리드.

## 연결
원장 #36. [[pf-generative-flow]] #35(PF머신) · #34 모션 flow · #25 JEPA/월드모델 · 비디오생성 암묵 물리 한계.

## 구조
`src/toy_physics_potential_smc.py` · `results/`. 요구: Python3 + numpy.
