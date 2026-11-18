# Spectral Credit Assignment & Robust Stochastic Approximation

두 개의 연결된 연구 라인에 대한 **개념 + 재현 가능한 toy 실험** 모음.
출발점: *칼만 갱신 `x ← x + K(z−Hx)` 와 RL 갱신 `V ← V + α(G−V)` 이 같은 오차보정 골격* 이라는 관찰.

> 상태: **연구 프로그램 / proof-of-concept.** 결과는 scalar·소형 MRP toy 기준이며, 이론(수렴 정리)은 미완. 정직하게 "되는 곳/안 되는 곳"을 함께 보고한다.

---

## 라인 1 — 스펙트럴 신용할당 (Spectral Credit Assignment)

**아이디어:** 리턴 추정기 = 보상열에 걸리는 LTI 필터. 그 임펄스응답 `h_k` 를 **원하는 주파수응답으로 직접 설계**한다.

핵심 동치:
```
복합리턴 G^w = Σ_n w_n G^(n)   ⟺   보상 r_{t+k} 가중치  h_k = Σ_{n≥k} w_n
```
- n-step (`w_N=1`) → `h_k = 1 (k≤N)` = **사각 FIR**
- TD(λ) (`w_n=(1−λ)λ^{n−1}`) → `h_k = λ^{k−1}` = **기하 IIR** (= 표준 eligibility trace)
- 설계형 FIR → `h` 를 저역통과(Hann) 등으로 **직접 설계**

**결과 (19-state random walk 정책평가, 200 episodes × 200 seeds, AUC=학습곡선 평균 RMSE, 낮을수록↑):**

| 설계 | h 성격 | AUC |
|---|---|---|
| TD(0) | FIR len1 | 0.412 |
| n-step n=4 | 사각 FIR | 0.216 |
| TD(λ=0.8) | 기하 IIR | 0.202 |
| FIR 사각 K=8 | 사각 FIR | 0.205 |
| **FIR Hann K=12 (설계)** | **저역 FIR** | **0.170 ← 최저** |

→ **주파수응답을 설계한 저역통과 FIR 신용할당이 표준 TD(λ)를 이김** (가치함수가 매끈/저역이라 저역통과가 바이어스-분산을 더 잘 잡음).

실행: `python3 src/spectral_credit.py`

---

## 라인 2 — 강인·비정상 확률근사 STSA (Super-Twisting SA)

**아이디어:** 오차보정 골격의 "1차·무기억" 가정을 깨고 혁신 `δ` 를 **분수차 + 적분기억**(슈퍼트위스팅)으로 처리.
```
θ ← θ + α(−k1|δ|^{1/2} sign(δ) + u),   u ← u − α k2 sign(δ)
```

**결과 A (고정 STW vs 표준, scalar tracking, 400 seeds, 후반 RMSE):**

| 시나리오 | SA(const) | SA(1/t) | STW(고정) | 승자 |
|---|---|---|---|---|
| A 정상(측정노이즈) | 0.101 | **0.019** | 0.164 | SA(1/t) |
| B 램프 드리프트 | 0.177 | 0.446 | **0.164** | STW |
| C 급변 스텝 | 0.101 | **0.067** | 0.164 | SA(1/t) |
| D 상수 외란 | 0.416 | 5.58(발산) | **0.164** | STW |
| E 정현 외란 | 0.376 | 2.36 | **0.308** | STW |

**🔑 핵심 발견:** STW의 후반 RMSE가 A·B·C·D에서 **≈0.164로 외란에 거의 불변** → 슈퍼트위스팅 강인성 정리의 실증. (단, 깨끗한 정상환경에선 노이즈 floor로 패배 — 정직한 음성 결과.)

**결과 B (적응게인 STW = "best of both", 후반 RMSE):**

| 방법 | A 정상 | D 외란 |
|---|---|---|
| SA(1/t 감쇠) | **0.019** | 5.58 (취약) |
| STW(고정) | 0.163 | 0.163 |
| **STW(적응 α_t)** | **0.051** | **0.132** |

→ **적응게인 STW가 고정 STW를 양 축에서 지배**: 정상환경 0.163→0.051(3배↓, 1/t에 접근) + 외란 강인성 유지(0.132). 완전히 1/t를 따라잡진 못함(잔여 floor) = 남은 이론 과제.

실행: `python3 src/stsa_robust_sa.py` · `python3 src/stsa_adaptive.py`

---

## 정직한 평가 (논문 가치)
- "더 빠른 TD"로는 **약함** (정상환경에서 짐).
- **"강인/비정상 확률근사"로 프레이밍하면 중상(中上)**: ① 외란 불변 오차 실증 ② 적응게인으로 양축 지배 ③ 응용 니치 분명(continual/robust RL, **시변 채널 빔포밍**).
- 남은 핵심: **확률 슈퍼트위스팅 유한시간 정리** + **노이즈 floor 특성화(σ·k1·k2·α의 함수)** 의 엄밀 증명.

## 구조
```
src/
  spectral_credit.py     # 라인1: FIR/IIR 신용할당 필터 설계
  stsa_robust_sa.py      # 라인2: 고정 STW vs 표준 SA (A~E)
  stsa_adaptive.py       # 라인2: 적응게인 STW 하이브리드 (A,D)
docs/                    # 개념 노트
results/                 # 실행 로그
references.md
```
요구사항: Python 3 + numpy (`pip install numpy`).

## 참고 맥락
LLM-wiki 연구원장 #28(스펙트럼 기억)·#7/#8/#9(슈퍼트위스팅)·#32(STSA)와 연결. 자세한 개념 노트는 `docs/` 및 wiki `concepts/` 참조.
