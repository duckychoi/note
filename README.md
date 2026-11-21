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

## 라인 3 — Flow-Matching 프라이어 = 스타일 리워드 (위상+분포)

**아이디어:** 모션 리워드를 손으로 짜지 말고, RoPE(=회전 위상)로 리듬을, flow matching(=분포)으로 flow/스타일을. AMP 판별자(적대)를 **flow-matching 잔차(수송·비적대)** 로 대체.
```
r_style(x, φ) = − E_{t, x0} || v_θ(x_t, t; φ) − (x − x0) ||²     (x_t=(1−t)x0+t·x)
```

**증명 스케치(docs/03):** 고정 에너지에 대한 리워드최대화는 $q$에 선형→**모드붕괴**. **max-ent(soft) RL** 과 결합해야 $q^*\propto\exp(r/\alpha)$, $r=\log p$, $\alpha=1$ → $q^*=p_{data}$ (분포매칭). 열린 과제: FM 잔차와 $-\log p$ 의 gap(경로 divergence) 특성화.

**toy 검증(src/toy_fm_vs_amp.py, numpy only, 3-모드 GMM):**
- **C1** 리워드최대화(argmax)=단일점 붕괴. soft α: 0.05→1모드, 1.0→목표가중 [0.6,0.3,0.1] 복원(H 0.82). → 이론 확인.
- **C2** 비모수 FM 잔차 에너지 corr(logp)=**0.93**, 모드 에너지 순서=가중치 순서, 모두 OOD보다 높음 → **유효 밀도점수**.
- **C3** 완전수렴 AMP corr 0.61 / 덜학습 AMP(실GAN) corr 0.54·표준편차 **0.07(불안정)·최저 0.37** vs **FM 0.93·분산 0** → FM이 더 정확·안정.

**설계 함의:** PPO 리워드최대화 ✗ → **soft/max-ent RL ✓**. 위상조건 $v_\theta(\cdot;\phi)$ → 위상별 분포매칭=리듬.

## 라인 3+ — Style/Content Loss 접목 (docs/04)
Gram 스타일손실 = 시간축 피처 **자기상관** ↔ (Wiener–Khinchin) **파워스펙트럼**. 분해: **CONTENT=위상 민감 목표, STYLE=위상 불변 PSD(리듬 텍스처), 분포=FM**. 세 축 상보: RoPE(위상)+FM(분포)+Gram/PSD(리듬). 싸고 비적대인 리듬 매칭 항.

**toy 검증 (src/toy_psd_style.py):**
- **S1** 같은 리듬·다른 위상 → per-frame L2 0.002→**2.08 폭발**, **PSD거리 0 유지**(위상불변 리듬 포착).
- **S2** PSD거리가 리듬(주파수) 정확히 판별(f0=6에서 최소).
- **S3** content 이벤트는 per-frame만 분리, PSD는 둔감 → content/style 분리 타당.
- **W** 순환 자기상관 = IFFT(PSD) 오차 **2.84e-14** → Wiener–Khinchin(Gram=PSD) 수치확인.

**toy 검증 — RoPE 위상조건 생성 (src/toy_rope_phase_flow.py):**
링(순환 위상 다양체)에서 비모수 조건 flow. **무조건 κ=0: 각도오차 1.63rad(명령 무시) → RoPE조건 κ=8: 0.33rad(추종), 반경 2.07(링 위)**. RoPE 위상조건이 생성 리듬 위상을 제어함을 실증(비모수라 다소 coarse; NN이면 더 sharp).

## 라인 3++ — FM 잔차 → divergence 상한 정리 (docs/05)
**정리:** 잔차는 바이어스-분산 분해 $\mathcal R(q)=B(q)+C(q)$. 바이어스항 $B(q)$ 는 $=0\iff q=p_{data}$ (proper) 이고,
$$\mathrm{KL}(q\|p_{data})\le\tfrac{1}{2\sigma^2}\!\int_0^1\!\mathbb E\|v^*-u_q\|^2dt\asymp\tfrac{1}{2\sigma^2}B(q)\quad(\sigma{>}0,\ \text{Girsanov+데이터처리}),$$
$\sigma{=}0$ 이면 $W_2^2(q,p_{data})\lesssim B(q)$ (연속방정식 안정성). → FM 잔차 리워드 최대화(+max-ent)가 분포매칭을 민다.

**수치검증(src/toy_fm_divergence.py):** 교란분포족에서 **base(KL=0) 잔차 최소(proper 확인)**, **Spearman corr(R, KL)=0.73**(단조·방향 확인). 큰 shift의 비단조는 비모수 $v^*$ 의 OOD 열화(§4.3, 추정 artifact).

**§4.1 C(q) 축소 시도 — 유용한 음성결과(src/toy_rectified_coupling.py):** OT결합이 데이터 경로는 직선화(CFM잔차 4.36→0.82)하나, eval에서 $x_0$ 독립샘플이면 **학습·평가 결합 불일치**로 corr(R,KL) 0.77→**0.25 역효과**. 교훈: C 축소엔 **train/eval 결합 일관성** 필수, 또는 **가능도/점수매칭형 리워드**(C floor 없음)가 더 유망.

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
  toy_fm_vs_amp.py       # 라인3: FM 잔차 리워드 vs AMP (분포매칭·안정성)
  toy_psd_style.py       # 라인3+: PSD 스타일손실 (위상불변 리듬, Wiener-Khinchin)
  toy_rope_phase_flow.py # 라인3+: RoPE 위상조건 flow 생성 (링=순환 위상)
  toy_fm_divergence.py   # 라인3++: FM 잔차 R(q) vs KL(q||p_data) 단조성 검증
  toy_rectified_coupling.py # 라인3++: OT결합 C축소 시도(음성결과: eval 결합일관성 필요)
docs/
  01-spectral-credit-assignment.md
  02-stsa-robust-sa.md
  03-fm-reward-divergence-proof-sketch.md
  04-style-content-loss.md
  05-fm-residual-divergence-theorem.md
results/                 # 실행 로그
references.md
```
요구사항: Python 3 + numpy (`pip install numpy`).

## 참고 맥락
LLM-wiki 연구원장 #28(스펙트럼 기억)·#7/#8/#9(슈퍼트위스팅)·#32(STSA)와 연결. 자세한 개념 노트는 `docs/` 및 wiki `concepts/` 참조.
