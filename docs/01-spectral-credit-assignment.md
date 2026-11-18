# 라인 1 — 스펙트럴 신용할당 (Spectral Credit Assignment)

## 동기
리턴 추정기를 **신호처리 필터**로 본다. "Demystifying the Recency Heuristic"(arXiv 2406.12284)은
리턴 추정을 LTI 임펄스응답으로 규정한다. 여기서 한 발 더: **임펄스응답을 원하는 주파수응답으로 설계**한다.

## 수식
복합리턴(compound return):
$$G^w_t = \sum_{n=1}^{K} w_n\, G^{(n)}_t,\qquad \sum_n w_n = 1$$
보상 $r_{t+k}$ 에 실리는 유효 가중치(임펄스응답):
$$h_k = \sum_{n\ge k} w_n \qquad (\text{누적 역합})$$
주파수응답:
$$H(e^{j\omega}) = \sum_k h_k e^{-j\omega k}$$

### 특수경우
- n-step: $w_N=1 \Rightarrow h_k=\mathbf 1[k\le N]$ = **사각 FIR**
- TD(λ): $w_n=(1-\lambda)\lambda^{n-1} \Rightarrow h_k=\lambda^{k-1}$ = **기하 IIR** (표준 eligibility)
- 설계형: $h$ 를 저역통과(Hann 창) 등으로 직접 설계

## 실험 (src/spectral_credit.py)
19-state random walk 정책평가, γ=1, 참값 선형. 200 episodes × 200 seeds.
AUC = 학습곡선 평균 RMSE.

결과: **FIR Hann K=12 (설계) AUC 0.170 < TD(λ=0.8) 0.202 < 사각 FIR 0.205 < n-step 0.216 ≪ TD(0) 0.412.**
→ 가치가 매끈(저역)할 때, 저역통과형 신용할당이 바이어스-분산을 더 잘 잡아 **표준 eligibility를 능가**.

## 🟢 기여 포인트 (미개척)
- "최근성 휴리스틱(단조 감쇠 eligibility)"을 **임의 FIR/IIR 필터 설계**로 일반화.
- 문제의 스펙트럴 구조(보상/가치의 주파수 특성)에 **신용할당 필터를 정합(matched filter)**.
- 연결: wiki 연구원장 #28 스펙트럼 기억(푸리에/복소 SSM) — 기억과 신용할당을 같은 스펙트럴 언어로.

## 다음
- 보상이 주기/대역 구조를 가진 환경에서 **대역통과 설계**가 이기는지.
- 함수근사(심층 가치망)에서 eligibility를 FIR/IIR 필터 레이어로 학습.
