# RL 자기개선 피처 큐레이션 (#33)

스트리밍 회귀에서 **유용한 피처가 시간에 따라 바뀜(비정상)**. 에이전트(피처 밴딧)가 피처를 능동 선택(큐레이션)해 다운스트림 예측을 돕고 전환에 적응하는가.

> 상태: proof-of-concept. **modest 양성** — 큐레이션이 all/fixed를 이김. 단 oracle엔 못 미침 → 피처 수준 신용할당이 크루스(정직한 한계).

## toy (src/toy_feature_curation.py, P=50 피처, 관련 S=3, 120라운드마다 교체)
큐레이터: 유용도 $q_f=\text{EMA}(|x_f\cdot e|)$ (모든 피처의 잔차상관 = 선택 안해도 계산), top-k 선택+ε탐색, decay로 적응. 다운스트림 = 선택 피처 online LMS(스텝 정규화).

**전체 MSE (낮을수록↑):**
- **RL-curated 2.31** < all(전부) 2.92 < fixed(고정랜덤) 3.03 / oracle(참 피처) 0.72(상한)
- 관련피처 recall: RL **0.39** (랜덤 ~0.12 상회 — 학습 O), oracle 1.0, fixed 0.11.

**해석(정직):**
- ✅ **능동 큐레이션이 "전부 사용"·"고정"을 이김** (비정상·고차원서 선택이 노이즈 피처 억제).
- ⚠️ oracle(0.72)과 격차 큼, recall 0.39로 불완전, 전환 적응속도 우위는 불명확.
- → **크루스 = 피처 수준 신용할당**: $|x_f\cdot e|$ 는 '아직 안 잡힌 피처' 신호라 피처가 fit되면 신호가 꺼져 선택이 불안정(discovery↔retention 딜레마). **더 나은 credit(discovery $|x\cdot e|$ + retention $|W_f|$ 하이브리드)** 이 핵심 to-do.

## 🟢 연구 각도 (원장 #33 개념노트 참조)
① 피처 수준 신용할당(RUDDER/HCA를 피처축) ② [획득+정렬+변환] 단일 미분가능 정책(Gumbel-Softmax) ③ 비정상 강인 업데이트. 이 toy는 ②의 '정렬(선택)'만 최소구현.

## 연결
원장 #33. [[rl-self-curating-features]] 개념노트 · 신용할당 스레드 · [[feature-map-encryption]] #31(큐레이션↔보호 쌍).

## 구조
`src/toy_feature_curation.py` · `results/`. 요구: Python3 + numpy.
