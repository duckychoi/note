# 01 — AI-native 빔포밍 재발명 지형

빔포밍은 1960~70년대(MVDR/Capon, 이후 MMSE/WMMSE)에 **사람이 손으로 유도한 모델 기반 알고리즘**이다.
"AI가 얼마나 깊이 침투하느냐"로 재발명 단계를 사다리로 그린다.

## 6단계 사다리

**① 솔버만 교체 (얕음)**
CSI → 가중치 매핑을 CNN/LSTM이 대신 학습. 문제정의·목적함수는 그대로, 추론만 빨라짐. "DL로 빔포밍을 한다" 수준.

**② 알고리즘 언폴딩 (하이브리드)**
WMMSE 등 반복 알고리즘을 신경망 레이어로 펼치고 일부 파라미터만 학습. 구조(사람의 수학)는 유지, 데이터로 보정.
- Deep Graph Unfolding(MU-MIMO), Unfolded WMMSE, Gradient-driven GNN precoder.

**③ 엔드투엔드 재정의 (방법론 재발명 시작)**
- **Neural Beamforming Transformer (NBT, 2026)**: 날 안테나 전압 → 가중치 직접 합성. 복소 트랜스포머 + 회전 위치인코딩 + 미분가능 매니폴드 투영 + 신경 공분산 추정. 임의 연속 빔포밍 매핑의 **보편 근사기** 증명.

**④ 목적함수 재발명 (진짜 패러다임 전환)** ⭐
기존 빔포밍은 SINR/전송률(Shannon 대리지표) 최대화. AI-native는 **다운스트림 과제 성능을 직접 최적화**:
- Task-Aware Beamforming for Semantic Localization — "위치추정에 유리한 필드"를 빚음.
- ISAC(통신+센싱) / 시맨틱 빔포밍 — 빔포밍이 "방향 전력 집속"에서 **"목표에 맞춘 전자기장 프로그래밍"**으로.
- Wireless Multimodal Foundation Model(6G ISAC).

**⑤ 하드웨어 공동설계 (codesign)**
가중치뿐 아니라 배열 기하·sparse array·RIS/메타표면 패턴을 신경망이 역설계. "배열 고정, 가중치만 푼다"는 분해 자체가 pre-AI 관습.

**⑥ AI가 알고리즘을 발견 (literal "vibe coding")** 🔥
- AlphaEvolve/FunSearch(LLM+진화연산)로 **무선 알고리즘 자율 발견·구현·최적화**(2026, "Autonomous Discovery of Wireless Communications Algorithms"). 기계가 새 빔포밍 알고리즘을 써냄.

## 두 가지 핵심 레버리지

1. **사람이 그은 블록 경계를 녹인다.**
   채널추정 → 빔포밍 → 스케줄링 → 센싱의 분리 설계는 pre-AI 분해 방식. AI-native는 하나의 미분가능 파이프라인으로 합쳐 전역 최적화(joint design).

2. **"무엇을 최적화하는가"를 바꾼다.**
   SINR 극대화(①~③)는 옛 목적함수. 진짜 재발명은 ④ task/semantic — 빔포밍이 "전력 집속"이 아니라 "과제가치 극대화 필드 조형".

## 제약과 스윗스팟
빔포밍은 CV/NLP와 달리 **Maxwell 방정식이 하드 제약**. 순수 블랙박스는 CSI 오차·OOD에서 붕괴 → 이기는 레시피는 **physics-informed + 언폴딩**(NBT의 매니폴드 투영·constant-modulus). AI는 "해의 탐색"을 재발명하고, 물리는 뼈대로 남긴다.

## 도메인 연결
통신학회에서 "트랜스포머 빔포머" 1등 경쟁은 늦음(NBT·CAT-MoEformer 선점). 해자는 '최초'가 아니라 **방어 가능한 조합 + 아무도 안 건드린 도메인**:
- ④⑤⑥, 그리고 SMC×빔포밍은 포화 안 됨.
- **로봇 지각·온디바이스**(로컬 LLM 휴머노이드 SLAM/캠핑)에 꽂으면 통신학회가 아닌 로봇-센싱 블루오션.
