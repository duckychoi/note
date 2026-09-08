---
title: JEPA (Joint Embedding Predictive Architecture)
type: concept
domain: ai-news
tags: [ai-news, concept, jepa, world-model, self-supervised, lecun, representation-learning]
created: 2026-07-12
updated: 2026-07-12
sources: [C-JEPA.md, JEPA-월드모델-서베이-2026.md, Flow-JEPA-연구아이디어.md]
reliability: high
---

# JEPA (Joint Embedding Predictive Architecture)

> [!insight] 핵심 인사이트
> **JEPA = "픽셀·토큰을 복원하지 말고, 표현(embedding) 공간에서 예측하라".** [[Yann-LeCun]]이 2022년 포지션 페이퍼("자율 기계지능을 향한 길")에서 제시한 [[월드모델]] 설계 철학. 세상은 픽셀 단위로는 예측 불가능하므로(잎사귀의 정확한 흔들림 등) **예측 불가능한 디테일은 버리고 추상 표현만 예측**한다. 이것이 생성형(재구성)·대조학습(negative pair)의 한계를 우회하며, 학습된 표현공간 안에서 **행동을 상상하며 계획(planning)** 할 수 있게 해 [[임바디드-AI]]·로봇 제어의 기반이 된다.

## 왜 중요한가

- **생성형(LLM·디퓨전)의 대안 축**: 오토리그레시브·[[Diffusion-월드모델]]이 "모든 픽셀/토큰을 복원"하는 반면 JEPA는 표현만 예측 → 예측 불가능한 정보에 용량을 낭비하지 않음. 2026년 두 축은 경쟁이 아니라 **상보적**(→ [[JEPA-vs-Diffusion-월드모델]]).
- **월드모델 붐의 이론적 심장**: 위키에 반복 등장하는 [[Kairos-World-Model-Stack]]·[[world-action-models]]·[[Qwen-AgentWorld]] 등 월드모델 소스의 개념적 뿌리. [[임바디드-AI]] 5각 루프의 "예측·계획" 꼭짓점.
- **효율화와 원리 공유**: "예측 가능한 건 버린다"는 철학이 2025~2026 LLM 경량화(어텐션 sparsity·KV 압축·CoT 압축)와 같은 곳을 겨눔 → [[Flow-JEPA-연구아이디어]]의 효율화×JEPA 교차.

## 구현 (3요소)

- **Context Encoder** `f` — 보이는 부분 x 인코딩
- **Target Encoder** `f̄` — 가려진 부분 y 인코딩, 보통 context 인코더의 **EMA 사본 + stop-gradient**
- **Predictor** — context 표현 + 위치/latent z → target 표현 예측. 손실 = 임베딩 공간 **L2 거리**
- **붕괴(collapse) 방지**: EMA 비대칭(BYOL·I-JEPA), VICReg류 분산·공분산 정규화([[MC-JEPA]]), 큰/의미 있는 블록 마스킹

> [!note] 마스킹이 곧 귀납편향
> I-JEPA 논문은 "크고 의미 있는 target block + 정보량 있는 context"가 성능을 좌우함을 입증. MAE식 작은 랜덤 마스킹은 표현이 나빠짐. [[C-JEPA]]의 기여도 마스킹을 "객체 단위"로 바꾼 것 하나. → 마스크·훈련 설계가 JEPA의 핵심 변수.

## 계보

- **2022** [[Yann-LeCun]] 포지션 페이퍼 (JEPA·H-JEPA 청사진)
- **2023** I-JEPA (이미지) · [[MC-JEPA]] (모션+콘텐츠, 옵티컬플로우)
- **2024** V-JEPA (비디오)
- **2025** V-JEPA 2 / 2-AC (행동조건 월드모델, 로봇 제로샷 플래닝)
- **2026** [[C-JEPA]] (객체/인과) · LLM-JEPA (언어) · LeJEPA/Var-JEPA (이론)

> [!question] 미해결 질문
> ① 잠재 비교 loss로 L2가 최선인가? (다봉 미래에선 mode-averaging으로 붕괴 → 코사인·프로토타입 CE·분포/생성 손실이 대안) ② 인과 JEPA 다음은? (개입형·계층적·다물체 인과 + 생성-예측 통합) — 상세: [[Flow-JEPA-연구아이디어]]

## 관련 페이지
- [[Yann-LeCun]] · [[Meta]] — 창안자·주도 조직
- [[월드모델]] · [[Diffusion-월드모델]] · [[JEPA-vs-Diffusion-월드모델]] — 상위/대비 개념
- [[C-JEPA]] · [[JEPA-월드모델-서베이-2026]] · [[Flow-JEPA-연구아이디어]] — 근거 소스
- [[임바디드-AI]] · [[world-action-models]] · [[Kairos-World-Model-Stack]] — 응용
- [[Mamba4]] — SSM 기반 효율 dynamics predictor 후보
- [[ai-news]]

## 원본
- 근거 소스: [[C-JEPA]] · [[JEPA-월드모델-서베이-2026]] · [[Flow-JEPA-연구아이디어]]
- 신뢰도: ⭐⭐⭐⭐⭐ (Meta/LeCun 주도, 다수 학회 논문으로 검증된 정립 개념)
