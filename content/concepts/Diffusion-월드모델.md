---
title: Diffusion 월드모델 (Diffusion World Model)
type: concept
domain: ai-news
tags: [ai-news, concept, world-model, diffusion, video-generation, flow-matching]
created: 2026-07-12
updated: 2026-07-12
sources: [JEPA-월드모델-서베이-2026.md, Flow-JEPA-연구아이디어.md]
reliability: high
---

# Diffusion 월드모델 (Diffusion World Model)

> [!insight] 핵심 인사이트
> **[[JEPA]]가 "표현공간 예측(비생성)"이라면, Diffusion 월드모델은 정반대 축 — 다음 프레임의 전체 분포를 학습하고 반복 디노이징으로 실제 픽셀을 렌더링한다.** 그래서 직접 **플레이하거나 운전할 수 있는 시뮬레이터**가 나온다(GameNGen·Genie·GAIA·[[NVIDIA]] Cosmos). DIAMOND의 핵심 발견: "시각 디테일이 성능을 좌우한다" — JEPA가 버리는 디테일이 여기선 자산.

## 왜 중요한가

- **인터랙티브·데이터 생성**: 게임 엔진(GameNGen·Oasis·Matrix-Game), 자율주행 시나리오 합성(GAIA-2·Vista·[[NVIDIA]] Cosmos-Drive), 로봇 상상 학습(Dreamer 4)까지 실용 폭이 넓음.
- **빅테크 파운데이션 경쟁**: [[NVIDIA]] Cosmos(오픈 웨이트 WFM), DeepMind Genie 3, OpenAI Sora("as world simulator"), Wayve GAIA. [[월드모델]] 산업화의 생성형 축.
- **JEPA와 상보**: 느린 롤아웃·오차 누적이 약점 → Diffusion Forcing·History Guidance·consistency로 완화. [[Flow-JEPA-연구아이디어]]는 "표현공간 조건부 flow"로 JEPA의 추상화 + 디퓨전의 확률성을 결합.

## 5대 분야 (서베이 요약)

- **Core/Theory**: DIAMOND(WM 안 RL) · Diffusion Forcing(next-token+디퓨전) · History-Guided Video Diffusion
- **Gaming**: GameNGen(DOOM 20FPS) · Genie 1/2/3 · Oasis · Matrix-Game · Hunyuan-GameCraft
- **Driving**: GAIA-2 · Vista · GenAD · Drive-WM · MagicDriveDiT · DriveLaW(CVPR'26)
- **Robotics**: iVideoGPT · AdaWorld · VPP · Dreamer 4 · NWM([[Meta]] LeCun팀의 디퓨전판 WM)
- **Foundation**: [[NVIDIA]] Cosmos · UniSim · Sora · Pandora

## 관련 페이지
- [[JEPA]] · [[월드모델]] · [[JEPA-vs-Diffusion-월드모델]]
- [[JEPA-월드모델-서베이-2026]] · [[Flow-JEPA-연구아이디어]] — 근거 소스
- [[continuous-latent-diffusion-lm]] — 잠재 디퓨전 인접 개념
- [[NVIDIA]] · [[Meta]] · [[임바디드-AI]]
- [[ai-news]]

## 원본
- 근거 소스: [[JEPA-월드모델-서베이-2026]] · [[Flow-JEPA-연구아이디어]]
- 신뢰도: ⭐⭐⭐⭐ (다수 학회 논문·빅테크 릴리스 기반, 일부 산업 모델은 코드 비공개)
