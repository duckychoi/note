---
title: 임바디드 AI (Embodied AI)
type: concept
domain: ai-news
tags: [ai-news, concept, embodied-ai, vla, robot, world-model]
created: 2026-07-09
updated: 2026-07-09
sources: [LaMem-VLA.md, LingBot-Video.md, RoboDojo.md, World-Infinity.md, VLA-Corrector.md, Embodied-cpp.md]
reliability: high
---

# 임바디드 AI (Embodied AI)

> [!insight] 핵심 인사이트
> **임바디드 AI = 물리·가상 환경에서 지각→행동하는 체화된 AI**(로봇 조작·VLA·월드 모델). 위키에 로봇/VLA 소스가 반복 등장하며(3개+ 소스 규칙 충족) 독립 개념으로 승격. 2026년 여름 흐름의 핵심은 임바디드 스택이 **학습 → 실시간 보정 → 온디바이스 배포 → 평가 → (학습)환경 생성**의 5각 루프로 성숙했다는 것. 각 꼭짓점에 대표 소스가 붙는다: **학습**([[LingBot-Video]] MoE 비디오 사전학습), **보정**([[VLA-Corrector]] 잠재공간 감지-수정), **메모리**([[LaMem-VLA]] 이중 잠재 메모리), **배포**([[Embodied-cpp]] 이식성 C++ 런타임), **평가**([[RoboDojo]] sim-and-real 벤치), **환경 생성**([[World-Infinity]]·[[AlayaWorld]] 인터랙티브 월드).

## 왜 중요한가

- **텍스트 에이전트의 다음 축**: 코딩 에이전트가 "보고·듣기"([[claude-video]]·[[speech-to-speech]])로 넓어진 데 이어, "움직이기"(로봇 조작)가 HF 데일리 상위를 반복 점유. AI 능력 지형이 디지털→물리로 확장.
- **메모리·롱컨텍스트와 수렴**: [[LaMem-VLA]]의 "단기 시각 + 장기 의미 이중 메모리"는 [[에이전트-메모리-레이어]]의 로봇판. 롱호라이즌 조작의 병목이 지각에서 **기억 통합**으로 이동 — 텍스트 에이전트 메모리 설계와 원리 공유.
- **평가의 신뢰성 문제 동형**: [[RoboDojo]]가 sim↔real·능력축 분해 평가를 미는 것은 [[Beyond-Static-Leaderboards]] "리더보드 ≠ 실제 신뢰성"의 로봇 버전. 임바디드도 벤치 과신 경계가 필요.
- **온디바이스가 관건**: [[Embodied-cpp]]("임바디드의 llama.cpp")는 로봇에 모델을 올리는 배포층. 로컬/엣지 실행([[speech-to-speech]]·[[pocket-tts]]와 공통 신호)이 임바디드에서도 핵심.

## 내 작업과의 접점

- **직접 접점은 약함** — 내 워크플로(영상·에이전트·로컬 LLM)와 로봇 조작은 도메인이 다름.
- **이식 가능한 개념**: ①[[LaMem-VLA]] "단기 시각/장기 의미 분리 메모리" → [[down-analysis]] 영상 이해 메모리 설계, ②[[RoboDojo]] "능력축 분해 평가" → 스킬 평가 프레임, ③[[World-Infinity]] 인터랙티브 월드 생성 → [[AI-영상-생성-2026]] video-saas와 교차(예쁜 클립 → 탐험 가능한 환경).

## 관련 페이지
- [[LaMem-VLA]] · [[VLA-Corrector]] — VLA 메모리·보정
- [[LingBot-Video]] — 임바디드 비디오 사전학습
- [[Embodied-cpp]] — 온디바이스 런타임
- [[RoboDojo]] — sim-and-real 평가
- [[World-Infinity]] · [[AlayaWorld]] · [[WorldDirector]] — 월드 모델/환경 생성
- [[에이전트-메모리-레이어]] · [[Beyond-Static-Leaderboards]] — 교차 개념
- [[ai-news]]

## 원본
- 근거 소스: [[LaMem-VLA]] · [[LingBot-Video]] · [[RoboDojo]] · [[World-Infinity]] · [[VLA-Corrector]] · [[Embodied-cpp]]
- 신뢰도: ⭐⭐⭐⭐ (다수 HF 논문·소스에서 반복 확인된 도메인 트렌드)
