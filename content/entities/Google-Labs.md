---
title: Google Labs
type: entity
domain: ai-news
tags: [entity, google, google-labs, design-system, agent-tooling]
created: 2026-06-25
updated: 2026-07-11
sources: [design-md.md, stitch-skills.md]
reliability: high
---

# Google Labs (google-labs-code)

> [!insight] 핵심 인사이트
> Google의 실험적 AI 제품·개발자 도구 조직. GitHub `google-labs-code` 네임스페이스로 에이전트 코딩 관련 명세·도구를 공개한다. 2026-06-25 트렌딩한 [[design-md]](코딩 에이전트용 비주얼 아이덴티티 명세)의 제작 주체로, "사람이 아니라 **에이전트가 읽는 표준 파일**"(코드의 `CLAUDE.md`/`AGENTS.md`에 대응하는 디자인판)을 정의하는 방향을 주도.

> [!note] 배경 정보
> Google Labs는 검색·생성형 AI 실험을 외부에 공개해온 조직. `google-labs-code` 레포 계열은 에이전트 코딩 워크플로우 표준화에 초점을 맞춘다.

## 주요 산출물
- [[design-md]] — 코딩 에이전트가 디자인 시스템을 영속 컨텍스트로 이해하도록 시각 아이덴티티를 기술하는 마크다운 명세 (⭐17,801, 2026-06-25)
- [[stitch-skills]] — Agent Skills 오픈 표준 기반 스킬 라이브러리(Design/Build/Utilities), Stitch MCP 연동·Claude Code/Cursor/Gemini CLI 지원 (⭐6,863, 2026-07-11)

> [!note] 2026-07-11 추가 — 명세에서 실행 스킬로
> [[design-md]](명세)에 이어 **[[stitch-skills]]**(실행 스킬 라이브러리)를 공개 — "디자인을 어떻게 기술하나"에서 "그 디자인으로 무엇을 만드나(코드·React Native·**Remotion 영상**·shadcn)"로 확장. Google Labs가 에이전트 코딩 표준화를 **명세→실행 도구** 양면에서 밀고 있음.

## 관련 페이지
- [[design-md]]
- [[stitch-skills]]
- [[Claude-Code-워크플로우]]
- [[LLM-Wiki]]
- [[AI-에이전트-프레임워크]]

## 원본
- 출처: https://github.com/google-labs-code
- 신뢰도: ⭐⭐⭐⭐ (Google 공식 조직)
