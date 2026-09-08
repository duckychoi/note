---
title: HeyGen
type: entity
domain: video-saas
tags: [video-saas, entity, company, video-generation, avatar, open-source, agent]
created: 2026-09-08
updated: 2026-09-08
sources: [hyperframes.md]
reliability: high
---

# HeyGen — AI 아바타·영상 생성 기업

**GitHub 조직**: https://github.com/heygen-com

> [!insight] 핵심 — 영상 AI 기업이 "모델이 아닌 렌더러"를 오픈소스로 냈다
> HeyGen은 AI 아바타/영상 생성 SaaS 기업이다. 그런데 공식 오픈소스로 낸 [[hyperframes]](**⭐47,070 · Apache-2.0**)는 **생성 모델이 아니라 HTML→영상 렌더 파이프라인**이다.
> → **자사 핵심 자산(모델)은 닫고, 주변 인프라(렌더러)는 연다**는 전형적 오픈코어 배치다. 볼트가 [[Firecrawl]](코어 AGPL·SDK MIT)·[[MiniMax]](*"오픈 가중치 ≠ 오픈 시스템"*)에서 관측한 패턴의 **또 다른 변형**이며, 여기선 **층을 나눠 연다**(모델 닫음 / 렌더 층 Apache-2.0 전면 개방).

> [!note] 배포물의 지향점이 "에이전트"다
> [[hyperframes]] 의 설명 원문은 *"Write HTML. Render video. **Built for agents.**"* 이고 토픽에 `mcp` 가 있다.
> → 영상 AI 기업이 **에이전트를 자사 제품의 사용자로 상정**하기 시작했다는 신호. [[Anthropic]]·[[Google]]·[[OpenAI]] 가 에이전트 도구를 내는 것과 달리, **응용 SaaS 기업이 에이전트용 인프라를 내는** 사례다.

## 관련 페이지
- [[hyperframes]] — 공식 오픈소스(소스 페이지)
- [[Higgsfield]] · [[Seedance]] · [[MiniMax]] — 영상 AI 경쟁/인접 기업
- [[AI-영상-생성-2026]] · [[에이전트-스킬]]

## 원본
- 출처: https://github.com/heygen-com/hyperframes
- 검증: GitHub API 실측 (2026-09-08 · ⭐47,070 · Apache-2.0 · 당일 push)
- 신뢰도: ⭐⭐⭐⭐
