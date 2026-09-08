---
title: 금융 AI
type: concept
domain: ai-news
tags: [concept, finance-ai, trading, time-series, multi-agent, LLM, quant]
created: 2026-04-11
updated: 2026-04-11
sources: [Kronos.md, TradingAgents.md, TimesFM.md]
reliability: medium
---

# 금융 AI

LLM·에이전트·파운데이션 모델을 금융 데이터 분석, 트레이딩, 예측에 적용하는 분야. 2024~2026년 급성장 중.

> [!insight] 핵심 인사이트
> 멀티에이전트 조직 모방([[TradingAgents]]) + 시계열 파운데이션 모델([[TimesFM]], [[Kronos]])이 동시에 트렌딩. 금융 AI는 "단일 모델 예측"에서 "에이전트 조직 시뮬레이션"으로 패러다임 전환 중.

## 주요 접근 방식

### 1. 시계열 파운데이션 모델
- 대규모 금융 데이터로 사전학습 → 제로샷 예측
- 대표: [[Kronos]], [[TimesFM]]

### 2. 멀티에이전트 트레이딩
- 실제 트레이딩 조직 구조를 에이전트로 모방
- 분석가·리스크·트레이더 역할 분리로 단일 LLM 편향 완화
- 대표: [[TradingAgents]]

### 3. 금융 특화 LLM
- Bloomberg GPT, FinGPT 계열
- 금융 텍스트(뉴스, 공시, 리포트) 이해 특화

## 신뢰도 기준

> [!warning] 금융 AI 평가 주의사항
> - 백테스트 수익률 ≠ 실거래 수익률
> - 특정 기간·시장 최적화된 결과가 일반화될 수 없음
> - "샤프지수 우수" 클레임은 항상 시기·마켓 컨텍스트 확인

## 관련 페이지

- [[Kronos]]
- [[TradingAgents]]
- [[TimesFM]]
- [[시계열-예측-파운데이션-모델]]
- [[AI-에이전트-프레임워크]]
