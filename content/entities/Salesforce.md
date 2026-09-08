---
title: Salesforce AI Research — 기존 패러다임을 "무용"으로 증명하는 쪽의 연구소
type: entity
domain: ai-news
tags: [entity, research-lab, salesforce, inference, agent]
created: 2026-09-04
updated: 2026-09-04
sources: [Random-Attention.md, DarwinX.md, StateAct.md]
reliability: high
---

# Salesforce AI Research

**성격**: SaaS 기업의 사내 AI 연구 조직. 볼트 관측 범위에서는 **추론 효율·에이전트 상태 관리** 축에 반복 등장.

> [!insight] 볼트 관측 패턴
> [[Random-Attention]](2026-09-04)에서 **KV 캐시 축출의 점수 계산이 사실상 무용**임을 통제 실험으로 보였다 — 새 기법을 더하는 대신 **기존 기법이 하던 일이 불필요함을 증명**하는 형태의 기여다. 코드도 함께 공개(`SalesforceAIResearch/Random-Attention`). 볼트에 축적된 [[StateAct]]·[[DarwinX]] 와 함께 **"에이전트를 무겁게 만들지 않는 방향"** 의 연구가 이 조직의 결이다.

## 볼트 내 소스
- [[Random-Attention]] — KV 캐시 무작위 축출, vLLM 처리량 32~43%↑ (2026-09-04)
- [[StateAct]] · [[DarwinX]]

## 관련 페이지
- [[에이전트-메모리-레이어]] · [[AI-에이전트-프레임워크]] · [[AttentionSink]] · [[LMCache]]

## 원본
- HF org: https://huggingface.co/Salesforce
