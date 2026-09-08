---
title: Moonshot AI — Kimi 시리즈 개발 중국 프론티어 랩
type: entity
domain: ai-news
tags: [ai-news, entity, china, frontier-lab, kimi, moe, open-weights]
created: 2026-07-28
updated: 2026-07-28
sources: [Kimi-K3.md, Kimi-K2.6.md, Kimi-K2.7-Code.md]
reliability: high
---

# Moonshot AI (月之暗面)

> [!insight] 핵심 인사이트
> **Kimi 시리즈**를 개발하는 중국 프론티어 AI 랩. [[GLM-5.2]]의 [[Zhipu-AI]], Qwen의 [[Alibaba]]와 함께 **오픈 웨이트 대형 MoE**로 클로즈드 프런티어를 추격하는 중국 축의 한 꼭짓점. 전작 [[Kimi-K2.6]]·[[Kimi-K2.7-Code]](코딩 특화)에 이어 2026-07-28 **[[Kimi-K3]]**(2.8T 총/104B 활성 MoE·1M 컨텍스트·네이티브 멀티모달)로 오픈 웨이트 규모 상단을 경신 — 논문·모델 모두 HF 트렌딩 1위. 특징적 아키텍처 노선: **Delta Attention·Attention Residuals·Stable LatentMoE** 등 어텐션·MoE 구조 혁신으로 스케일링 효율을 밀어붙임(K3 초록 K2 대비 ~2.5배 주장).

> [!note] 포지션
> Kimi K3 논문 초록은 스스로 **최강 독점모델(Claude Fable 5·GPT-5.6 Sol)에는 뒤진다**고 명시 — Moonshot의 전략은 "절대 SOTA"가 아니라 **오픈 웨이트로 프런티어 근접 + 1M·멀티모달·에이전틱 제공**. 코딩([[Kimi-K2.7-Code]])·멀티모달·롱호라이즌 에이전트를 중점 겨냥.

## 관련 페이지
- [[Kimi-K3]] — 2.8T/104B 멀티모달 MoE (2026-07-28)
- [[Kimi-K2.6]]
- [[Kimi-K2.7-Code]]
- [[kimi-cli]]
- [[Zhipu-AI]] — 오픈 웨이트 중국 축 동료([[GLM-5.2]])
- [[Alibaba]] — Qwen, 중국 오픈 축
- [[ai-news]]

## 원본
- 대표 모델: https://huggingface.co/moonshotai/Kimi-K3
- 신뢰도: ⭐⭐⭐⭐ (Kimi K2/K3 실물 모델·논문 공개, HF 트렌딩 상위 실사용층 확인. 벤치 자체발표는 잠정)
