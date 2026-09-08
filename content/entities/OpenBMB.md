---
title: OpenBMB
type: entity
domain: local-llm
tags: [local-llm, edge-ai, slm, openbmb, on-device, minicpm]
created: 2026-07-10
updated: 2026-07-10
sources: [MiniCPM5-1B.md, MiniCPM-V-4.6.md]
reliability: high
---

# OpenBMB

> [!insight] 핵심 인사이트
> **온디바이스/엣지용 소형 LLM(SLM)에 특화된 오픈 연구·개발 조직**으로, 이 위키에서는 **MiniCPM 시리즈**의 제작사로 등장한다. ①**[[MiniCPM5-1B]]**(1.08B, 131K 컨텍스트, think/fast 2모드, 도구호출, RL+OPD로 수학·코드 +16점) — "1B급 오픈 SOTA"를 표방하는 순수 텍스트 SLM, ②**[[MiniCPM-V-4.6]]** — 동급 스케일 멀티모달 VLM. OpenBMB의 일관된 노선은 "**작게 만들되 도구 사용·추론까지 넣어 실제 온디바이스에서 쓸모 있게**" — BF16/GGUF/MLX 다포맷과 FlagOS 다칩 지원으로 CPU·스마트폰·임베디드 배포를 겨냥. 내 [[에이전트-메모리-레이어]]·경량 에이전트 백엔드 후보 공급원.

## 관련 소스
- [[MiniCPM5-1B]] — 1.08B 온디바이스 SLM (131K, think/fast, RL+OPD) *(2026-07-10 갱신)*
- [[MiniCPM-V-4.6]] — 동급 멀티모달 VLM

## 관련 페이지
- [[에이전트-메모리-레이어]] — 경량 로컬 에이전트 백엔드 후보
- [[온폴리시-증류]] — MiniCPM5 학습에 쓰인 OPD 흐름
- [[Qwen3.6-27B]] — 오픈 멀티모달 계열(스케일 대극)
- [[local-llm]] · [[ai-news]]

## 원본
- 조직: OpenBMB (온디바이스 SLM 오픈 개발)
- 대표 산출물: [[MiniCPM5-1B]], [[MiniCPM-V-4.6]] (MiniCPM 시리즈)
- 신뢰도: ⭐⭐⭐⭐ (다운로드 수십만+ 지속, 다포맷 오픈 배포)
