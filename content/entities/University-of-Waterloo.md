---
title: University of Waterloo — Program-as-Weights 계보를 밀고 있는 대학 연구 축
type: entity
domain: local-llm
tags: [entity, university, waterloo, program-as-weights, distillation, adapter]
created: 2026-09-04
updated: 2026-09-04
sources: [Compile-by-Training.md, Program-as-Weights.md, Code2LoRA.md, FORGE.md]
reliability: high
---

# University of Waterloo

**성격**: 볼트 관측 범위에서 **"프로그램을 가중치로 굳히는"** 계보를 **가장 일관되게** 밀고 있는 기관.

> [!insight] 볼트 관측 패턴 — 한 기관이 하나의 축을 4건 연속으로 판다
> [[Program-as-Weights]](PAW) → [[Compile-by-Training]](고정확도 컴파일 모드 추가) 로 이어지는 **직접 후속 관계**가 확인되고, [[Code2LoRA]]·[[FORGE]] 도 같은 **어댑터·증류** 결을 공유한다. 볼트에 쌓인 소스 대부분이 *일회성 기관*인 데 반해 **Waterloo는 축을 누적**하고 있다 — 다음 PAW 후속이 나올 때 **이 페이지가 계보를 잇는 지점**이 된다.
> 방향 요약: *반복되는 원격 LLM 호출을 **로컬에서 도는 재사용 가능한 아티팩트**로 바꾼다.* 스킬 생태계([[anthropics-skills]])가 **텍스트 지시문**으로 하는 일을, 이쪽은 **가중치**로 한다.

## 볼트 내 소스
- [[Program-as-Weights]] — 자연어 함수 설명 → 0.6B 로컬 인터프리터 위 신경 프로그램 (단일 forward, 수 초)
- [[Compile-by-Training]] — 교사 생성 예제로 어댑터 증류, FuzzyBench-Hard 의미 정확도 83.6% / 컴파일 약 1분 (2026-09-04)
- [[Code2LoRA]] · [[FORGE]]

## 관련 페이지
- [[온폴리시-증류]] · [[에이전트-메모리-레이어]] · [[에이전트-스킬]] · [[magnitude]] · [[PAWBench]]

## 원본
- HF org: https://huggingface.co/UWaterloo
