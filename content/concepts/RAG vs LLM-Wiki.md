---
title: RAG vs LLM-Wiki
type: concept
tags: [RAG, LLM-Wiki, 비교, knowledge-management]
created: 2026-04-09
updated: 2026-04-09
sources: [llm-wiki-karpathy-2026.md]
---

# RAG vs LLM-Wiki

두 접근의 철학적·실용적 차이 비교.

---

## RAG (Retrieval-Augmented Generation)

**작동 방식**: 소스 업로드 → 쿼리 시 관련 청크 검색 → 답변 생성
**예시**: NotebookLM, ChatGPT 파일 업로드, 대부분의 RAG 시스템

| 특성 | RAG |
|------|-----|
| 지식 축적 | ❌ 없음. 매번 재추출 |
| 교차 참조 | ❌ 쿼리마다 새로 조합 |
| 모순 감지 | ❌ 자동 감지 없음 |
| 복잡한 합성 | △ 5개 문서 합성 시 매번 새로 찾아야 함 |

---

## LLM-Wiki

**작동 방식**: 소스 추가 → LLM이 위키에 통합 → 쿼리 시 위키에서 검색
**예시**: Karpathy의 LLM-Wiki 패턴, Farzapedia

| 특성 | LLM-Wiki |
|------|----------|
| 지식 축적 | ✅ 복리 축적 |
| 교차 참조 | ✅ 이미 구성되어 있음 |
| 모순 감지 | ✅ 인제스트 시 자동 표시 |
| 복잡한 합성 | ✅ 위키에 이미 반영됨 |

---

## 핵심 차이 한 문장

> RAG: 지식을 매번 재발견한다.
> LLM-Wiki: 지식을 한 번 컴파일하고 계속 갱신한다.

---

## 언제 어떤 걸 쓸까

| 상황 | 추천 |
|------|------|
| 단발성 문서 Q&A | RAG |
| 시간에 걸쳐 축적되는 지식 | LLM-Wiki |
| 빠른 프로토타입 | RAG |
| 장기 연구/개인 지식베이스 | LLM-Wiki |
| 팀 내부 위키 | LLM-Wiki |

---

## 관련 페이지

- [[LLM-Wiki]]
- [[Andrej Karpathy]]
