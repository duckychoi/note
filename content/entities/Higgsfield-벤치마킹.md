---
title: Higgsfield 전체 기능 벤치마킹
type: entity
domain: video-saas
tags: [higgsfield, benchmarking, cinema-studio, chat, video-saas, 벤치마킹]
created: 2026-04-10
updated: 2026-04-10
sources: [higgsfield.ai 직접 확인 2026-04-10]
reliability: high
---

# Higgsfield 전체 기능 벤치마킹

> 직접 확인 일자: 2026-04-10
> 출처: https://higgsfield.ai, https://higgsfield.ai/cinema-studio, https://higgsfield.ai/chat-intro

---

## 전체 아키텍처

```
Higgsfield
├── Image           ← 이미지 생성 (Soul, Nano Banana, Flux, GPT Image 등)
├── Video           ← 영상 생성 (Seedance, Kling, Sora, Veo 등 멀티모델)
├── Audio           ← 오디오
├── Edit            ← 인페인팅 / 편집 / 업스케일
├── Character       ← Soul ID 캐릭터 라이브러리
├── Moodboard       ← 스타일 레퍼런스
├── Cinema Studio   ← 스튜디오 통합 인터페이스 (핵심)
├── Apps            ← 원클릭 앱 100+개
├── Chat            ← 실시간 협업 공간 (New)
├── Original Series ← AI 시리즈 스트리밍 플랫폼
├── Community       ← 갤러리 / 피드
└── Assist/Copilot  ← AI 어시스턴트
```

---

## Cinema Studio 3.0

> [!insight] 씬 설명에 @멘션으로 캐릭터/로케이션을 불러오는 것이 핵심
> 단순 프롬프트 입력이 아니라 라이브러리에 저장된 캐릭터/장소를 @로 참조 → 일관성 자동 유지

### UI 구조

```
프로젝트 관리 (폴더)
    ↓
씬 설명 입력창
    ├── @ 멘션 → 캐릭터 라이브러리 자동완성
    ├── @ 멘션 → 로케이션 라이브러리 자동완성
    └── General / Character / Location 탭
    ↓
생성 옵션
    ├── Image / Video 토글
    ├── 비율 선택 (16:9, 2K 등)
    └── GENERATE
```

### 핵심 기능

- **캐릭터 일관성**: Soul ID로 만든 캐릭터를 @멘션으로 재사용
- **로케이션 라이브러리**: 배경/장소도 저장 후 @멘션 재사용
- **컬러 그레이딩 내장**: 생성 시 색보정 자동 적용
- **프로젝트 = 폴더**: 이미지+영상+오디오를 하나의 프로젝트로 통합 관리
- **tagline**: "What would you shoot with infinite budget?"

### 구현 포인트

- `@mention` 자동완성 → 캐릭터/로케이션 임베딩 삽입
- Character Library: Soul ID 캐릭터를 재사용 가능한 임베딩으로 DB 저장
- 프로젝트 단위 에셋 관리

---

## Higgsfield Chat

> [!insight] 생성 플랫폼 + 협업 공간 + 커뮤니티를 하나로
> ChatGPT 스타일 UI가 아닌 "공유 작업실" 개념. 생성 결과물이 채팅방에 바로 공유되고 피드백 → 재생성 루프가 한 화면에서 완결.

### 핵심 기능

**협업**
- 공유 프로젝트 (팀/친구/커뮤니티 단위)
- 실시간 생성 진행도 공유
- 프로젝트 내 채팅 + 코멘트
- 화상통화 내장 (플랫폼 이탈 없이)
- 에셋 공유 시 모델명+프롬프트+미리보기 메타데이터 함께 전달

**커뮤니티**
- 커뮤니티 피드 (퍼블리시 → 바이럴)
- 카테고리별 탐색 + 지역 기반 추천
- 프로젝트 공개 설정: private / public / open

**카르마 경제**
- 좋아요/댓글/생성/초대 → 카르마 획득
- 카르마 → 크레딧 전환

**조직(Organisation)**
- Free Org: 공유 이름, 개인 크레딧
- Team Plan (유료): 공유 크레딧 + SSO + 어드민 컨트롤

### 사용 시나리오

| 대상 | 활용 |
|---|---|
| 크리에이티브 팀 | 캠페인 공동 제작, 실시간 에셋 리뷰 |
| 커뮤니티 빌더 | 팬/협업자 모집, 오픈 프로젝트 |
| 마케터 | 클라이언트 에셋 공유/승인 워크플로우 |
| AI 영화 제작자 | 스토리보드 공동 작업, 씬 피드백 |
| 교육자 | 수강생 작업물 크리틱 |

---

## 멀티모델 전략 (핵심 해자)

> [!insight] Higgsfield는 자체 모델 개발보다 최고 모델들의 통합이 핵심 경쟁력
> 한 플랫폼에서 Sora 2, Kling 3.0, Seedance 2.0, Veo 3.1을 모두 사용 → 사용자가 이탈하지 않는 구조

### 영상 모델 라인업

- Seedance 2.0 / Fast / Pro (ByteDance)
- Kling 3.0 / 2.5 Turbo / 2.1 Master / o1 (Kuaishou)
- Sora 2 (OpenAI) — 독점 제공
- Veo 3.1 (Google)
- MiniMax Hailuo 02
- WAN 2.6 / 2.5

### 이미지 모델 라인업

- Soul 2.0 (자체): 패션/하이엔드
- Nano Banana Pro (자체): 4K 범용
- Seedream 5.0 Lite / 4.0 (ByteDance)
- Flux 2 (Black Forest Labs)
- GPT Image 1.5 (OpenAI)

---

## Apps 생태계 (100+)

> [!note] 각 App = 미리 세팅된 모델+프롬프트+후처리 파이프라인
> 신규 앱 추가가 곧 신규 기능처럼 인식됨. 무한 확장 구조.

**카테고리:**
- 얼굴/인물: Face Swap, Video Face Swap, Character Swap, Skin Enhancer, Relight
- 광고/상업: Click to Ad, Product Placement, Billboard Ad, Truck Ad, Packshot
- 콘텐츠: ASMR 시리즈, Meme Generator, Shots, Zooms, Style Snap
- 편집 유틸: Background Remover, Color Grading, Upscale, Expand Image

---

## VFX 프리셋 라이브러리 (100+)

**카테고리:**
- 트랜지션: Raven, Splash, Melt, Smoke, Flying Cam, Seamless
- 원소/마법: Air/Water/Earth/Fire Bending, Plasma Explosion
- 변신: Animalization, Werewolf, Cyborg, Disintegration
- 액션: Explosion, Giant Grab, Fast Sprint, Hero Flight

---

## Mixed Media 스타일 프리셋 (30+)

Sketch / Canvas / Flash Comic / Overexposed / Paper / Noir / Particles / Hand Paint / Vintage / Acid / Comic / Marble / Ocean 등

---

## 빌드 우선순위 (내가 만든다면)

> [!action] Phase 1 — MVP
> 멀티모델 영상/이미지 생성 API 통합 + 프로젝트 폴더 + 기본 크레딧

> [!action] Phase 2 — 차별화
> Cinema Studio (@멘션 캐릭터/로케이션) + Character Library + VFX 프리셋 10~20개

> [!action] Phase 3 — 커뮤니티
> Chat 공유 프로젝트 + 커뮤니티 피드 + 카르마 → 크레딧 시스템

> [!action] Phase 4 — 생태계
> Apps (원클릭 앱 빌더) + Original Series + Enterprise/Team Plan

---

## 경쟁 우위 빈틈 (한국 시장)

1. **한국어 특화 UI/UX** — 기존 도구 전부 영어 중심
2. **B2B 광고 자동화** — 중소기업 광고팀 대체 SaaS 수요 미성숙
3. **K-콘텐츠 특화 캐릭터** — Soul ID의 한국 패션/미용 버전

---

## 관련 페이지

- [[Higgsfield]] — 엔티티 기본 페이지
- [[AI-영상-생성-2026]] — 경쟁 도구 전체 지형도
- [[Seedance]] — 핵심 영상 모델
- [[Claude-Code-워크플로우]] — 자동화 파이프라인 연계
- [[video-saas 도메인]] — 도메인 누적 인사이트
