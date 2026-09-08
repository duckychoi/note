---
title: Higgsfield 심층 분석 — Cinema Studio & Chat 노하우
type: entity
domain: video-saas
tags: [higgsfield, cinema-studio, soul-cast, prompt-engineering, 심층분석, video-saas]
created: 2026-04-10
updated: 2026-04-10
sources: [instagram 저장 영상 직접 분석 — Gemini 2.5 Pro, higgsfield.ai 직접 확인]
reliability: high
---

# Higgsfield 심층 분석 — Cinema Studio & Chat 노하우

> 분석 방법: 저장된 인스타그램 영상 5개를 Gemini 2.5 Pro 멀티모달로 직접 분석
> 단순 기능 나열이 아닌 **실제 화면/프롬프트/워크플로우**에서 추출한 인사이트

---

## 1. Cinema Studio — 핵심 메커니즘

### Soul Cast + Scene의 구조

> [!insight] Cinema Studio는 "캐릭터"와 "모션"을 분리해서 합성한다
> Soul Cast = 캐릭터 외형 (정면 사진 1장으로 추출)
> Scene = 모션/카메라워크 구조 (실제 촬영 영상이 뼈대)
> AI가 Scene의 모션을 유지하면서 외형만 Soul Cast로 교체

```
[Soul Cast 이미지 1장]           [Scene 클립 N개]
  캐릭터 외형 고정                 카메라/액션 구조
       ↓                                ↓
       └──────────── AI 합성 ───────────┘
                         ↓
              [새로운 일관된 캐릭터 영상]
```

**실제 확인된 사례:**
- 스파이더맨 의상 남성 (Soul Cast) + 도시 스윙/점프/착지 클립 6개 (Scene)
  → AI가 배우를 교체하면서 모든 액션 시퀀스를 해당 캐릭터로 재현
- 의상(보라색 점퍼, 청바지), 체형, 헤어스타일 전 씬 일관 유지

### Soul Cast 이미지 노하우

> [!action] Soul Cast 최적 이미지 조건
> - **정면/3/4 앵글 사진**이 가장 높은 캐릭터 일관성
> - 조명: 스튜디오 조명처럼 고른 조명 > 강한 명암 대비
> - 배경: 단순할수록 캐릭터 인식 정확도 향상
> - 해상도: 얼굴이 선명하게 보일 것

### Scene 클립 노하우

> [!insight] Scene은 창의적 생성이 아닌 "구조 템플릿"으로 사용된다
> 원하는 카메라워크와 액션이 담긴 **실제 촬영 영상**을 Scene으로 사용하면 제어력 극대화.
> 스톡 영상 사이트(Pexels, Shutterstock)의 액션 클립을 Scene으로 활용하는 것이 핵심 워크플로우.

---

## 2. 프롬프트 엔지니어링 — 실제 화면에서 추출

### 패턴 1: 카메라 구도 대문자 명시 (Soul Cinema Studio)

실제 영상에서 확인된 프롬프트:

```
THREE-QUARTER VIEW, angyal, moody, handsome man with black hair,
heavy smoky eye makeup, tribal neck tattoo, and arm tattoos,
wearing a...

EXTREME LOW ANGLE WIDE LENS, Rebellion in women's a black combat...

OVER THE SHOULDER SHOT (TIGHT), Rebellious woman with...
```

**구조 템플릿:**
```
[카메라 구도 — 대문자] [분위기 형용사] [인물 외모 상세] [의상/설정]
```

**자주 쓰는 구도 키워드:**
- `THREE-QUARTER VIEW` — 인물 3/4 앵글
- `EXTREME LOW ANGLE WIDE LENS` — 극단적 로우앵글
- `OVER THE SHOULDER SHOT (TIGHT)` — 어깨 너머 클로즈업
- `EXTREME CLOSE-UP` — 극클로즈업
- `MEDIUM SHOT` — 미디엄 샷
- `WIDE ESTABLISHING SHOT` — 넓은 배경 샷

> [!insight] 구도를 대문자로 맨 앞에 쓸수록 결과가 더 정확해진다
> 소문자로 "close-up shot"보다 대문자 "EXTREME CLOSE-UP"이 프롬프트 우선순위 더 높게 처리됨

---

### 패턴 2: 타임스탬프 구간 제어 (invideo + Seedance)

> [!insight] 가장 강력한 프롬프트 기법 — 타임스탬프로 씬을 스토리보드화
> 2초 단위로 장면을 지정하면 AI 영상을 스토리보드처럼 정밀 제어 가능

실제 영상에서 추출한 완전한 프롬프트 구조:

```
Create a 15-second stylized 3D animated magic scene using @image as full
identity, character, and world.

0:00-0:02:
Extreme close-up. The magician performs fast flourish cuts.
The deck splits, rotates, and recombines.

0:02-0:04:
Camera pulls back to medium shot. He spreads the deck toward camera,
but keeps most cards slightly angled away. Only ONE card clearly visible:
Ace of Hearts. He leans forward with a playful, challenging expression.

0:04-0:06:
Push-in. A different person's hand (from camera POV) reaches in naturally
and grabs the ONLY clearly visible card: Ace of Hearts.

0:06-0:08:
The magician cleanly inserts it into the middle of the deck.
Camera stays tight on the hands so the insertion feels clear and fair.

0:08-0:10:
Extremely fast, stylish shuffle sequence. Multiple flourish shuffles,
snap cuts, and elegant hand transfers.

0:10-0:12:
With a sharp flick, he launches the deck upward. Cards burst outward
and float dramatically in mid-air around him.

0:12-0:13.5:
Quick reaction montage. Audience stares upward, wide-eyed and shocked.

0:13.5-0:15:
He slowly pulls out one card. It is clearly the Ace of Hearts.

Rules:
- The selected card must always be Ace of Hearts
- No card duplicates
- The final revealed card must be the same Ace of Hearts

Tone: playful, impressive, family-friendly, polished

Camera: dynamic cinematic coverage, clean push-ins, close-ups for card
logic, wider shots for magical payoff

Negative prompt:
magician selecting the card himself, wrong card, duplicate Ace of Hearts,
unreadable reveal, broken card
```

**5섹션 구조 템플릿:**
```
[요약 지시 + @image 참조]

[타임스탬프 구간별 상세 씬]
0:00-0:02: ...
0:02-0:04: ...

Rules:
- 논리 일관성 제약 조건

Tone: 분위기/감성 키워드

Camera: 촬영 스타일 전반

Negative prompt: 자주 발생하는 오류 방지
```

> [!action] @image 키워드 사용법
> 프롬프트 내 `@image`를 쓰면 업로드한 이미지를 캐릭터/세계관 기준으로 참조.
> "using @image as full identity, character, and world" 구문이 캐릭터 고정에 핵심.

---

### 패턴 3: Manga → Live Action (Seedance 2.0)

> [!insight] 만화 패널 자체가 복합 프롬프트 역할을 한다
> 텍스트 입력창 없이, 여러 장의 만화 패널을 순서대로 업로드하면
> 캐릭터 디자인 + 구도 + 대사 + 감정 흐름이 자동으로 실사 영상으로 변환

**실제 확인된 파이프라인:**
```
만화 패널 1: 조련장 아침 훈련 장면 + 대사
만화 패널 2: 마법 대결 선언 대사
만화 패널 3: 선생님 등장 + 감정 충돌
        ↓
Seedance 2.0
        ↓
실사 단편 영화 (HD, 전문 조명, 캐릭터 일관성 유지, 음향 포함)
```

말풍선 대사가 자동으로 음성/자막으로 변환됨.

---

## 3. Claude Code + Higgsfield 완전 자동화 파이프라인

> [!insight] Claude Code가 픽셀 좌표로 Higgsfield UI를 직접 조작한다
> 스크린샷 인식 → `Clicked at (305, 759)` 같은 좌표 클릭 → 완전 무인 자동화

### 실제 확인된 에이전트 로그

```
Used 2 tools
Navigated to https://higgsfield.ai
Clicked at (305, 759)
Scrolled down by 3 ticks
Update Todo list [✓]
```

### 전체 자동화 파이프라인 (도리토스 예시)

```
INPUT: 제품 이미지 1장 + 명령어
"generate 50 brand assets for this product."
    ↓
[Step 1] 에이전트: 소셜미디어 경쟁사 리서치 + 10가지 광고 컨셉 수립
    → "The Last Chip", "Launch Shatter", "Cheese Pull Still Life" 등
    ↓
[Step 2] 에이전트: higgsfield.ai 자동 접속
    → Seedream 4.5 모델 선택
    ↓
[Step 3] 에이전트: 컨셉별 상세 프롬프트 자동 작성 + 입력
예시 프롬프트:
"A Doritos Nacho Cheese chip captured at the exact moment of breaking apart,
fragments suspended mid-air in a freeze-frame explosion. Orange cheese dust
particles create a glowing cloud around the shatter point. Deep red-to-black
gradient background..."
    ↓
[Step 4] 에이전트: Generate 버튼 자동 클릭 × 10회+
    ↓
[Step 5] 사용자 명령: "download all of them and save it local folder downloads"
    ↓
[Step 6] 에이전트: 전체 이미지 자동 다운로드 → 로컬 폴더 저장
    ↓
[Step 7] 에이전트: Video 탭 이동 → Kling 3.0 선택 → 베스트 이미지 → 비디오 변환
    ↓
OUTPUT: 50개 브랜드 에셋 (이미지 + 비디오)
```

**사용자 입력량:** 명령 2줄 + 제품 이미지 1장
**AI 작업량:** 리서치 + 기획 + 프롬프트 작성 + 생성 + 다운로드 전체

---

## 4. 실제 영상 제작 워크플로우 (전체 파이프라인)

### 패션 필름 제작 (@sferro21 확인)

```
[1] Soul Cast 생성
    → 주인공 얼굴 사진 1장 업로드 → Soul ID 추출

[2] Soul Cinema Studio 스타일 시트 생성
    → 카메라 구도 대문자 프롬프트로 수십 장 스틸 이미지 생성
    → 각 씬 설정: 사막/불꽃 런웨이/달 표면 등 배경 다양화

[3] Kling 3.0 Image-to-Video
    → 스틸 이미지들을 3~5초 영상 클립으로 변환

[4] Adobe Premiere Pro 편집
    → V1~V6 비디오 트랙에 수십 클립 배치
    → 음악/SFX 추가 (A1~A5 오디오 트랙)

[5] 최종 패션 필름 완성
    → 실제 제작비 대비 99% 절감
```

**Premiere Pro 타임라인 구조 (실제 확인):**
- 비디오 트랙: V1~V6 (6개 레이어)
- 오디오 트랙: A1~A5 (5개 레이어)
- 클립명 패턴: `shot 67.mp4`, `shot1.mp4` (씬 번호 순서)
- 프로젝트 구조: `Soul cinema.prproj\Bin`

---

## 5. UI/UX 디테일 — 실제 확인된 요소

### Higgsfield 메인 탭 구조

```
상단 네비게이션:
[Image] [Video] [Audio] [Chat New] [Edit]

우측 버튼:
[Buy Credits] [Assets]

이미지 생성 화면:
- 프롬프트: "Describe the scene you imagine"
- 모델 드롭다운: "Seedream 4.5" (기본값)
- 그리드: 4열 이미지 갤러리

Video 탭 활성화:
- 모델 선택: "Kling 3.0"
```

### Cinema Studio 3패널 레이아웃

```
┌─────────────────────────────────────┐
│   FINAL RESULT (핑크 테두리 강조)    │
│              50%                     │
├──────────────────┬──────────────────┤
│  SCENE CLIPS     │  SOUL CAST       │
│  2×3 그리드      │  캐릭터 이미지   │
│  6개 클립        │  1장             │
└──────────────────┴──────────────────┘
```

---

## 6. 핵심 노하우 요약 (바로 쓸 수 있는 것)

| 기법 | 방법 | 효과 |
|---|---|---|
| 카메라 구도 대문자 | `EXTREME CLOSE-UP,` 앞에 위치 | 구도 제어 정확도 ↑ |
| 타임스탬프 제어 | `0:00-0:02:` 구간별 씬 지시 | AI 영상 스토리보드화 |
| @image 앵커 | `using @image as full identity` | 캐릭터/세계관 고정 |
| Rules 섹션 | 논리 오류 명시적 금지 | 품질 일관성 ↑ |
| Negative prompt | 자주 발생 오류 타입 열거 | 실패 확률 ↓ |
| Soul Cast 최적화 | 정면/고른 조명/단순 배경 | 캐릭터 일관성 ↑ |
| Scene = 스톡 영상 | 스톡 액션 클립을 뼈대로 사용 | 카메라/액션 제어 ↑ |
| Claude Code 자동화 | 픽셀 좌표 기반 UI 제어 | 대량 에셋 무인 생성 |

---

## 7. 구현 시 우선 재현해야 할 것

> [!action] 최우선 구현 — @멘션 캐릭터 앵커링
> `@image` + `@캐릭터명` 시스템. 한 번 만든 캐릭터를 모든 씬에 참조. 이것이 캐릭터 일관성의 핵심.

> [!action] 타임스탬프 프롬프트 에디터
> 사용자가 `0:00-0:02:` 구간별로 씬을 작성할 수 있는 에디터 UI. 스토리보드와 프롬프트의 통합.

> [!action] Rules/Negative prompt 섹션 UI
> 생성 오류 방지를 위한 구조화된 제약 조건 입력창. 자유 텍스트가 아닌 섹션 분리 UI.

> [!action] Scene + Soul Cast 분리 업로드 UI
> 두 개의 독립된 드롭존: 왼쪽(Scene 클립들), 오른쪽(Soul Cast 이미지). 직관적 구조.

---

## 관련 페이지

- [[Higgsfield-벤치마킹]] — 전체 기능 구조 개요
- [[Higgsfield]] — 엔티티 기본 정보
- [[Claude-Code-워크플로우]] — 자동화 파이프라인
- [[AI-영상-생성-2026]] — 경쟁 도구 지형도
- [[Seedance]] — 영상 생성 모델
