---
title: AI 3D 생성
type: concept
domain: slam-3dgs
tags: [3d-generation, ai-3d, meshy, tripo, blender, slam-3dgs]
created: 2026-04-10
updated: 2026-04-10
sources: [instagram-저장-2026-02-2026-04.md]
reliability: medium
---

# AI 3D 생성

2026년 현재 AI 기반 3D 모델 생성 기술 지형도. 단일 이미지나 텍스트 프롬프트에서 고품질 3D 에셋을 수 초 안에 생성하는 것이 가능해졌다.

---

## 핵심 인사이트

> [!insight] 사진 한 장 → 3D 세계 탐험 가능
> "3D worlds from one photo. You can now turn any image into a 3D world you can walk through." OpenArt 기반. 단순한 메시 생성이 아닌, 인터랙티브 3D 공간으로 변환.

> [!insight] 이미지 → 여러 개의 분리된 메시 자동 생성
> [[Tripo]]의 핵심 기능: 단일 이미지에서 부위별로 분리된 메시 생성 → Blender 후작업 편집성 대폭 향상.

---

## 주요 도구

### 범용 3D 생성
- **[[Meshy]]**: 텍스트 프롬프트 / 이미지 → 3D 모델. 제품 디자인 특화.
- **[[Tripo]]**: 단일 이미지 → 여러 분리 메시. Blender 연동 강점.

### 캐릭터 특화
- **나노바나나2** (Nano Banana 2): 사진 한 장 → 3D 캐릭터. 무료. 한국어 워크플로우 공유 활발.
  - 워크플로우 + 프롬프트: @beyondbetterbrand (2026-04-07)

### 이미지 → 3D 공간
- **OpenArt**: 이미지 → 워크스루 가능한 3D 세계 생성 (@keanu.visuals, 2026-03-29)

### AI + Blender 통합
- **Claude + Blender MCP**: Claude에 자연어로 명령 → Blender가 3D 씬 자동 생성
  - "Connecting Claude with Blender using MCP, creators can control Blender through simple text commands"
  - → [[Claude-Blender-MCP]]

---

## 워크플로우 패턴

### 패턴 1: 캐릭터 굿즈 제작
```
스케치 → 나노바나나2 / Tripo.ai → 3D 모델
→ Blender (후작업, 애니메이션) → 영상 / 출력
```

### 패턴 2: 제품 프로토타이핑
```
레퍼런스 이미지 → Meshy → 3D 프로토타입
→ 디자인 검토 → 수정
```

### 패턴 3: 스튜디오 영상 제작
```
텍스트 프롬프트 → Meshy / Tripo → 에셋 생성
→ After Effects / Blender → 최종 영상
```

---

## 도메인 인사이트 (slam-3dgs 템플릿)

**현재 SOTA:**
- 무료: 나노바나나2 (캐릭터), OpenArt (이미지→3D 공간)
- 유료: Meshy (제품 디자인), Tripo (메시 분리)

**실시간 가능성:**
- 현재 도구들은 오프라인 생성 (수 초~수 분) — 실시간 30fps+ 아직 아님

**응용 가능성:**
- 영상 SaaS에 3D 에셋 자동 생성 통합
- 1인 크리에이터의 3D 콘텐츠 제작 비용 제거

---

## 관련 페이지

- [[Meshy]] — 텍스트/이미지 → 3D
- [[Tripo]] — 단일 이미지 → 분리 메시
- [[Claude-Blender-MCP]] — Claude로 Blender 제어
- [[AI-영상-생성-2026]] — 3D 에셋이 연결되는 영상 파이프라인

## 원본

- @beyondbetterbrand (2026-04-07): 나노바나나2 워크플로우
- @meshy.ai (2026-04-07): Meshy 기능 소개
- @re4ee (2026-04-03): Tripo.ai 메시 분리
- @keanu.visuals (2026-03-29): 사진 → 3D 세계
- @appinventiv4ai (2026-03-09): Claude + Blender MCP
- @ho_bum_ (2026-02-23): Sketch → Tripo → Blender
- 신뢰도: ⭐⭐ (소셜미디어 소스, 일부 직접 검증 필요)
