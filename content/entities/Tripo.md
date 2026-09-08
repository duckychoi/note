---
title: Tripo
type: entity
domain: slam-3dgs
tags: [3d-generation, tripo, ai-3d, blender, slam-3dgs]
created: 2026-04-10
updated: 2026-04-10
sources: [instagram-저장-2026-02-2026-04.md]
reliability: medium
---

# Tripo

단일 이미지에서 3D 메시를 생성하는 AI 도구. Tripo.ai. 특히 **이미지 → 여러 개의 분리된 메시** 생성이 강점으로, Blender 후작업 워크플로우와 잘 연동된다.

---

## 핵심 인사이트

> [!insight] 단일 이미지 → 여러 메시 자동 분리
> "Upload a single image in Tripo.ai and get several separate meshes, so there's no need to split them manually." 기존엔 하나의 덩어리로 나오던 3D 모델을 부위별로 자동 분리. Blender에서 후작업 시 편집성 대폭 향상.

> [!insight] Sketch → Tripo → Blender 워크플로우
> 한국 크리에이터 @ho_bum_이 실제 사용한 파이프라인: "이미지 한 장으로 기본 모델을 생성하고, 후작업을 통해 영상까지 완성." 아이디어 스케치 → 최종 3D 영상까지 1인이 처리 가능.

---

## 표준 워크플로우

```
스케치 / 이미지 1장
    ↓
Tripo.ai (3D 메시 생성, 자동 분리)
    ↓
Blender (후작업, 리깅, 애니메이션)
    ↓
최종 3D 영상
```

---

## [[Meshy]]와 비교

| 항목 | Tripo | Meshy |
|---|---|---|
| 입력 | 단일 이미지 | 텍스트 프롬프트 / 이미지 |
| 강점 | 메시 자동 분리, Blender 연동 | 제품 디자인 프로토타이핑 |
| 후작업 | Blender 친화적 | 독립 사용 가능 |

---

## 활용 시나리오

- 캐릭터 피규어/키링 제작 (사례: 2026 붉은 말의 해 키링)
- 이미지 레퍼런스 → 3D 에셋 변환
- 소규모 스튜디오의 3D 제작 비용 절감

---

## 관련 페이지

- [[AI-3D-생성]] — AI 3D 도구 전체 지형도
- [[Meshy]] — 텍스트 기반 3D 생성
- [[Claude-Blender-MCP]] — Claude로 Blender 제어

## 원본

- https://www.instagram.com/p/DWp-5qrDPHf/ (@re4ee, 2026-04-03)
- https://www.instagram.com/p/DVFiihtgAwZ/ (@ho_bum_, 2026-02-23)
- 신뢰도: ⭐⭐ (실사용자 후기 기반)
