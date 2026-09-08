---
title: After Effects MCP
type: concept
domain: ai-news
tags: [after-effects, mcp, motion-design, claude-code, automation]
created: 2026-04-10
updated: 2026-04-10
sources: [instagram-저장-2026-02-2026-04.md]
reliability: medium
---

# After Effects MCP

MCP(Model Context Protocol)를 통해 Claude 등 AI가 After Effects를 자연어로 직접 제어하는 통합. 모션 디자인 워크플로우의 자동화를 크게 앞당길 수 있다.

---

## 핵심 인사이트

> [!insight] "놀라워서 숨이 턱-막혀요"
> @hyun2xyz (2026-04-09): "애프터이펙트 mcp 잠깐 테스트 했는데 놀라워서 숨이 턱-막혀요" — 실제 모션 디자이너가 처음 접하고 받은 인상. 자연어 명령으로 AE를 제어할 수 있음을 시사.

---

## 작동 방식

```
사용자 (자연어): "로고 페이드인 1초, 바운스 이징으로"
    ↓
Claude Code + MCP
    ↓
After Effects API
    ↓
AE 타임라인에 키프레임 자동 생성
```

[[Claude-Blender-MCP]]와 동일한 MCP 아키텍처. Blender에서 이미 작동하는 패턴이 After Effects로 확장된 사례.

---

## 모션 디자인 워크플로우 영향

**기존 워크플로우:**
- 키프레임 수동 설정
- 이징 곡선 조정
- 레이어 속성 직접 입력

**MCP 통합 후 가능성:**
- 자연어로 애니메이션 의도 설명
- 반복 작업 자동화 (로고 애니메이션, 타이틀 시퀀스)
- 스크립트 없이 복잡한 AE 효과 적용

> [!question] 미확인 사항
> After Effects MCP의 정확한 구현 도구/레포지토리가 무엇인지 확인 필요. @hyun2xyz 영상에서 실제 사용 장면이 있으나 도구명 불명확.

> [!action] AE MCP 조사
> After Effects MCP 오픈소스 프로젝트 조사. GitHub에서 "after-effects MCP" 검색. 현재 워크플로우에 통합 가능 여부 판단.

---

## 관련 모션 디자인 인사이트

### GeoLayers 지도 애니메이션
@fxpoob: "GeoLayers Map Animation + Montage for a corporate presentation — blending geospatial data with cinematic motion design in After Effects." 지리 데이터 + 영화적 모션디자인 결합.

### Smear Animation 기법
@motionbyakash: "Smear animation is one of those motion design tricks that looks complex — but once you know the setup, it takes less than a minute. Instead of animat[ing every frame]..." 복잡해 보이지만 설정만 알면 1분 안에 완성.

---

## 관련 페이지

- [[Claude-Blender-MCP]] — Blender에서 동일 MCP 패턴
- [[Claude-Code-워크플로우]] — MCP 활용 전체 맥락
- [[AI-영상-생성-2026]] — AI 기반 콘텐츠 제작 지형도

## 원본

- https://www.instagram.com/p/DW5HZUYj6bL/ (@hyun2xyz, 2026-04-09)
- https://www.instagram.com/p/DWeEIvOjxkB/ (@fxpoob, 2026-03-29): GeoLayers
- https://www.instagram.com/p/DWyq70TE6w8/ (@motionbyakash, 2026-04-06): Smear animation
- 신뢰도: ⭐⭐ (단일 실사용자 반응, 도구 세부 정보 확인 필요)
