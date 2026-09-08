---
title: Claude Code 워크플로우
type: concept
domain: ai-news
tags: [claude-code, workflow, automation, dotclaude, ai-coding]
created: 2026-04-10
updated: 2026-04-10
sources: [instagram-저장-2026-02-2026-04.md]
reliability: high
---

# Claude Code 워크플로우

[[Claude Code]]를 단순 챗봇이 아닌 **엔지니어링 시스템**으로 사용하는 방법. 2026년 상반기 기준, 1인이 Claude Code + 올바른 설정만으로 연 5억 인건비를 대체하는 자동화 시스템을 구축하는 사례가 다수 등장.

---

## 핵심 인사이트

> [!insight] .claude/ 폴더가 핵심이다
> "Most people use Claude like a chatbot. This is how you turn it into a real engineering system. The .claude/ folder." CLAUDE.md, 메모리, 훅을 올바르게 설정하면 Claude Code가 컨텍스트를 유지하며 자율적으로 작동.

> [!insight] Claude Max $200으로 SaaS 25개 대체
> @ianpark.vc: "지난 한 달, Claude로 25개 넘는 도구를 직접 만들었다. 그중 15개는 매일 쓴다. 비용은 단 하나, Claude Max $200 구독료뿐이었다." 기존 SaaS 스택을 Claude로 내부화하는 트렌드.

> [!insight] .claude/ 폴더 설정 하나로 상용 제품 2개 출시
> @leadgenman: "This .claude folder setup has shipped two live products to production. Content gets planned through a trend radar, code auto-commits after every single change." 트렌드 감지 → 코드 자동 커밋까지 완전 자동화.

---

## .claude/ 폴더 구조

```
.claude/
├── CLAUDE.md          ← 프로젝트 컨텍스트, 규칙, 역할 정의
├── memory/            ← 대화 간 지속 메모리
├── skills/            ← 재사용 가능한 스킬 정의
└── hooks/             ← 이벤트 기반 자동화 훅
```

핵심: CLAUDE.md에 역할, 규칙, 금지사항을 명확히 정의할수록 Claude의 자율성이 높아짐.

---

## 토큰 최적화

**RTK CLI 프록시** (90% 토큰 절감):
- "RTK: a high-performance CLI proxy that filters and compresses command outputs before they reach Claude Code."
- npm run, git 명령어 출력을 압축 후 전달
- 링크: @codingknowledge (2026-04-08)

> [!action] RTK 설치 검토
> Claude Code 토큰 소비가 많다면 RTK CLI 프록시 적용. 90% 절감 주장은 과장일 수 있으나 실질적 개선 효과 있음.

**기타 절감 방법**:
- npm/git 명령어를 Claude에게 직접 타이핑하지 말고 `.claude/hooks/`로 자동화
- 불필요한 파일 컨텍스트 로드 방지

---

## 음성 코딩 (Voice-to-Code)

> [!note] 폰으로 말하면 코드가 나온다
> "A developer built a walkie-talkie app on his phone. Voice-to-text streams directly to Claude Code terminal sessions. Speak into phone, code appears on screen." 이동 중에도 코딩 가능.

---

## PWA 자동 배포

Claude로 만든 웹앱을 스마트폰 앱으로 전환:
1. Claude로 웹앱 제작
2. PWA(Progressive Web App) 설정 추가
3. 앱스토어 제출 없이 스마트폰 홈화면에 설치 가능

사례: @suho_hp — Claude 웹앱 → PWA → 스마트폰 앱 (2026-04-04)

---

## Blender MCP 통합

```
Claude Code (자연어 명령)
    ↓
MCP (Model Context Protocol)
    ↓
Blender (3D 씬 자동 생성)
```

"By connecting Claude with Blender using MCP, creators can control Blender through simple text commands." → [[Claude-Blender-MCP]]

---

## 실사용 성과 사례

| 사례 | 기간 | 결과 |
|---|---|---|
| @ug0_builds | 3시간 | 앱스토어 출시 |
| @creyongfarm | 하루 | 수강신청 + 결제 연동 홈페이지 |
| @leadgenman | N/A | 상용 제품 2개 출시 |
| @ianpark.vc | 한 달 | SaaS 25개 → Claude 내부화 |

---

## After Effects MCP

> [!insight] AE MCP로 After Effects를 자연어로 제어
> @hyun2xyz: "애프터이펙트 mcp 잠깐 테스트 했는데 놀라워서 숨이 턱-막혀요" — Claude Code가 After Effects를 직접 제어. 모션 디자인 워크플로우의 자동화 가능성. → [[After-Effects-MCP]]

---

## 관련 페이지

- [[바이브코딩]] — 비개발자도 앱을 만드는 패러다임
- [[Higgsfield]] — Claude Code와 통합한 영상 자동화
- [[AI-영상-생성-2026]] — 영상 도메인 자동화 파이프라인
- [[LLM-Wiki]] — Claude Code로 운영하는 이 위키

## 원본

- @techwith.ram (2026-03-30): .claude/ 폴더 엔지니어링 시스템
- @leadgenman (2026-04-07): .claude/ 폴더로 제품 2개 출시
- @ianpark.vc (2026-04-09): Claude Max $200, SaaS 25개 대체
- @codingknowledge (2026-04-08): RTK 90% 토큰 절감
- @fastcampus (2026-04-09): Claude Code 자동화 시스템
- 신뢰도: ⭐⭐⭐ (다수 실사용자 사례 확인)
