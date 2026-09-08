---
title: OpenAI
type: entity
domain: ai-news
tags: [ai-news, openai, coding-agent, interop, entity]
created: 2026-07-04
updated: 2026-09-07
sources: [codex-plugin-cc.md, openai-codex.md]
reliability: high
---

# OpenAI

> [!update] 2026-08-23 — 본체 레포 [[openai-codex]] 편입 (⭐114,383·Apache-2.0·Rust)
> 그간 이 엔티티의 "대표 레포"는 브리지 플러그인 `openai/codex-plugin-cc`로 잡혀 있었으나, 2026-08-23 배치에서 **본체 `openai/codex`가 처음 위키에 편입**되며 정정한다 — ⭐**114,383**·포크 17,459·**Apache-2.0**·**Rust**·생성 2025-04-13·최종 푸시 2026-08-23(당일)로 전량 API 실검증.
> **이로써 OpenAI의 상호운용 전략이 한 겹 더 드러난다**: 경쟁사 CLI용 브리지([[codex-plugin-cc]])를 배포하는 데 그치지 않고, **자사 에이전트 본체 소스를 Apache-2.0으로 공개**하고 있다. 같은 배치에서 실측한 [[anthropic-claude-code]]가 **라이선스 미명시·본체 비공개**인 것과 정반대다 — 스타는 Anthropic이 앞서지만(142,680 vs 114,383), **개방성 축에서는 OpenAI가 우위**다.
> ⚠️단 개방성이 성능은 아니다. 같은 배치 [[SWE-bench-Science]]는 최고 성능 에이전트를 **Claude Code + Opus-5(max)**로 실측했고 codex는 상위에 없다.

> [!insight] 핵심
> GPT·Codex 개발사. 이 위키 맥락에서 주목점은 **경쟁사(Anthropic) CLI인 [[anthropic-claude-code]]용 플러그인 [[codex-plugin-cc]]를 공식 배포**했다는 것 — 코딩 에이전트 시장을 폐쇄 잠금이 아닌 **상호운용(interop)**으로 넓히려는 전략 신호. [[openai-agents-python]]·[[awesome-codex-skills]]와 함께 에이전트 도구를 Claude Code 생태계로 침투시키는 중.

---
## 🔴 2026-09-07 — 스킬 카탈로그 레포를 **폐기**하고 플러그인으로 이동

[[openai-skills]](⭐**25,754** · 2026-09-07 API 실측)의 README 최상단이 폐기를 선언한다 — *"This repository is **deprecated**. … use the **OpenAI Plugins repository**. … create a **skill-only plugin**."*

> [!insight] 배포 단위를 한 층 올렸다
> Codex의 스킬은 사라지지 않고 **플러그인 안으로 들어갔다**. [[Anthropic]] 의 [[claude-plugins-official]] 마켓플레이스와 **같은 방향**이며, 경쟁 벤더 둘이 **독립적으로 같은 배포 단위**에 도달했다. → [[에이전트-스킬]] 3층 재편.

> [!warning] 폐기 신호가 기계 판독 가능한 곳에 없다
> API `archived` 는 **`false`**, 라이선스는 **`null`**, 최종 push **2026-07-14**(약 2개월 정체)인데 **스타는 계속 붙는다**(raw 25,751 → 실측 25,754). **README를 읽어야만 폐기를 안다.**

> [!note] 폐기됐어도 남긴 관측
> README가 [[agentskills]](`agentskills.io`)를 **"Agent Skills open standard"** 로 명시 링크한다 — [[Anthropic]] 의 [[anthropics-skills]] 가 *"표준은 밖에 있다"* 고 선언한 것과 **동일한 외부 표준 참조**. **벤더 레포가 폐기돼도 표준 참조는 유지된다.**

⚠️ **추적 공백**: 후속인 `openai/plugins` 가 볼트에 **없다.** 다음 배치 수집 대상.

## 관련 페이지
- [[openai-codex]] — **본체 코딩 에이전트 CLI**(⭐114,383·Apache-2.0·Rust)
- [[codex-plugin-cc]] — Claude Code ↔ Codex 브리지 플러그인
- [[openai-agents-python]]
- [[awesome-codex-skills]]
- [[anthropic-claude-code]]
- [[AI-에이전트-프레임워크]]

## 원본
- 대표 레포: https://github.com/openai/codex (본체·Apache-2.0·Rust·⭐114,383 2026-08-23 API 실검증) / https://github.com/openai/codex-plugin-cc (브리지)
- 신뢰도: ⭐⭐⭐⭐⭐ (메이저 AI 랩)
