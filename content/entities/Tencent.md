---
title: Tencent
type: entity
domain: ai-news
tags: [ai-news, tencent, agent-infra, sandbox, hunyuan, entity]
created: 2026-07-04
updated: 2026-08-26
sources: [CubeSandbox.md, Multi-Layer-Agent-Red-Teaming.md, Distribution-wise-Rewards.md, Hy3.md, TencentDB-Agent-Memory.md]
reliability: high
---

# Tencent (텐센트)

> [!insight] 2026-08-26 — **실배포 증거를 동반한** 멀티모달 임베딩 오픈 릴리스 ([[WeMM-Embedding]])
> `WeMM-Embedding`(HF 데일리 **2위**·업49·저자 6인·arXiv 2608.24053·**초록 원문 실검증**·가중치·코드 공개 https://github.com/Tencent/WeMM-Embedding)이 관측됐다.
> **2B/4B/9B** 계열로 텍스트·이미지·비디오·시각문서·인터리브 입력을 지원하고 **출력 차원이 가변**이다. **2B가 기존 최상위 8B 오픈소스 베이스라인을 MMEB-v2에서 추월**하고 **9B가 전체 80.6으로 SOTA**.
> **Tencent 노선의 특징이 여기서 선명해진다 — 공개 벤치 점수가 아니라 배포 면적을 근거로 낸다.** 위챗 **채널스·공식계정·모먼트·이커머스**에 실배포됐고, **사내 26개 태스크 벤치 개선 + 온라인 A/B 14건 일관 개선**을 보고한다. 볼트에 쌓인 임베딩·검색 소스 중 **프로덕션 트래픽 증거를 제시한 드문 사례**이며, 앞서 기록한 [[CubeSandbox]](에이전트용 경량 샌드박스)와 함께 *"사내에서 쓰던 것을 열어 내보내는"* 패턴으로 일관된다.
> ⚠️ **사내 26태스크·A/B 14건의 수치가 원문에 없다**("substantial"·"consistent"라는 정성 서술뿐). **학습 데이터(curated data)는 비공개**라 2단계 레시피 재현은 불가. 한국어 성능 미검증.

> [!insight] 핵심
> Hunyuan(HY) 모델군·클라우드·에이전트 인프라를 보유한 중국 빅테크. 위키 맥락의 신규 신호는 **에이전트용 경량 동시성 샌드박스 [[CubeSandbox]]**(TencentCloud) — "LLM이 생성한 코드를 안전·병렬 실행"이라는 에이전트 스택의 숨은 병목을 겨냥. 앞서 [[HY-Embodied]]·[[HY-World-2.0]]·[[MegaStyle]] 등 모델·데이터 쪽 기여와 함께, 에이전트 **실행 런타임**([[CubeSandbox]])에 이어 **에이전트 보안**([[Multi-Layer-Agent-Red-Teaming]], AI-Infra-Guard)까지 확장 — "코드 실행+공급망 감사"로 에이전트 인프라 양면을 채우는 포지션.

> [!note] 2026-07-09 추가 — Hunyuan 오픈 모델 + 로컬 메모리 동시 배출
> 하루에 모델과 도구를 나란히 공개: **[[Hy3]]**(Hunyuan Hy3, 295B/21B 활성 MoE·256K·Apache 2.0, "스캐폴딩 간 편차 4% 이내" 주장)로 오픈 MoE 플래그십 경쟁에 참전하고, 같은 날 **[[TencentDB-Agent-Memory]]**(완전 로컬 에이전트 장기 메모리, MIT·TypeScript)로 [[에이전트-메모리-레이어]]의 실물 구현체를 냄. 실행 런타임([[CubeSandbox]])·보안([[Multi-Layer-Agent-Red-Teaming]])에 이어 **모델·메모리**까지 = 에이전트 인프라 전방위 커버.

## 관련 페이지
- [[Hy3]] — Hunyuan 295B/21B MoE 오픈 모델 (2026-07-09)
- [[TencentDB-Agent-Memory]] — 완전 로컬 에이전트 장기 메모리 (2026-07-09)
- [[CubeSandbox]] — 에이전트 코드 격리 실행 샌드박스
- [[Multi-Layer-Agent-Red-Teaming]] — 다층 에이전트 레드팀(AI-Infra-Guard)
- [[Distribution-wise-Rewards]] — Hunyuan 분포 보상 시각생성
- [[HY-Embodied]]
- [[HY-World-2.0]]
- [[AI-에이전트-프레임워크]]

## 원본
- 대표 레포: https://github.com/TencentCloud/CubeSandbox
- 신뢰도: ⭐⭐⭐⭐ (빅테크·다수 오픈 릴리스)
