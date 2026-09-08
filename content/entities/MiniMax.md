---
title: MiniMax — 옴니모달 생성 시스템 H3를 내놓은 중국 AI 기업
type: entity
domain: ai-news
tags: [ai-news, entity, minimax, china, video-generation, omni-modal, hailuo, partial-open-source]
created: 2026-08-29
updated: 2026-08-29
sources: [MiniMax-H3.md, MiniMax-M2.7.md, MiniMax-M3.md, MiniMax-Sparse-Attention.md, MiniMax-H3-Turbo-Lora.md]
reliability: medium
---

# MiniMax (미니맥스)

> [!insight] 핵심 인사이트 — 볼트에서 **영상 생성 축을 지키는 사실상 유일한 대형 오픈 벤더**
> HF 트렌딩 상위가 텍스트·멀티모달 LLM([[Alibaba]] Qwen 계열 · [[Zhipu-AI]] GLM 계열)으로 뒤덮인 가운데, [[MiniMax-H3]]는 **상위권에서 영상 생성 축을 대표하는 유일한 대형 모델**이다(2026-08-29 HF 트렌딩 내 다운로드 **2위** · DL **5,018,833** · 좋아요 **4,591**).
> 제품 축은 **Hailuo AI**(hailuoai.video) 웹앱·데스크톱·API로 이어지며, **모델 공개와 상용 서비스가 같은 자산을 공유**하는 구조다. 이는 순수 연구 배포([[Alibaba]] apache-2.0)와도, 폐쇄 API([[OpenAI]])와도 다른 **중간 노선**이다.
> 볼트에는 텍스트 계열([[MiniMax-M2.7]]·[[MiniMax-M3]])과 아키텍처 연구([[MiniMax-Sparse-Attention]])도 축적돼 있어, **텍스트·영상 양 축을 모두 내는 소수 벤더** 중 하나로 분류된다.

> [!warning] ⚠️ **"부분 공개(open weights ≠ open system)" 패턴의 볼트 대표 사례**
> [[MiniMax-H3]] 모델카드 원문(2026-08-29 대조): 품질에 **결정적(critical to the quality of the final output)** 이라고 스스로 밝힌 전처리 모듈 **`H3-Context-IR`이 오픈소스 릴리스에 포함되지 않는다** — 다중 호스팅 모델·서비스에 의존하며 **API로만 제공**된다.
> → **공개 가중치만으로는 공식 데모 품질을 재현할 수 없다.** 데모·API 결과를 근거로 오픈 가중치 성능을 추정하면 **체계적으로 과대평가**하게 된다.
> 추가 제약: **희소 어텐션을 네이티브 지원한다면서 초기 공개는 full attention 추론만** 제공(희소 구현은 추후) · 예제 커맨드가 **`--num-gpus 4`** · 라이선스가 apache가 아닌 **커뮤니티 라이선스(other)** 로 **상용 조건 별도 확인 필요**.
> ⚠️ **공개 벤치마크(VBench·Elo 등)는 모델카드 전문 검색 결과 0건**이며, 볼트는 08-06 이후 **네 회차 연속** 이를 확인했다. **reliability medium**의 주된 근거다.

> [!note] 아키텍처 요지 ([[MiniMax-H3]] 모델카드 실측)
> **3모듈**: `H3-Context-IR`(멀티모달 입력 이해·정제 → 중간표현 · **비공개**) → `H3-Base`(768p 오디오·비디오 생성) → `H3-Regenerate-2K`(2K 재생성).
> `H3-Omni-Transformer`는 **33B 밀집 단일스트림**이며 그중 **약 13B가 AdaLN 분기** — AdaLN 변조 출력은 사전계산·캐시가 가능해 **추론 전용 배포엔 로드 불필요**(실질 추론 약 20B). 어텐션·FFN에 모달리티별 구조가 없고 모달리티 특화 파라미터는 **입출력 층과 AdaLN 분기에만** 국한된다.
> **인코더는 [[Alibaba]] Qwen3-VL-32B의 전체 사전학습 가중치**(50번째 층 은닉상태 사용) — 즉 "33B"는 생성기만의 수치이고 시스템 전체는 **32B 인코더가 추가로 얹힌다.**
> **출력**: 4~15초 · 24FPS · **32kHz 스테레오 네이티브 동기 오디오** · 짧은 변 기본 768px(2K는 Regenerate 경유) · 대사 **11개 언어** 안정.

## 도메인별 추출 (ai-news · 교차 [[video-saas]])

- **신뢰도**: ⭐⭐ medium — 지표·모델카드 스펙은 **API/원문 실검증(high)** 이나 **성능 벤치 0건 · 핵심 모듈 비공개 · 상용 라이선스 조건 미검증**.
- **즉시 활용**: **조건부** — [[video-saas]] 스택의 생성 백본 후보이나, **Context-IR 없이 나온 768p 결과를 기준으로 평가**해야 한다. `H3-Base-Ref2VA`의 레퍼런스 상한(**이미지 ≤9 · 영상 ≤3클립 · 총 12파일**)은 캐릭터 일관성 설계에 직접 들어오는 제약.
- **6개월 영향력**: 중간~큼. **네이티브 동기 스테레오 오디오**는 볼트에서 드문 축이며, 영상+오디오를 따로 만들어 붙이는 현행 파이프라인을 단축할 수 있다.
- **대체 관계**: [[Seedance]]·[[LTX-2]] 등 영상 생성 백본과 경쟁. 편집 축([[InfinityEdit]])과는 보완.
- **허와 실**: **실** = 33B 옴니 아키텍처와 입출력 스펙이 카드에 상세히 공개돼 있다(이 정도 투명도는 드물다). **허** = *"오픈소스"* 표기와 달리 **품질 결정 모듈이 빠져 있고 벤치가 없다.**
- **액션**: 768p full-attention 4GPU 전제로 **비용 모델 산출** 후, 내 레퍼런스 캐릭터 3~5장으로 `Ref2VA` 일관성 스팟체크.

## 관련 페이지
- [[MiniMax-H3]] — 옴니모달 플래그십(부분 공개) · [[MiniMax-H3-Turbo-Lora]] — 파생
- [[MiniMax-M2.7]] · [[MiniMax-M3]] — 텍스트 계열 · [[MiniMax-Sparse-Attention]] — 아키텍처 연구
- [[Alibaba]] — **인코더 공급자**(Qwen3-VL-32B) · 예상 밖의 의존 관계
- [[Zhipu-AI]] · [[DeepSeek]] · [[Moonshot AI]] — 중국 오픈 모델 벤더 대비군
- [[PAWBench]] — 영상 생성물을 **분포**로 평가해야 한다는 요구(데모 근거의 무효화)
- [[AI-영상-생성-2026]] · [[video-saas]] · [[ai-news]]

## 원본
- 근거 소스: [[MiniMax-H3]] 모델카드 원문 대조 + HF API 실호출 (2026-08-29)
- 지표: DL **5,018,833**(30일) · 좋아요 **4,591** · 라이선스 other(minimax-h3-community-license-agreement) · pipeline `image-text-to-video`
- 신뢰도: ⭐⭐ medium (스펙은 실검증 · 성능 근거 없음 · 핵심 모듈 비공개)
