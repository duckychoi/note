---
title: k2-fsa
type: entity
domain: ai-news
tags: [ai-news, entity, org, tts, asr, speech, open-source, multilingual]
created: 2026-09-08
updated: 2026-09-08
sources: [OmniVoice.md]
reliability: high
---

# k2-fsa — 오픈소스 음성(ASR/TTS) 연구 조직

**GitHub 조직**: https://github.com/k2-fsa

> [!insight] 핵심
> 오픈소스 음성 인식/합성 생태계(k2, icefall, sherpa 계열)를 운영하는 조직. 볼트에는 [[OmniVoice]](**600+ 언어 음성복제 TTS**)로 등록돼 있다.
> **지표(2026-09-08 실측)**: HF 다운로드 **월 1,153,331** · 좋아요 1,355 · GitHub **⭐10,549** · **Apache-2.0** · **arXiv 논문 보유**(2604.00688).

> [!insight] 볼트가 이 조직에서 배운 것 — 표면이 여러 개다
> [[OmniVoice]] 는 **HF 모델 · GitHub 코드 · arXiv 논문 · HF Space 데모** 네 표면을 동시에 갖는다.
> 볼트는 2026-04-19에 **HF 표면 하나만 보고 페이지를 닫았고**, 나머지 셋을 5개월간 기록하지 않았다. 2026-09-08 이름 토큰 대조로 발견.
> → **오픈 음성 조직은 논문+코드+가중치+데모를 세트로 낸다**는 것이 이 조직에서 확인된 패턴이며, 인제스트 시 **네 표면을 함께 찾아야 한다.**

## 특징
- **Apache-2.0** — 코드·가중치 모두 제약 없음(볼트 실측). 상업 이용 가능.
- **경량 설계 전통** — 4월 볼트 기록 유지.
- **다국어 폭이 강점** — 600+ 언어 주장(자기신고, 언어별 품질 편차 미검증).

## 관련 페이지
- [[OmniVoice]] — 소스 페이지
- [[VoxCPM]] · [[VibeVoice]] · [[silero-vad]] — 음성 축 인접 소스
- [[kyutai-labs]] — **같은 성격의 오픈 음성 랩**(CPU 초경량 TTS)
- [[OpenBMB]] — 온디바이스 SLM 조직

## 원본
- 출처: https://github.com/k2-fsa/OmniVoice · https://huggingface.co/k2-fsa/OmniVoice
- 검증: GitHub API + HF 모델 API 실측 (2026-09-08)
- 신뢰도: ⭐⭐⭐⭐
