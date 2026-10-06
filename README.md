# NeuralBeamForming

AI-native 빔포밍(beamforming) 재발명을 위한 연구 노트·아이디어·수식 유도·시뮬 설계 모음.

> "바이브 코딩(AI-native) 이후 관점에서, 빔포밍을 *딥러닝으로 하는 것*을 넘어 *방법론 자체를 재발명*할 수 있는가?"
> 이 레포는 그 질문에서 출발한 리서치 로그와 두 개의 플래그십 연구 가설을 정리한다.

## 무엇이 들어있나

- **`docs/00_research_log.md`** — 대화에서 조사·제시한 내용 전체 타임라인 정리.
- **`docs/01_ai_native_beamforming_landscape.md`** — 빔포밍 재발명의 6단계 사다리(①솔버 교체 → ⑥AI 알고리즘 발견), neural operator·task-oriented·ISAC·foundation model 지형.
- **`docs/02_vf_monbt.md`** — **플래그십 A**: `VF-MoNBT` = Neural Beamforming Transformer(NBT) + CAT-MoEformer(MoE 라우팅) + 멀티빔 동시조준(set-prediction) + Volumetric Beam Focusing(근거리장 3D 점집속) 융합 아키텍처.
- **`docs/03_spatial_supertwisting_aperture.md`** — **플래그십 B**: SMC/슈퍼트위스팅의 *이론 구조 자체*를 빔합성 수식에 내재 — "공간 슈퍼트위스팅 개구". 양자화 로브 유도 + 노이즈 셰이핑 분석 + 시뮬 설계.
- **`docs/04_physical_twist_slide_beams.md`** — 물리적으로 꼬인/미끄러지는 빔(OAM 보텍스·Airy 자동집속·FDA), sliding-mode-as-optimization, "twist×twist" 결혼 논의.
- **`sim/spatial_supertwisting.py`** — 공간 슈퍼트위스팅 개구 시뮬레이션(순진 1-bit vs 1차 ΔΣ vs 선형 2차 ΔΣ vs 슈퍼트위스팅) 뼈대 코드.
- **`references.md`** — 수집한 arXiv·저널 레퍼런스 전체.

## 두 플래그십 요약

**A. VF-MoNBT (아키텍처 융합)**
- NBT = 엔진(How), MoE = 전문가=초점 선택(Which), 포커스 쿼리 = 멀티빔 set-prediction(What), Volumetric = 근거리장 3D 점집속(Where). 네 축이 직교 → 스택 가능.
- 신규성: "전문가=초점 / 쿼리=타겟" 구성성으로 가변 다중표적 해결. (DETR/TAPNet식 쿼리 토큰)

**B. 공간 슈퍼트위스팅 개구 (이론 융합)**
- `1-bit 개구 = 공간 sign()`, `ΔΣ = 1차 슬라이딩`, `슈퍼트위스팅 = 2차 슬라이딩`.
- 슈퍼트위스팅 = 리야프노프-안정·유한정착 **비선형 2차 ΔΣ** → 양자화 로브를 메인빔 근방에서 40 dB/decade 셰이핑.
- 응용이 아니라 **빔합성 수식에 SMC 이론이 내재**.

## 포지셔닝
통신학회 경쟁(NBT·CAT-MoEformer 등 2026 최신)보다, **로봇-센싱·온디바이스 지각** 도메인에 꽂으면 블루오션. (로컬 LLM 휴머노이드 SLAM/캠핑 트랙과 연결)

## 상태
초기 연구 가설 단계(pre-implementation). 수식 유도·시뮬 설계는 작성됨, 실험·검증은 TODO.

---
생성: 대화 기반 리서치 정리. 세부 출처는 `references.md` 참조.
