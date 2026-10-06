# 02 — 플래그십 A: VF-MoNBT

**Volumetric-Focusing Mixture-of-experts Neural Beamforming Transformer**

NBT + CAT-MoEformer + 멀티빔 동시조준 + Volumetric Beam Focusing 융합.

## 왜 융합이 되나 — 네 조각이 서로 다른 축

- **NBT = 엔진(How)**: 날 안테나 전압 → 가중치, 복소 트랜스포머 + 매니폴드 투영(constant-modulus), 보편근사.
- **CAT-MoEformer = 선택기(Which)**: 장면조건부 MoE 게이팅. 융합에선 "어떤 빔 전문가를 몇 개 켤까" → **가변 개수 타겟** 자연 처리.
- **멀티빔 동시조준 = 출력 구조(What)**: 한 번에 K개 빔 + 빔 간 간섭 널링.
- **Volumetric Focusing = 출력 기하(Where)**: 원거리 "방향(θ)"이 아니라 **근거리장 3D 점 (x,y,z) 집속** → 같은 각도·다른 거리의 두 타겟 분리(진짜 "점 조준").

→ **How × Which × What × Where** = 축이 직교 → 스택 가능.

## 아키텍처 데이터 흐름

1. **입력**: 날 안테나 스냅샷(NBT 섭취) + **컨텍스트 토큰**(추정 타겟 수·대략 위치·속도, 근/원거리 레짐) → MoE 게이트.
2. **복소 트랜스포머 인코더(NBT)** → 잠재표현.
3. **포커스 쿼리 토큰 (핵심 신규성)**: 타겟 1개 = 쿼리 토큰 1개(DETR object query / TAPNet 질의 점과 동형). K개 쿼리 → K개 빔. **가변 K를 set-prediction**(헝가리안 매칭)으로 처리.
4. **MoE-FFN 층**: **전문가 = 집속 프리미티브**(공간 서브볼륨별 / LOS·NLOS별 / 근·원거리별). 게이트가 쿼리별로 전문가 라우팅.
5. **출력헤드**: 쿼리별 복소 가중치 → 매니폴드 투영 → 배열 여기(excitation)로 중첩.
6. **포워드 모델**: 근거리장 **구면파 스티어링 a(x,y,z)**(원거리 a(θ) 아님)로 손실 계산.

## 손실 (다목적, physics-informed)

- 타겟별 **근거리장 집속 이득** 최대화
- **빔 간 누화(멀티포커스 크로스토크) 널링** 최소화 (NCBF 아이디어)
- constant-modulus·전력 제약
- ISAC 확장 시 **위치추정 CRB**까지 합산

개념적으로:
```
L = -Σ_k Gain(w, p_k)              # 각 타겟 3D 점 p_k 집속이득
    + λ1 Σ_{k≠l} Leak(w, p_k, p_l) # 초점 간 누화
    + λ2 ConstMod(w)               # 모듈러스 제약
    + λ3 CRB_loc(w; {p_k})         # (ISAC) 위치추정 하한
```

## 이 융합이 만드는 "풀어야 할 문제" (= 논문거리)

1. **가변 타겟 수 K** → set-prediction + MoE **부하균형**(CAT-MoEformer가 경고한 expert collapse 해결). ← 1등 신규성.
2. **근거리장 = 각도+거리(위치)** → 집속공간 조합폭발 → 쿼리토큰·far-to-near GNN으로 파일럿/학습부담 축소.
3. **볼륨 내 멀티포커스 간섭** → 3D 근접 초점 누수 → 능동 널링 목표.
4. **K·기하 일반화** → MoE 전문가 + 쿼리토큰의 **구성성(compositionality)**이 핵심 셀링포인트.

## 포지셔닝 & 신규성

- 한 줄: *"근거리장 다중표적 ISAC을 위한 집합예측 신경 빔포머 — 전문가=초점, 쿼리=타겟."*
- 아키텍처 신규성: ① MoE 전문가를 **초점(focus)으로** 해석 + ② DETR/TAPNet식 **포커스 쿼리로 멀티빔 set-prediction**. 이 조합은 미발표.
- 근접 선행: NCBF-DNN 근거리장 널링, 학습 홀로그래픽 MU(GNN), CAT-MoEformer, NBT, Volumetric Beam Focusing — **넷 동시 융합은 0건**.
- 블루오션: 로봇/온디바이스 — "로봇이 센서 어레이로 다중표적을 3D 점집속·동시추적"(TAPNet이 점을 추적하듯, 점에 빔을 꽂는다).

## 정직한 리스크

- 근거리장은 XL-MIMO·mmWave/THz 초대형 개구에서만 성립 → 적용 레짐 명시.
- 근거리장 채널 실측 희소 → 레이트레이싱 시뮬로 학습셋 구축(공수 큼).
- MoE = 파라미터↑ vs 온디바이스 긴장 → 경량 게이팅·소수 전문가(top-1/2).
- "볼륨 집속 이득"은 근거리장 한정 — far-field 일반 우위로 포장 ❌.
- 베이스라인(NCBF-DNN·홀로그래픽 GNN)을 가변-K 일반화·추론지연·다중표적 분해능으로 명확히 이겨야.

## TODO
- [ ] 블록다이어그램 그림(HTML/SVG)
- [ ] 근거리장 포워드 모델 미분가능 구현
- [ ] set-prediction(헝가리안) 멀티빔 헤드 프로토타입
- [ ] MoE 부하균형(load-balancing loss) 실험
- [ ] 베이스라인 대비표
