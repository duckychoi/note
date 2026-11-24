# CPU 스모크 테스트 결과 (2026-11-24, torch 2.14.1+cpu, timm 1.0.31)

성능 판정이 아니라 "끝까지 실행 + 손실 감소" 확인. 출력 원문.

## depth/smoke_test.py (합성 장면, 축소 ViT d96·4층, 56px)
```
[teacher] loss 처음10 2.361 → 마지막10 1.596
[pseudo] 16장 라벨
[student] loss 1.368 → 0.177 (스텝 8)
[metric] loss 1.129 → 0.473 (스텝 6)
[sharpen] loss 7.443 → 4.032 (스텝 6)
[eval] relative {'absrel': 0.1367, 'delta1': 0.9270, 'boundary_f1': 0.0} | metric {'absrel': 0.3652, 'delta1': 0.2161}
SMOKE OK
```
- boundary_f1=0.0 은 지표 버그가 아님을 단위검증: 정답 대 정답 1.0 / 5픽셀 블러 0.119 / 평평 0.0. 원 정의가 같은 픽셀 일치만 인정해 가혹 → `tol=2` 변형 추가(블러 0.55), 두 값 모두 보고.
- tol 옵션 추가 후 재실행도 통과: relative absrel 0.120 / δ1 0.930 / boundary_f1 0.0 / boundary_f1_tol2 0.0, metric absrel 0.301 / δ1 0.405 (60스텝 축소 모델이라 경계를 아직 못 잡음).

## gen/smoke_test.py (DiT d64·2층, 무작위 클래스 패턴 잠재)
```
[stage0 image] loss 처음10 1.918 → 마지막10 0.802
[params] 공유 adaLN 705360 vs 블록별 827856 (15% 절감)
SMOKE OK — 2단계 커리큘럼(이미지/영상 혼합)·EMA·CFG 샘플러 동작
```

## llm/smoke_test.py (문자 토크나이저, GPT d64·3층)
```
[pretrain dense] val loss 3.726 → 1.113
[pretrain MoE(aux-free)] val loss 3.687 → 1.119, 전문가 부하 max/mean 1.32
[SFT] 정확도 0.11 → 0.56
[GRPO] 30스텝, 유지 그룹 206, 평균 보상 0.28 → 0.39, 정확도 0.56 → 0.54
[distill] KL 3.693 → 2.610
SMOKE OK
```
- 첫 실행(GRPO lr 3e-4, 15스텝)에서 정확도 0.56 → 0.42 하락 → 부호 버그 의심으로 **양성 대조** `llm/diag_grpo_sign.py` 실행(같은 SFT 시작점, lr 5e-5, 40스텝, 시드 3):
```
GRPO: 탐욕 정확도 변화 [0.16 0.32 0.3 ] 평균 +0.260 | 샘플 보상 변화 평균 +0.045
대조(어드밴티지 섞음): 탐욕 정확도 변화 [-0.2 -0.08 -0.29] 평균 -0.190 | 샘플 보상 변화 평균 -0.079
```
  → 부호·마스크 정상, 하락 원인은 과대 학습률. 스모크 설정을 lr 5e-5로 교체 후 위 결과.
- 설정 파일 실측 파라미터: 티어 B dense 114M / MoE 208M(활성 111M), 티어 A dense 313M / MoE 693M(활성 313M). 계획서 수치를 실측으로 정정.

## vlm/smoke_test.py (무작위 ViT d48 동결 + GPT d64, 색 사각형 → 색 이름)
```
[stage0] 학습 LLM 파라미터 텐서 0개, loss 3.373 → 3.105, 색 정답률 0.00 (우연 0.25)
[stage1] 학습 LLM 파라미터 텐서 22개, loss 2.503 → 0.192, 색 정답률 1.00 (우연 0.25)
SMOKE OK
```
- stage0 정답률 0 은 LLM이 사전학습 안 된 무작위라 단어를 못 만드는 것(정상). 실제 실행은 사전학습 LLM 사용.
- 버그 수정 기록: 임베딩이 VLM·LLM 양쪽에 등록돼 옵티마이저 파라미터 중복 → `set_stage`에서 id 기준 중복 제거.
