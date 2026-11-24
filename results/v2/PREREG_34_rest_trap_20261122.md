# #34 사전 등록: 정지 함정 사전 예측 (2026-11-22, 새 RL 실행 전에 작성)

진단: note/src/v2/il/diag_rollout_predict.py → results/v2/diag_rollout_predict.json
규칙: D_medium(중간 품질 보행 에피소드의 할인 누적 보상이 '가장 높게 채점된 현실적 정지'의 중앙값을 이기는 비율) > 0.5 이면 "탈출",
아니면 "정지". 판정: 고정 prior 단독 보상(*_plain), 30만 스텝, 3 seed, 최종 평가 전진 속도 평균 ≥ 1.0 m/s 이면 탈출.
보정점(시험 아님): HalfCheetah none_plain(정지, 실제 vx 0.00), film4_plain(탈출, 실제 vx 2.01). v1 규칙(D_simple)은 film4를 틀려 폐기.
비교 예측자: 옛 진단(창 AUC simple>0 토크 정지 > 0.5)

| 환경 | 보상 | D_medium | 새 진단 예측 | 옛 진단 예측(AUC) |
|---|---|---|---|---|
| HalfCheetah | film4shuf | 0.05 | 정지 | 정지 (0.05) |
| Hopper | none | 0.00 | 정지 | 탈출 (0.74) |
| Hopper | film4 | 1.00 | 탈출 | 탈출 (0.90) |
| Hopper | film4shuf | 0.00 | 정지 | 탈출 (0.74) |
| Walker2d | none | 1.00 | 탈출 | 탈출 (0.93) |
| Walker2d | film4 | 1.00 | 탈출 | 탈출 (0.91) |
| Walker2d | film4shuf | 1.00 | 탈출 | 탈출 (0.90) |

위상의 몫: 같은 FiLM 구조에서 정상 위상표(film4) vs 뒤섞은 위상표(film4shuf).
한계: Walker2d 정지 기준은 PD로 못 세워서 다른 보상(_like_state)으로 학습된 파일럿 정책 2개를 사용.

## 결과 (2026-11-23 채점, 위 예측은 수정하지 않음)

로그: E:/0.Project/fm34_env/prereg/logs/*.log, seed별 JSON 같은 폴더. Hopper·Walker는 11-22 재부팅으로 260k에서 끊겨 11-23 처음부터 재실행(끊긴 기록: interrupted_20261122/).

| 환경 | 보상 | 최종 vx 평균 (seed 0/1/2) | seed 최고 vx(학습 중) | 실제 | 새 진단 | 옛 진단 |
|---|---|---|---|---|---|---|
| HalfCheetah | film4shuf | 0.00 (0.00/0.00/0.01) | 0.04 | 정지 | O | O |
| Hopper | none | 0.00 (0.03/−0.02/0.00) | 0.80 | 정지 | O | X |
| Hopper | film4 | 0.12 (0.31/0.01/0.06) | 0.73 | 정지 | X | X |
| Hopper | film4shuf | 0.07 (0.01/0.26/0.13) | 0.82 | 정지 | O | X |
| Walker2d | none | −0.02 (−0.14/0.07/0.01) | 0.54 | 정지 | X | X |
| Walker2d | film4 | 0.15 (0.13/0.26/0.07) | 0.71 | 정지 | X | X |
| Walker2d | film4shuf | 0.02 (−0.22/0.14/0.11) | 0.53 | 정지 | X | X |

- 적중: 새 진단 3/7, 옛 진단 1/7. **그러나 "항상 정지" 예측이 7/7** — 7개 모두 정지라 두 진단 모두 자명한 기준보다 못함.
- 새 진단이 "탈출"이라 한 4건은 전부 틀림(탈출 적중 0/4). 새 진단이 맞힌 3건은 모두 "정지" 예측.
- 판정: **사전등록 가설(D_medium으로 정지 함정을 학습 전에 예측) 불지지.** 탈출 사례가 HalfCheetah film4 하나뿐(보정점)이라 판별력 검정 자체가 성립하지 않음.
- 위상의 몫(film4 vs film4shuf): Hopper 0.12 vs 0.07, Walker 0.15 vs 0.02 — 둘 다 1.0에 한참 못 미쳐 의미 있는 차이로 보기 어려움.
