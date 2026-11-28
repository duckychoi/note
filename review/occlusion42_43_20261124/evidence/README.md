# #42·43 핵심 원행

[manifest.json](manifest.json)에 작업공간 안 원본 경로, 파일 크기, SHA-256을 기록했다. 이 폴더의 원행은 기존 출력의 바이트 사본이다. 일부 파일명 필드는 생성 당시 컴퓨터에서의 위치를 포함한다.

| 원행 | 확인할 수 있는 내용 | 재현 문서 |
|---|---|---|
| [전체 합성 학습 큐](full-students-queue.jsonl) · [개발 선택](full-development-choice.json) | 실행 완료·보류와 Hypersim 기준 선택 | [통제 관측과 학습](../CONTROLLED_OBSERVATIONS.md) |
| [A·B·C 첫 성능](pf-paths/results.json) | 같은 공개 S에서 출발한 세 학습 경로 | [세 경로 결과](../PF42_TRIAL_RESULT.md) |
| [A 등록 비교](a-complexity/results.json) · [짧은 BP 검산](a-complexity/fast-bp.json) | 36구역별 계산 시간, 우도·경사 오차, 실사 AbsRel | [실행 코드](../pf42_a_complexity.py) |
| [교차 시점 교사](crossview/results.json) · [가림 실패](crossview/visibility-diagnosis.json) · [새 교사](crossview/visibility-results.json) · [단안 인코더](crossview/visibility-encoder-results.json) | B의 가시성, 빈 공간, 여러 시점에서 RGB 한 장으로의 전이 | [가림과 전이](../PF42_VISIBILITY_RESULT.md) |
| [HAMMER 배율 진단](monocular-identifiability/diagnostic-hammer.json) · [iBims-1 배율 진단](monocular-identifiability/diagnostic-ibims.json) · [HAMMER 분리 학습](monocular-identifiability/matched-transfer-hammer.json) · [iBims-1 분리 학습](monocular-identifiability/matched-transfer-ibims.json) | B의 배율·형상 교환, 기존·별도 사진, 세 시드 비교 | [단안 식별과 교환](../PF42_MONOCULAR_RESULT.md) |
| [교사 배율 승인](scale-confidence/calibration.json) · [고정 형상 진단](scale-confidence/diagnostics.json) · [HAMMER](scale-confidence/hammer.json) · [iBims-1](scale-confidence/ibims.json) | 정답을 쓴 보정과 RGB 한 장을 받는 학생 평가 | [선택적 감독](../PF42_SCALE_CONFIDENCE_RESULT.md) |
| [C 실제 가중치](c-encoder/results.json) | 세 방향의 계수, 혼합 모형, 두 보류 장면 | [실행 코드](../pf42_c_encoder.py) |
| [DiLiGenT](studio43/teacher-diligent.json) · [HAMMER](studio43/scores-1024-hammer.json) · [DIODE](studio43/scores-1024-diode.json) · [iBims-1](studio43/scores-1024-ibims.json) · [재질·광원](studio43/material-light-middlebury.json) · [이동 경계](studio43/edge-motion.json) | #43 법선·형상 관측과 #42 단안 전이 | [광도 스테레오](../PHOTOMETRIC_STEREO_RESULT.md) · [재질과 광원](../MATERIAL_LIGHT_RESULT.md) · [이동 경계](../EDGE_MOTION_RESULT.md) |

RGB·LiDAR·렌더·가중치 전체는 작업공간의 `research-sources/`에 있다. 새 클론에서 이 원행으로 보고서 수치를 대조할 수 있다. 전처리와 학습을 다시 실행하려면 원본 데이터·라이선스·실행 환경을 각 규약에 맞게 준비한다. [전용 인계 문서](../HANDOFF.md)
