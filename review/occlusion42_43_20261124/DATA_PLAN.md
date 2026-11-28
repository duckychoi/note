# #42·43 후속 데이터 확보 계획

## 연구 질문

단안 입력(RGB 한 장과 알 수 있는 경우 카메라 내부 파라미터)에서 미터 단위 깊이를 추정한다. 학습 중에만 정답 깊이·카메라 자세·조명 정보를 이용하는 조건을 먼저 비교한다. 여러 RGB를 입력하는 추론은 별도 조건으로 둔다. 현재 단계에서 조명별 RGB 생성은 연구 목표에 포함하지 않는다.

합성 학습법의 강한 기준과 실사 단독·혼합 지도학습에 사용할 분할과 평가 조건은 [본시험 비교 기준](BENCHMARK_PROTOCOL.md)에 정리했다.

출력 깊이를 보정해 학생에 전달하는 방식, 학습 가능한 깊이 보정층을 관측으로 직접 업데이트하는 방식, 두 방식의 결합을 모두 조사한다. 개별 방식과 결합이 어떤 조건에서 유리한지 비교하고 변형 가능성을 열어 둔다. 같은 사전학습 모델, 장면, 관측 수, 추가 학습량의 비교군으로 정답 깊이 지도 학습, 기하 일관성, 조명 불변 학습을 둔다. 파라미터 업데이트의 최적성은 고정 특징 위 선형 보정층과 명시한 관측 오차 모델의 범위에서 먼저 유도한다.

## 데이터 출처와 확보 순서

| 순서 | 출처 | 역할 | 확보 범위 |
|---|---|---|---|
| 실사 확인 | [Middlebury Mobile 2021](https://vision.middlebury.edu/stereo/data/scenes2021/) | 이미 보관한 좌우 RGB·시차·보정값으로 정답과 정렬 확인 | 시험에 이미 쓴 장면은 처리 검산용. [공식 다운로드 목록](https://vision.middlebury.edu/stereo/data/scenes2021/zip/)의 `ambient-skates.zip`(84.1 MiB)부터 같은 카메라의 조명·노출을 구분한다. |
| 실사 본시험 | [DTU MVS](https://roboimagedata.compute.dtu.dk/?page_id=36) | 같은 장면에서 카메라와 조명을 교차한 실사. 장면당 49/64개 시점, 7가지 조명, 구조광 기준 기하 | 장면과 물체 계열 단위로 개발·시험을 나눈다. 공식 카메라로 스캔을 투영한다. 투영이 희소하거나 보이지 않는 점은 유효 정답에서 제외한다. DA3처럼 이미 DTU를 평가한 모델의 결과를 고려해 독립 실사 검증을 함께 둔다. |
| 실사 확장 | [OpenSubstance](https://opensubstance.github.io/) | 반사 재질·조명 변화·다중 시점과 정렬된 스캔이 있는 물체 | 계정 및 다운로드 요청이 필요하다. 접근이 가능하면 Part 10의 보정·정렬 메시와 해당 물체 이미지부터 받는다. |
| 공개 합성 | [Hypersim](https://github.com/apple-aiml-research/ml-hypersim), [TartanAir V2](https://tartanair.org/), [IRS](https://github.com/HKBU-HPML/IRS) | Hypersim은 사실적인 실내, TartanAir는 카메라 궤적, IRS는 실내 스테레오 시차·법선 | 전체를 받기 전에 각 출처에서 장면·궤적 일부의 RGB, 기준 깊이 또는 시차, 카메라 정보를 확보한다. Hypersim의 광선 거리를 카메라 z 깊이로 환산한다. 서로 다른 광원에서 같은 장면을 찍은 대응쌍은 확인된 경우에만 사용한다. |
| 통제 합성 | [BlenderProc](https://github.com/DLR-RM/BlenderProc)와 물리 기반 재질의 실내 자산 | 카메라와 조명을 독립 제어. 같은 장면의 RGB·z·카메라·가시성 기록 | 실사급 장면에서 카메라 2개×조명 2개부터 시작한다. 거리 단위·축·광선 정의를 검산한 뒤 장면과 조명 수를 늘린다. |

2026-11-24 확인 범위에서 Depth Anything V2의 교사 학습, Depth Anything 3의 학습, MoGe-3의 학습, PointDiT의 미세조정, Depth Pro 2의 학습에 Hypersim·TartanAir·IRS가 공통으로 쓰였다. 각 논문의 역할과 전체 학습 조합은 다르다. [Depth Anything V2](https://arxiv.org/html/2406.09414) · [Depth Anything 3](https://arxiv.org/html/2511.10647v1) · [MoGe-3](https://arxiv.org/html/2607.17967v2) · [PointDiT](https://github.com/google-research/pointdit/blob/main/DATASETS.md) · [Depth Pro 2](https://arxiv.org/html/2611.26603v1)

## 첫 검산

- 같은 카메라와 장면에서는 조명만 바꾼 깊이 정답이 일치해야 한다. 카메라가 바뀌면 같은 표면점의 깊이와 픽셀은 새 카메라 좌표로 변환한다.
- 시점 간 기하 잔차는 가려짐, 화면 밖, 움직이는 물체, 센서 정답 결측을 제외하고 계산한다. 조명 변화에서는 RGB 픽셀 차이를 깊이 오차로 사용하지 않는다.
- 거리 오차, 깊이 경계와 가림 순서, 조명만 바꿨을 때의 깊이 변동, 시점 재투영 오차를 따로 기록한다. 출처·장면·촬영 및 렌더 조건·보정행렬·유효 마스크·파일 해시를 보관한다.

이 기록은 자료 확보와 실험 설계 계획이다. Middlebury 기본 24개 촬영 묶음은 기존에 보관돼 있다. 추가 조명 파일, 공개 합성 데이터, 새 렌더 및 모델 비교 결과는 이 계획에 따라 확보할 항목이다.
