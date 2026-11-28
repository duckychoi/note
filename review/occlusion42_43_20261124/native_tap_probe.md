# Middlebury 실사에서 TAPNext++ 점추적 검산

Google 공식 TAPNext++ 256 가중치로 Middlebury 공개 보류 21장면과 13개 물체 계열을 채점했다. 한 사진으로 깊이를 예측하는 학생 시험과 별도로 실행했다. [공식 모델과 가중치](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapnextpp) · [Middlebury 2005](https://vision.middlebury.edu/stereo/data/scenes2005/) · [Middlebury 2006](https://vision.middlebury.edu/stereo/data/scenes2006/)

## 입력과 좌표

기준 영상은 조명 1의 정류 시점 1이다. 목표는 같은 조명 또는 조명 2의 시점 3과 5다. 각 조건은 기준과 목표 두 프레임으로 시작하고 추적 상태를 초기화했다. 쿼리는 영상 가장자리 16픽셀을 제외한 24픽셀 격자에서 뽑았다. 쿼리와 입력 영상에 깊이 정답은 제공하지 않았다. 공식 래퍼는 uint8 BGR와 원래 이미지 픽셀 순서 (x,y)를 받아 모델 안에서 (t,y,x)로 변환한다.

채점에 쓴 정류 시차 역투영은 목표 수평 위치 x−fd다. d는 1/3해상도 영상에서 측정한 시점 1→5의 크롭된 시차이고 f는 시점 3에서 0.5, 시점 5에서 1이다. 시점 5의 가시성은 양 끝 정답 시차를 대조했다. 시점 3은 등간격 이동과 선형 크롭을 가정하고 정답 시차 표면을 중간에 래스터화해 근사했다. 여기서 시점 3에는 별도 측정 깊이 지도가 없다. 물리 역깊이는 (d+dmin/3)/[(3740/3)×0.16]이다. 양의 시차 오차는 추적이 정답보다 더 큰 왼쪽 이동을 보고한 경우다. 학생의 배율 보정 후 깊이 보정 방향은 별도 계산이 필요하다.

## 실측

수치는 장면을 계열 안에서 평균한 뒤 13계열을 동일 가중했다. EPE는 예측이 보인다고 한 정답 유효 쿼리의 2차원 위치 오차다. PCK@3은 모든 정답 유효 쿼리를 분모로 사용해 모델의 비가시 예측도 실패로 반영했다.

| 목표 시점 | TAP EPE 동일→다른 조명, px | TAP PCK@3 동일→다른 조명 | SGBM 가시점 비율 동일→다른 조명 |
|---|---:|---:|---:|
| 3 | 0.657 → 0.931 | 97.38 → 95.05% | 53.54 → 33.91% |
| 5 | 1.405 → 1.823 | 92.70 → 87.54% | 57.32 → 32.95% |

| 두 모델 공통 유효점 | 공통 점 수 | TAP EPE, px | SGBM EPE, px |
|---|---:|---:|---:|
| 동일 조명, 시점 3 | 2356 | 0.423 | 0.304 |
| 동일 조명, 시점 5 | 2523 | 0.891 | 0.433 |
| 다른 조명, 시점 3 | 1538 | 0.681 | 5.509 |
| 다른 조명, 시점 5 | 1496 | 1.348 | 6.255 |

TAP 추적은 정답 유효점의 98.90~99.59%를 보였다고 판정했다. 이 점들에서 수평 이동 방향의 오류는 네 조건 모두 0이었다. 시차 크기와 배율 보정 방향의 정확성은 별도로 검증한다. SGBM은 고정 설정과 좌우 일관성 1픽셀 기준을 적용했다. 이 비교는 두 방법의 공통 유효점 조건과 전체 유효점 조건을 함께 보여준다.

모든 영상에 Exp1을 사용했지만 실제 조명별 노출 시간이 다른 장면이 7개였다. 노출 시간까지 같은 14장면·11계열에서도 조명 1→2에서 TAP EPE가 시점 3은 0.700→0.923px, 시점 5는 1.465→1.845px로 증가했다.

## 재현 파일

- [검산 코드](native_tap_probe.py)
- [장면 점수와 프레임 ID](../../../research-sources/encoder42_eval/native-surface-direction/tap-probe/result.json), SHA-256 8C557CB3E2EC1E2AEB00E25786EBC3D6D891FE12327DEFDC3408350A34F474CB
- [쿼리, 추적 결과와 검증용 정답](../../../research-sources/encoder42_eval/native-surface-direction/tap-probe/result.npz), SHA-256 C4144C751030923CAD9FD8E0527BE6AAD95A56041029CE248DFE8D5CD7B7AD77
- 공식 소스 커밋 ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b, 체크포인트 SHA-256 cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da

NPZ의 각 장면 query_xy와 조건별 gt_xy, tap_xy, sgbm_xy는 동일 순서다. 프레임 ID는 JSON의 frame_paths_relative_to_physical에 있다. GT 유효 마스크, 1→5 시차, 역깊이 및 추적기의 가시성 출력을 각각 저장했다. 재실행은 torch 환경에 기존 ml-depth-pro-2-deps를 PYTHONPATH로 추가한 뒤 python -B native_tap_probe.py 명령으로 한다.
