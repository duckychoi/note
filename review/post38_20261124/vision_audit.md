# 디지털 스튜디오 비전 #43–58 독립 반박검수

심사 판정은 **원장 문구 그대로는 보류**다. #43의 작은 수치 실험은 재현됐다. #58은 실제로 겹치는 관측이 부족하면 재투영만으로 빈 정보를 채울 수 없다. #44와 #53은 적은 비용으로 별도 실험을 열 가치가 있다.

## 기존 근거와 새로 남긴 근거

- 원장 `lab`의 `origin/main@3aaa471`, `wiki/synthesis/research-topics-ledger.md` 180–184, 198–214행. #43 코드는 `note`의 `origin/selfcorrecting-studio-engine@956d8cf`, `src/toy_selfcorrecting_engine.py`.
- 이번 검수에서는 원본 실행 로그 `vision_original_toy.log`와 이 문서를 남겼다. 본 검수는 독립 1라운드다. 원고가 없어 원고 문체 판정은 해당 사항이 없다. 선행 논문은 확인한 범위에서 가장 가까운 직접 비교 후보를 골랐다. 전체 문헌에 대한 최초성 판정은 미확인이다.
- 아래의 통과 기준은 판정용 제안이다. #43 원본 이외의 학습·공개 벤치 실험은 실행하지 않았다. 각 후보의 원 논문 설정과 공정하게 재튜닝한 판을 함께 둔다. 동일한 사전학습, 정보 출처, 장면 분할, 렌더·학습량을 맞춘다. 실측 비용에는 촬영, 라벨, 기준기 교정까지 포함한다. 독립 장면 또는 실험 단위로 쌍별 신뢰구간을 보고하고, 검사 장면으로 튜닝하지 않는다.

## #43 자기교정 데이터 엔진

| 원본 장치 | 원본 재현 수치 | 뜻 |
|---|---:|---|
| 합성만 → 일관성만 | 7.800 → 3.012 | 모델이 정해 둔 방향의 오차를 줄임 |
| 실측 12회 → 일관성+실측 12회 | 2.451 → 0.990 | 마지막 세대 평균 오차가 약 60% 줄어듦 |
| 실측 12회 → 일관성+실측 3회 | 2.451 → 1.879 | 이 설정에서 실측을 4분의 1로 줄여도 평균 오차가 약 23% 작음 |
| 원본 기준 오차 1 초과율 | 100% → 46% | 실패가 줄었고 46%가 남음 |

최대 오차는 1.214다. 제안법의 평균은 세대 10에서 0.29, 119에서 0.99로 증가했다. 코드의 “낮게 수렴”, “붕괴 없음”은 이 실행이 뒷받침하지 않는다. `Q`는 필터 내부 공분산만 키운다. 시뮬레이션 상태에는 일정한 편향만 들어간다. 관측은 참 상태가 0이라고 아는 모형에서 생성되고, 관측 방향도 정확히 제공된다. 실측은 `H=I₂`여서 두 축을 모두 본다. 방법별 무작위 수열 소비량이 달라서 같은 시드 번호여도 실측 잡음 배열은 공유되지 않는다. 마지막 값은 주기 내 같은 위상을 뜻하는지 확인해야 한다. 다른 간격에서 마지막 실측 후 4·9·19·39·119세대가 지나므로 마지막 값뿐 아니라 주기 평균·최악값을 보고한다.

**수축 조건.** 관측 직후의 오차 차이는 선형 모형에서 `(I−KH)δe`다. `Hv=0`이면 `(I−KH)v=v`다. 따라서 잡히지 않는 축이 있는 한 매 단계 전 공간 균일 수축은 성립하지 않는다. 추정 오차 자체에는 편향·잡음이 계속 들어간다. 상한이 있는 간격, 일관성/앵커의 합성 감도에 양의 하한, 잡음·편향의 제약을 두고 여러 단계의 곱으로 수축을 확인하는 것이 다음 정리의 후보이다. 비선형 모델에서는 선형화 동네와 데이터 연관 오류에 대한 조건이 추가된다.

**가까운 선행과 비교.** Triggs 등의 BA는 투영에 드러나지 않는 방향과 게이지 고정을 이미 다뤘다. Carrillo 등의 active SLAM은 D-opt 등 정보에 기반한 관측 선택을 비교했다. 2512.13411은 Gaussian splat 합성과 실데이터를 합치는 데이터 생성의 직접 선행이다. 남는 질문은 재학습으로 함께 흔들리는 센서와 예측기에서, 약하게 보이는 방향을 겨냥한 실측 선택이 일반 정보이득보다 싼가다. [BA §9](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Triggs00.pdf) · [Carrillo et al.](https://www.robots.ox.ac.uk/ActiveVision/Publications/carillo_etal_icra2012/carillo_etal_icra2012.pdf) · [Gaussian splat 데이터 생성](https://arxiv.org/abs/2512.13411)

**관문.** 같은 이미지·렌더와 실측 후보로 robust BA/EKF, 고정 실혼합, 무작위·주기 앵커, D-opt 및 관심축 A-opt를 비교한다. 고정 게이지 기준과 물리 단위를 먼저 정한다. 위치·깊이·포즈를 동시에 조정하고, 앵커 방식마다 비용과 관측할 수 있는 물리량을 맞춘다. 공유 픽셀·공유 인코더에서 생긴 관측 상관을 백색화 또는 공동 공분산으로 반영한다. 같은 비용에서 독립 장면의 절대 스케일·카메라 오차가 개선되거나, 같은 허용 오차를 더 적은 비용으로 만족할 때 진행한다. 일관성 잔차만 줄거나 일반 정보이득 기준과 구별되지 않으면 엔진 논문 신규성은 중단한다.

## #44–57 개별 점검

### #44 조명·시점에 강한 metric depth

Depth Pro 2는 최신 단안 metric 깊이와 시간 일관성을 비교할 후보다. Depth Anything 3는 여러 뷰에서 함께 맞는 기하를 직접 겨룰 수 있다. 같은 카메라·기하에서 조명을 바꾸면 정답 깊이는 같다. 카메라가 이동하면 픽셀과 카메라 좌표 깊이가 함께 바뀐다. 투명 컵에서는 전면 표면과 배경 중 어느 깊이를 라벨로 잡았는가도 중요하다. 렌더 수가 같은 독립 조명 랜덤화, 짝지은 조명 일관성, 같은 실측으로 직접 fine-tune을 겨룬다. 조도 3수준으로 실물 정적 장면을 촬영하고 다른 장면·카메라의 절대 AbsRel과 경계 F1을 측정한다. 정답으로 스케일이나 오프셋을 맞춘 점수는 따로 둔다. 고정 예산에서 절대 오차와 경계 중 하나가 좋아지고 다른 하나가 유지되면 진행한다. 정렬 후 점수에서만 이기면 metric 주장을 중단한다. [Depth Pro 2](https://arxiv.org/abs/2611.26603) · [Depth Anything 3](https://arxiv.org/abs/2511.10647)

### #45 법선·알베도·셰이딩

LumiMotion은 움직임으로 같은 표면을 다른 조명 아래 드러내어 재질과 조명을 분리한다. 입력 재구성 손실만으로는 반사율 0.25×조명 2와 0.5×1을 구별하기 어렵다. 작은 실험은 알려진 원판·구·유광 물체에 조명 방향을 바꾸고, 고정 광량 기준으로 보정한 뒤 미사용 조명에서 재조명한다. 법선 각도와 선형 RGB 알베도 오차를 각각 집계한다. 같은 뷰와 캘리브레이션을 쓰는 photometric stereo 및 LumiMotion을 비교한다. 정답 법선·알베도와 미사용 재조명까지 좋아질 때 진행한다. 재투영만 줄며 재질 오차가 커지면 중단한다. [LumiMotion](https://joaxkal.github.io/LumiMotion/)

### #46 합성 뷰로 카메라 재위치추정

2412.00138은 이미 splat 렌더와 외관 변화를 활용한다. CVPR 2026의 Hierarchical Visual Relocalization도 가까운 뷰 합성으로 실영상 카메라 위치를 구한다. 반복 복도의 한 칸이 어긋나도 리프로젝션이 작을 수 있다. 동일 지도, 실제 등록 프레임, 카메라 내부 파라미터, 렌더 수, 추론 시간 한도로 7-Scenes의 held-out 경로를 비교한다. PnP+RANSAC, 최신 계층 방식, 고정 실측 보정과 경쟁하고 cm·deg 및 5cm/5° 성공률·실패 꼬리를 기록한다. 두 장면 이상에서 성공률과 심한 오위치가 개선될 때 진행한다. 지도 구성 때 쓴 실데이터를 숨기면 예산 주장을 중단한다. [Adversarial Exploitation](https://arxiv.org/abs/2412.00138) · [Hierarchical Visual Relocalization](https://openaccess.thecvf.com/content/CVPR2026/papers/Tao_Hierarchical_Visual_Relocalization_with_Nearest_View_Synthesis_from_Feature_Gaussian_CVPR_2026_paper.pdf)

### #47 광류와 장면 흐름

FlyingDrones는 합성·실영상 전이를 비교하며 SEA-RAFT 등 광류 기준을 제공한다. KITTI는 scene flow의 공개 검증 후보다. 2D 동일 이동이 카메라 움직임이나 물체의 3D 이동에서 모두 생길 수 있다. 가림은 왕복 워프 손실의 정의역에서 빼고 실패율을 따로 센다. 같은 입력 프레임·카메라·깊이 사용량으로 synthetic-only/단순 real-adapt/워프 일관성을 비교한다. FlyingDrones는 광류 EPE, KITTI scene flow는 맞는 입력 계층의 D1/D2/Fl/SF 및 가림 조건으로 본다. 미사용 카메라에서 실제 흐름 지표가 좋아지면 진행한다. 워프 개선만 있거나 GT를 이용한 가시성에서만 이기면 중단한다. [FlyingDrones](https://arxiv.org/abs/2611.07705) · [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

### #48 동적 3DGS·4DGS 시간 추적

MotionScale은 영상 외관·기하·모션을 함께 복원한다. TAPVid-MV는 동기화된 여러 뷰의 실제 3D 트랙을 검증할 후보다. 가우시안 두 개가 ID를 맞바꿔도 렌더 픽셀은 같을 수 있다. 동기화 뷰 수와 사전학습 추적기·카메라 정보를 동일하게 주고 MotionScale 또는 같은 관측을 쓰는 multi-view 기준과 비교한다. 가림 뒤 실제 대응이 복구되는지 3D AJ·위치오차·가림 판정과 새 뷰 품질을 함께 본다. 실제 3D 트랙이 좋아질 때 진행한다. 렌더 개선만으로 추적 개선을 주장하면 중단한다. [MotionScale](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_MotionScale_Reconstructing_Appearance_Geometry_and_Motion_of_Dynamic_Scenes_with_CVPR_2026_paper.pdf) · [TAPVid-MV](https://arxiv.org/abs/2609.01899)

### #49 새 시점 영상과 물리 일관성

WorldStereo와 HY-World 2.0은 카메라가 이동해도 맞는 외관과 구조를 생성하는 직접 선행이다. 시각적으로 자연스러운 공이 탁자를 통과할 수 있다. 같은 시작 이미지와 경로, 생성 프레임·모델 규모에서 기하 메모리와 데이터 개입을 비교한다. 알려진 경로를 왕복하고 간단한 낙하·접촉 장면을 만든다. 카메라 오차, held-out 뷰, 접촉 위반·운동 보존 지표를 나란히 센다. 외관과 개입 후 실제 기하·물리 지표가 모두 납득할 수준이면 진행한다. PSNR·영상 점수만 올라가면 물리 주장은 중단한다. [WorldStereo](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_WorldStereo_Bridging_Camera-Guided_Video_Generation_and_Scene_Reconstruction_via_3D_CVPR_2026_paper.pdf) · [HY-World 2.0](https://arxiv.org/abs/2604.14268)

### #50 HDR 환경광

LiMo는 한 지점의 조명뿐 아니라 영상의 위치별·시간별 HDR 조명도 추정한다. 단안으로 비교할 때는 논문의 이미지 모델에 같은 노출 정보와 깊이 입력을 준다. 카메라 뒤 조명을 다르게 놓아도 관측 표면이 무광이면 단일 이미지가 차이를 거의 드러내지 않을 수 있다. 실제 브래킷 노출과 거울구를 정답용으로만 사용해 고정 위치에서 방향 오차·로그 휘도·가상 구의 재조명 오차를 잰다. 다른 방에도 재조명이 좋아지고, 다봉/저감도에서 잘 모른다는 판단도 교정될 때 진행한다. 정확한 단일 HDR을 보장하는 문구는 중단한다. [Lighting in Motion](https://openaccess.thecvf.com/content/CVPR2026/papers/Bolduc_Lighting_in_Motion_Spatiotemporal_HDR_Lighting_Estimation_CVPR_2026_paper.pdf)

### #51 재조명·그림자·반사

LumiMotion과 직접 겨루되 실제 스튜디오에서는 단일 영상 편집과 멀티뷰 재조명을 분리한다. 검정 얼룩과 접촉 그림자가 같은 픽셀 밝기를 만들 수 있고, 가려진 질감은 원본에 없다. 같은 입력 마스크·HDR/조명 조건·정밀 포즈를 제공해 조명을 끈 쌍과 바꾼 쌍을 각각 촬영한다. 전체 LPIPS뿐 아니라 그림자 경계, 반복 영상 깜빡임, 유광 방향 오차를 측정한다. 실제 미사용 소재의 그림자 경계와 시간 안정성이 개선되면 진행한다. 디테일을 지워서만 일관성이 좋아지면 중단한다. [LumiMotion](https://joaxkal.github.io/LumiMotion/)

### #52 영상 열화 복원

DCTTA는 열화가 일치하도록 다시 변형해 테스트 중 복원기를 조정하는 직접 경쟁자다. 다운샘플링된 회색 1픽셀에서 서로 반대인 흑백 체크무늬를 구별할 수 없다. 우선 저조도나 블러 하나를 고르고 LOL-v2-real 또는 RealBlur-J에서 같은 백본, 실측 장면 수, 추론 시간으로 단순 증강·DCTTA·제안법을 비교한다. 실제 카메라, ISO 또는 셔터를 분리하고 입력 복원 PSNR·LPIPS와 데이터 재현성을 관찰한다. 두 실제 카메라에서 개선되면 그 열화만 진행한다. 한 열화에서 얻은 결과로 안개와 초해상도까지 일반화하면 중단한다. [DCTTA 논문](https://openaccess.thecvf.com/content/CVPR2026/papers/Tang_Degradation-Consistent_Test-Time_Adaptation_for_All-in-One_Image_Restoration_CVPR_2026_paper.pdf) · [실제 열화 비교표](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tang_Degradation-Consistent_Test-Time_Adaptation_CVPR_2026_supplemental.pdf)

### #53 마스크·amodal·알파 매팅

MatAnyone 2는 GT 없이 매팅 품질을 학습해 실데이터에 적용하는 직접 선행이다. MatAnyone-E는 알파 전파의 효율 비교 후보다. 이진 마스크와 알파는 서로 다르다. 픽셀값 0.5는 회색 전경이 완전히 덮거나, 흰 전경이 검은 배경을 반쯤 덮어도 나온다. 첫 프레임 마스크나 무프롬프트 등 정보 수준을 고정하고, 머리카락·반투명 천·빠른 움직임을 포함한 독립 샷의 알파 Grad·Conn·시간 dtSSD를 본다. 얇은 경계와 시간 오차가 함께 좋아지면 진행한다. 실제 alpha GT가 없으면 배경 교체 영상의 블라인드 전문가 평가를 보조로 쓰고, amodal 회복과 매팅을 한 수치로 합치지 않는다. 시간 평활만 좋아지고 가는 디테일이 지워지면 중단한다. [MatAnyone 2](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_MatAnyone_2_Scaling_Video_Matting_via_a_Learned_Quality_Evaluator_CVPR_2026_paper.pdf) · [MatAnyone-E](https://arxiv.org/abs/2610.35724)

### #54 몸·손·얼굴 모캡

BEDLAM은 합성 SMPL-X를 이미 제공한다. MAMMA는 멀티뷰 모캡과 합성 확장을 직접 다루고 CADENCE는 영상에서 손 동작을 유지하는 시간 보정 기준이다. 헐렁한 옷 아래 체형이나 등 뒤 손의 관절은 두 눈에 모두 가려질 수 있다. 카메라 수와 초기 모델, 실제 라벨 수를 맞춰 3DPW와 MAMMA의 조건에 맞는 시퀀스에서 몸/손을 분리 평가한다. 영상 흔들림과 함께 세계 좌표 오차·속도·발 미끄러짐을 본다. 얼굴을 재려면 별도 얼굴 정답을 쓴다. 새로운 사람의 손/몸 지표가 좋아지고 빠른 움직임을 지우지 않을 때 진행한다. PA 정렬 후에만 좋아지면 실거리 모캡 주장을 중단한다. [BEDLAM](https://bedlam.is.tuebingen.mpg.de/) · [MAMMA](https://openaccess.thecvf.com/content/CVPR2026/papers/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.pdf) · [CADENCE](https://arxiv.org/abs/2611.26783)

### #55 6DoF 물체 포즈

ConceptPose는 모델 없는 zero-shot 포즈의 가까운 기준이며, BOP는 CAD·깊이 허용 수준에 맞춰 비교할 수 있다. 민무늬 원통은 수직축으로 돌아도 사진이 같다. 실제 T-LESS/YCB-V에서 CAD·깊이·입력 검출 마스크를 통제해 대칭을 반영하는 MSSD·MSPD·VSD와 실패율을 본다. 실측 없이 그대로 적용하는 공개 방식과, 같은 합성량과 실측 예산으로 재튜닝한 방식을 둘 다 둔다. 미관측 객체의 가림 구간 AR이 좋아지면 진행한다. 대칭 등가해를 틀렸다고 세거나 CAD를 한쪽에만 주면 비교를 중단한다. [ConceptPose](https://openaccess.thecvf.com/content/CVPR2026/papers/Kuang_ConceptPose_Training-Free_Zero-Shot_Object_Pose_Estimation_using_Concept_Vectors_CVPR_2026_paper.pdf) · [BOP](https://bop.felk.cvut.cz/leaderboards/)

### #56 비디오 추적·VOS

SAM 3.1은 객체 마스크 추적, CoTracker3는 실제 영상 가짜 라벨을 이용한 점추적 기준이다. 유사한 두 사람이 가림 후 자리를 바꾸면 평활한 마스크가 다른 사람을 따라갈 수 있다. 같은 최초 프롬프트, 추적 질의, 메모리 및 온라인 지연에서 DAVIS나 SA-Co Video의 J&F/마스크 성능과 TAP-Vid의 AJ·가림 후 복귀율을 따로 평가한다. 실제 가림 후 오류와 ID 교체가 줄고 시간 비용이 유지될 때 진행한다. 깜빡임만 줄고 재등장 감지를 잃으면 중단한다. [SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/) · [CoTracker3](https://cotracker3.github.io/)

### #57 sim2real 체화

SimVLA와 합성 사전분포로 real 적응하는 world-action 연구가 각각 zero-shot·few-shot의 직접 기준이다. RGB와 3D 외관이 일치해도 마찰이 낮으면 집기가 실패한다. 로봇, 사전학습, 실 시범 수, 온라인 상호작용 예산을 고정하고 우선 tabletop 한 행동에서 모습 랜덤화와 동역학 보정의 효과를 분리한다. 독립 실제 환경에서 성공률, 접촉 실패, 실측 시간을 집계하고 동일한 초기 조건으로 paired 비교한다. 실제 성공률과 비용-성공 곡선이 좋아질 때 진행한다. 시뮬레이터 안에서만 이기는 경우 실제 제어 주장을 중단한다. 자율주행 지각은 별개 검증이 필요하다. [SimVLA](https://arxiv.org/abs/2610.11248) · [Efficient Sim-to-Real Transfer](https://arxiv.org/abs/2606.31101)

## #58 저중첩 재구성과 앵커

**가까운 비교.** MVSplat360은 낮은 중첩을 애초에 문제로 설정한다. Gupta 등의 CVPR 2026 논문은 비정제 소수 뷰를 쓰며 합성 가림까지 훈련에 넣는다. 따라서 단순 저중첩 합성 학습 또는 합성 가림 자체를 논문 기여로 삼기 어렵다. DA3/VGGT 계열의 pose-free 비교와, 같은 GT 카메라를 받는 희소 합성 기준을 따로 둔다. 전통 robust BA는 비용 대비 순위를 확인할 기준이다. [MVSplat360](https://arxiv.org/abs/2411.04924) · [Generalizable Sparse-View 3D Reconstruction](https://openaccess.thecvf.com/content/CVPR2026/papers/Gupta_Generalizable_Sparse-View_3D_Reconstruction_from_Unconstrained_Images_CVPR_2026_paper.pdf) · [Depth Anything 3](https://arxiv.org/abs/2511.10647)

**작은 반례.** 정규화 핀홀에서 카메라 중심이 0과 1이고 점이 (2,2)라면 영상 가로좌표는 1과 0.5다. 카메라 중심과 점을 두 배로 해서 중심 0과 2, 점 (4,4)로 두어도 두 좌표가 같다. 그래프가 연결돼도 보정된 단안 SfM에는 일반적으로 전체 Sim(3)의 자유도 7개가 남는다. 3D–2D 대응 그래프가 두 독립 성분으로 나뉘면 각 성분의 변환이 독립이다. 한 성분의 기준 거리를 쟀다고 다른 성분의 거리까지 결정되지는 않는다. 매칭이 있어도 순수 회전은 삼각측량의 깊이 시차를 만들지 못한다. 카메라가 이미 metric 단위로 고정됐으면 이 포즈-스케일 예의 전제는 바뀌지만, 한 카메라에만 보이거나 가려진 점의 깊이는 여전히 고유하지 않다.

**다음 벤치.** 카메라 수를 고정하고 GT로 양방향 가시 표면 비율을 정의한다. 0, 5–10, 10–25, 25–50, 50% 이상 구간과 가시 그래프 연결도, 유효 대응과 시차를 함께 기록한다. 같은 평균 중첩에서 다리 연결 하나의 유무를 맞바꾸고, 깊이 앵커 또는 연결 샷에 같은 가격표를 붙인다. 실측 위치 선택은 동일 후보에서 균일·고정·D-opt·관심축 A-opt를 비교한다. 실측값과 저감도 방향을 결정할 때 검사 GT를 쓰지 않는다. 포즈 ATE를 전역 정렬 전후로 분리한다. 절대 스케일, 깊이/점군 정확도와 완전도, 그리고 생성 이미지 품질을 따로 보고한다. “60–80% 필요”라는 범용 임계값은 연결·시차·정합이 섞여 있어 제외한다.

**관문.** 0 중첩에서는 식별 불가 표시 또는 외부 앵커로 회복하는지 확인한다. 낮지만 연결된 구간에서는 같은 예산에서 기하 오차와 기권률의 비용 곡선이 표준 정보 선택보다 좋아질 때 진행한다. 겹치지 않는 영역의 외형을 그럴듯하게 채우기만 하면 재구성 주장에는 쓰지 않는다.

## 정리하면

#43의 원본 평균값은 재현됐다. 주기의 강제 관측과 정확한 일관성 방향에 크게 의존하고, 원본의 실패율은 46%였다.  
#58은 연결과 시차가 약하면 정확한 복원이 어려워진다는 점이 핵심이다. 실측에 돈을 어디에 써야 이득인지 실험할 수 있다.  
추천은 #43의 정보 선택 비교와 #58의 연결·시차 비교를 먼저 여는 것이다. 제품과 바로 연결할 실험 한 건을 고른다면 #44의 절대 깊이 또는 #53의 실제 매팅을 골라야 한다.
