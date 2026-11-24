# 음향 스튜디오와 신뢰도 필터 독립 반박 검수

판정 1라운드 **NOT READY**. #59–62의 넓은 차별점은 선행연구가 이미 구현했다. #63의 1차원 성능 수치는 재현됐다. 일반 수축, 이상치 주입 유계, 무정보 품질에서 무손해를 주장하려면 각각 별도 근거가 필요하다.

## 기존 근거와 이번에 추가한 검수

- 기존 원장: lab의 origin/main 3aaa4717046ba327995acae73cce737ce064632d, wiki/synthesis/research-topics-ledger.md 216–230행. 음향 보충은 wiki/synthesis/digital-studio-benefit-map.md, 필터 상세는 wiki/concepts/reliability-gated-filtering.md.
- 기존 실행: note의 origin/reliability-gated-filtering a71f9adc0b0fa8e5ad7b2e0cbf2e1d47e19919bd. src/toy_reliability_gate.py 29–95·134행, src/toy_learned_reliability.py 29–135·155행.
- 추가 근거: 이 문서의 원문 대조, 두 toy 재실행, 공통 최악값 역추적, 수축 반례와 조건의 재유도. 원문의 넓은 주장과 실험의 좁은 범위를 함께 판정했다. 투고 원고 문체 검수는 대상 원고 미제공으로 미실시.

## 음향 네 후보

| 역할 | 가장 가까운 원문이 이미 한 일 | 남는 실험과 실패 조건 |
|---|---|---|
| #59 음성 향상 · 잡음 속에서 대사를 살림 | DNN-augmented colored-noise Kalman filter는 신경망 파라미터 추정과 칼만 잡음제거를 결합했다. DNS Challenge 및 CHiME는 합성 쌍과 실측 평가·음향 시뮬레이션을 제공한다. [하이브리드 KF 원문](https://www.sciencedirect.com/science/article/abs/pii/S0167639320302831), [MS DNS 공식 저장소](https://github.com/microsoft/DNS-Challenge), [CHiME 공식 비교군](https://www.chimechallenge.org/challenges/chime4/software.html) | 고정 합성 학습 대 실측 보정 대 **실측 신호로 다음 합성 RIR·잡음을 선택**을 같은 분량과 프레임 지연에서 비교. 방·마이크·화자·잡음 계열을 단위로 분리. SI-SDR, STOI와 실측 주관 점수를 함께 사용. 실측 잡음이 말소리와 비슷하거나 클리핑·AGC가 지배하면 RIR 보정으로는 부족하다. 보정 루프가 동일 실측량의 단순 미세조정을 못 이기면 보류. |
| #60 AV 분리 · 화면에서 말하는 사람의 소리를 골라냄 | Looking to Listen은 화자 독립 영상 기반 음성 분리를 실증했다. CaffNet은 전역·프레임별 AV 대응과 지연·지터 상황의 분리까지 다룬다. VoiceFormer는 비동기 시청각 단서를 처리한다. [Google 원문](https://storage.googleapis.com/gweb-research2023-media/pubtools/4588.pdf), [CaffNet CVPR 원문](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Looking_Into_Your_Speech_Learning_Cross-Modal_Affinity_for_Audio-Visual_Speech_Separation_CVPR_2021_paper.pdf), [VoiceFormer CVPR 원문](https://openaccess.thecvf.com/content/CVPR2022/papers/Rahimi_Reading_To_Listen_at_the_Cocktail_Party_Multi-Modal_Speech_Separation_CVPR_2022_paper.pdf) | 같은 분리 모델로 영상 없음·기존 affinity·품질 게이트·잘못 연결한 입 영상 비교. 새 영화/방/화자에서 장면 전환, 더빙, 화면 밖 발화, 두 사람의 동시 발화, 가림을 교차. SI-SDRi와 화자 바꿔치기율, 영상 없는 구간의 음성 훼손을 사전 지정. 실제 측정한 일관성이 맞는 화자와 틀린 화자를 가르지 못하면 종료. AV 일관성 자체는 신규성이 없다. |
| #61 방향 추적·빔포밍 · 마이크를 소리 쪽으로 돌리는 역할 | Berghi 등은 시각 의사라벨을 마이크 어레이 빔포밍과 결합해 화자 위치를 학습하고, 추론은 음향만으로 수행했다. AV 다채널 MVDR와 실제 재생 평가도 선행한다. LOCATA는 고정·이동 조건의 실측 비교 기반이다. [MMSP 원문 기관 저장소](https://openresearch.surrey.ac.uk/esploro/outputs/conferenceProceeding/Visually-Supervised-Speaker-Detection-and-Localization/99598320002346), [AV 다채널 MVDR 원문](https://arxiv.org/abs/2307.02909), [LOCATA 공식](https://www.locata.lms.tf.fau.de/challenge-description/) | 고정 지오메트리와 온라인 보정을 비교하고, 시각 의사라벨 모델 및 오디오 SRP-PHAT+MVDR를 같은 어레이에서 비교. 카메라-어레이 외부 파라미터 변화, 클록/마이크 위상 오차와 반사 착시를 나눠 시험. 각도오차·놓침·음성 출력의 SI-SDRi를 함께 본다. 가림과 반사가 겹치면 두 센서의 신뢰오류가 연동될 수 있다. 이 경우 보정 개선이 재현되지 않으면 보류. |
| #62 잔향·에코 제거 · 방 울림과 스피커 되울림을 걷어냄 | 참조 신호가 있는 AEC는 NKF가 공분산을 암묵적으로 학습해 게인을 바로 출력한다. 급변·동시발화·실녹음도 시험했다. 잔향 제거는 WPE가 기본 비교군이고, Kalman-EM이 에코·잔향·잡음의 온라인 동시 제거를 이미 다뤘다. [NKF 원문 및 저자 코드](https://fjiang9.github.io/NKF-AEC/), [Kalman-EM 원문](https://link.springer.com/article/10.1186/s13636-021-00219-2), [WPE 원 논문 서지](https://cir.nii.ac.jp/crid/1362544418584529024), [AEC 2023 공식](https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-icassp-2023/) | AEC와 잔향 제거를 별도 시험한다. 전자는 NKF/FDKF와 같은 입력에서 에코 경로 급변과 근단 발화를 따로 조절하고, ERLE·근단 손상·재수렴시간을 함께 측정한다. 후자는 동일 마이크 수·목표 정의에서 WPE/DNN-WPE와 비교한다. 새 기기·방·비선형 스피커, 참조 클록 오차에서 개선이 남는지 본다. 동시발화의 근단 음성까지 자르면 ERLE 향상으로 통과시킬 수 없다. |

공통 관문으로는 중복 방·화자·마이크 분리를 먼저 고정하고, 논문 구현 원형과 같은 데이터·지연에 맞춰 튜닝한 버전을 모두 둔다. 독립 방/장면으로 짝지은 신뢰구간을 보고한다. **같은 비용의 무작위 합성 갱신·무갱신·실측 미세조정**을 모두 이긴 경우에만 적응적 합성 갱신의 효과로 해석한다. 합성기에서만 얻는 품질 정답과 실제로 배포 중 얻을 수 있는 품질을 분리한다. 음향 네 후보는 이 단계에서 문헌과 실험 설계 검수만 했다.

## #63 선행연구와 독창성

| 역할 | 무엇을 하나 | 쉬운 비유 |
|---|---|---|
| 관측 품질 → 잡음 공분산 | 카메라·마이크 입력에서 읽은 품질로 이번 관측을 얼마나 믿을지 정한다. | 흐린 사진에 작은 점수를 준다. |
| 모델 잔차 → 이상치 완화 | 예측한 값과 관측의 차이를 확인하고 큰 차이의 업데이트를 줄인다. | 예상에서 크게 벗어난 수치를 다시 확인한다. |
| 오차 수축 | 같은 입력에서 두 초기값의 차이가 시간과 함께 줄어드는지 본다. | 다른 출발점에서 같은 길로 모인다. |

- **비잔차 품질로 관측 공분산을 바꾸기**는 선행한다. GIAOTracker의 NSA KF는 식 (2)에서 검출 confidence로 관측 공분산을 조절한다. StrongSORT는 이 식을 채택했다. DICE는 원 센서 데이터에서 공분산을 지도학습하며, Backprop KF도 이미지 관측으로부터 필터의 관측과 불확실성을 학습한다. DICE에 필요한 참 관측오차 라벨을 의도적으로 빼고 동일한 재보정 성능을 얻는 조건은 별도 질문이다. [GIAOTracker CVF 원문, Algorithm 1](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Du_GIAOTracker_A_Comprehensive_Framework_for_MCMOT_With_Global_Information_and_ICCVW_2021_paper.pdf), [StrongSORT 저자 원문, NSA KF](https://arxiv.org/pdf/2202.13514), [DICE MIT 원문](https://groups.csail.mit.edu/rrg/papers/LiuOk18icra.pdf), [Backprop KF NeurIPS 원문](https://proceedings.neurips.cc/paper/2016/file/697e382cfd25b07a3e62275d3ee132b3-Paper.pdf)
- **잔차에서 잡음 공분산을 맞추기**는 기존 혁신 기반 공분산 식별과 이어진다. 원 toy의 B는 고정 자유도에서 한 번만 스케일을 계산한 구현이다. Huang 등의 반복 VB Student-t KF 전체를 대표하는 대조는 아니다. 잔차기반 방법도 전이모형에 급변을 허용하거나 공정하게 Q를 맞추면 결론이 달라질 수 있다. [Mehra IEEE 원문 DOI](https://doi.org/10.1109/TAC.1970.1099422), [Huang 등 IEEE 원문](https://ieeexplore.ieee.org/document/7814285/)
- **시변 환경의 필터 수축**도 기존 축이다. Bougerol은 Riccati 변환의 양정치 행렬 공간 수축을 분석했다. 이는 필터 상태 오차의 매 스텝 유클리드 수축과 다른 결과다. 결측이 비관측 구간을 길게 만들 때는 관측 도착률이 공분산 유계성을 좌우할 수 있다. [Bougerol SIAM 원문](https://epubs.siam.org/doi/10.1137/0331041), [Sinopoli 등 저자 원문](https://people.eecs.berkeley.edu/~sastry/pubs/SastryAllJuly07/SastryALL/2003/SinopoliKalman2003.pdf)

남는 연구 질문은 **참 상태 라벨 없이 품질을 교정했을 때, 갑자기 움직인 상태를 빨리 따라가면서 품질 센서가 틀린 구간도 버틸 수 있는가**이다. 라벨 없는 공분산 식별, 품질 분포 이동, 보장 범위를 별도로 엄밀화해야 한다.

## #63 수축 식 재유도

관측 업데이트를 $B_k=I-K_kH_k$로 두면, $P^-\succ0$일 때 행렬 역행렬 항등식으로

$$B_k=\bigl(I+P^- H_k^\top R_k^{-1}H_k\bigr)^{-1}.$$

같은 관측·품질·게인 경로를 쓰는 두 필터의 사후 차이에는 **동역학을 포함한** $F_k=B_k A_{k-1}$가 곱해진다. 참 상태와 비교하는 오차에는 과정 잡음과 관측 잡음이 추가된다. 게인이 잔차나 추정치에 따라 달라지면 두 필터의 게인까지 달라져 이 차이식만으로는 수축을 증명할 수 없다.

- **하한과 상한**. 1차원에서 $H=1$이면 $B=R/(P^-+R)$. $R\ge r_{\min}>0$은 $B$를 **아래에서** 제한해 $K$ 상한을 준다. 수축에 필요한 $B$ 상한은 $R\le r_{\max}$와 $P^-\ge p_{\min}>0$에서 $B\le r_{\max}/(p_{\min}+r_{\max})<1$로 나온다. $P^-=1$에서 $R=1\to100$이면 $B=0.5\to0.9901$이다. $P^-\to0$인 축도 $B\to1$을 만든다. 이상치에서 센서 관측을 외면할수록 관측으로 오차를 줄이기는 어려워진다.
- **동역학**. $A=2,H=1,P^-=1,R=4$면 $I-KH=0.8$이지만 $F=1.6$이다. 이 값은 해당 한 스텝 사례다. 같은 상수를 사용하는 가변 모델의 장기 발산을 이 숫자만으로 추론할 필요는 없다.
- **행렬과 거리**. $M=(P^-)^{1/2}H^\top R^{-1}H(P^-)^{1/2}\succeq0$라 두면 $(P^-)^{-1/2}B(P^-)^{1/2}=(I+M)^{-1}$. 이 대칭 행렬 노름은 $1/(1+\lambda_{\min}(M))$이다. $H$가 전열 계수이며 최소 특잇값이 균일하게 양수이고, $P^-\succeq p_{\min}I$, $R\preceq r_{\max}I$면 이 거리에서 한 번의 업데이트를 균일하게 줄인다. 원래 유클리드 노름의 보장은 아니다. $H=I$, $P^-=\left[\begin{smallmatrix}1&2\\2&5\end{smallmatrix}\right]$, $R=\operatorname{diag}(1,100)$이면 $B=\left[\begin{smallmatrix}105&-2\\-200&200\end{smallmatrix}\right]/206$이고 $\|B\|_2=1.42441$이다. 양의 공분산과 가역 관측이 모두 성립한다. 행렬 게인의 성분이 일제히 증가한다는 주장도 순서·거리의 정의 없이 성립하지 않는다.
- **시간 창**. $H$ 전열 계수를 매 시점 요구할 필요는 없다. 개별 $H_k$가 일부 축만 보거나 결측이면 $A_k$가 보이지 않는 축의 오차를 키울 수 있다. 시간 창마다 $\sum_j\Phi_{j,k}^\top H_j^\top R_j^{-1} H_j\Phi_{j,k}\succeq\alpha I$ 같은 균일 관측성으로 궤적 전체에서 신호가 들어오는지를 확인하고, 표준 Riccati 유계성 조건을 별도 증명·인용한다. 예를 들어 $H=(1,0)$이면 두 번째 축은 그 스텝에서 줄지 않는다. $H$가 간헐적으로 달라져 두 축을 반복 관측하는 경우는 창 전체로 판정해야 한다. 창의 평균 관측률만으로 최장 공백이 제한되는 것은 아니다.
- **가능한 수정 명제**. 표준 일관 Riccati 업데이트로 $P_k^+=B_kP_k^-$, $P_k^-=A_{k-1}P_{k-1}^+A_{k-1}^\top+Q_{k-1}$라 두자. $Q_{k-1}\succeq qI\succ0$ 및 $P_k^-\preceq\bar p I$이면 $F_k^\top(P_k^+)^{-1}F_k \preceq A_{k-1}^\top(P_k^-)^{-1}A_{k-1}$이다. 이는 $P_k^+\preceq P_k^-$를 대입하면 나온다. 또한 $A_{k-1}P_{k-1}^+A_{k-1}^\top=P_k^--Q_{k-1}\preceq(1-q/\bar p)P_k^-$이므로 $A_{k-1}^\top(P_k^-)^{-1}A_{k-1}\preceq(1-q/\bar p)(P_{k-1}^+)^{-1}$. 사후 공분산 역행렬을 거리로 쓰면 동일 외생 게인 경로의 초기값 차이가 줄어든다. 공통 유클리드 지수 오차 상계로 옮기려면 $P_k^+$의 균일 상하한이 추가로 필요하다. 해석은 초기값 망각이며, 실제 잡음에 대한 성능 보장은 잡음 항과 가정이 따로 필요하다.
- **게인 상한과 이상치 영향**. $\|P^-\|\le \bar p,\ \|H\|\le\bar h,\ R\succeq r_{\min}I$이면 $\|K\|\le\bar p\bar h/r_{\min}$. 그래도 고정된 $q$에서 $K\nu$는 $\nu$가 무제한이면 무제한이다. $K=0.5,\nu=2\to200$이면 주입이 $1\to100$이다. 유계 주입에는 잔차 포화나 등가 구조가 필요하다. 포화 후에는 급변 추종과 수축 조건을 재검토한다. toy의 잔차 백업도 분산을 유한 배수로만 올려 이 수학적 유계성을 보장하지 못한다.

## 원 toy 재실행과 학습형 반박

확인 명령은 각 소스에 대해 PowerShell에서 git -C note show origin/reliability-gated-filtering:src/파일.py 출력을 C:/ProgramData/miniforge3/python - 로 넘겼다. PYTHONIOENCODING=utf-8, NumPy 2.4.4. 실험 파일을 변경하지 않았다.

| 비교(낮을수록 좋음) | 기존 고정 R → 품질 변형 | 뜻 |
|---|---:|---|
| toy 1 비점프·비이상치 RMSE | 1.126 → 0.498 | 약 56% 감소 |
| toy 1 점프 뒤 5스텝 평균절대오차 | 3.101 → 1.376 | 약 56% 감소. 회복시간은 미측정 |
| toy 1 이상치 시점 최대오차의 시드 평균 | 3.74 → 1.75, 백업 1.55 | 약 53%, 백업은 약 59% 감소. 모든 방법의 공통 최악 17.8904286은 시드 56의 첫 프레임 |
| toy 2 비점프·비이상치 RMSE, 품질 0.8 | 1.148 → 0.563, 참분산 0.481 | 약 51% 감소. 참분산 비교군도 모델과 이상치 분산 클램프가 있는 진단용 |
| toy 2 점프 뒤 5스텝 평균절대오차 | 3.108 → 1.715, 백업 1.730 | 약 45% 감소. 백업이 일반 무손해라는 근거는 없다. |
| toy 2 비점프·비이상치 RMSE, 품질 1.0 → 0.0 | 0.579 → 1.119 | 약 1.9배 커짐. 고정 R 1.148은 이 고정 표본·고정 비교군의 값 |

- 두 toy는 생성 코드에서 점프와 이상치가 동시에 발생하지 않는다. README의 “점프+이상치 겹침” 설명은 틀렸다. $x_0$를 첫 관측값으로 초기화하면 첫 잔차가 0이라 네 방법 모두 첫 이상치를 그대로 받는다. 비점프·비이상치 RMSE에는 점프 직후 1–4프레임과 이상치 직후 프레임이 남는다.
- 학습기는 현재 품질 신호만 보고 테스트 게인을 정하지만, 보정 중에는 혁신 제곱을 쓴다. 일반식은 $\nu=He^-+v$이고 $E[\nu\nu^\top\mid c]=HE[e^-(e^-)^\top\mid c]H^\top+E[vv^\top\mid c]+$ 교차항이다. 교차항 0, 조건부 편향과 예측오차 공분산의 올바른 책정이 선행돼야 한다. toy의 예고 없는 점프, 품질과 점프의 강한 연동, $Q_{\rm true}=0.01$과 $Q_{\rm filt}=0.02$ 차이는 이 가정을 자동으로 충족하지 않는다. 품질을 현재 관측으로 계산할 때 잔차를 직접 입력하지 않아도 통계적 독립은 성립하지 않을 수 있다.
- 코드가 맞추는 것은 **조건부 중앙값**이다. 깨끗한 정규혁신만 있어도 $\operatorname{median}(\nu^2-P^-)\simeq0.455(P^-+R)-P^-$여서 $R$와 다르다. 예를 들어 $P^-=R=1$이면 평균은 1이고 중앙값은 약 −0.09가 되어 코드에서는 하한으로 잘린다. 데이터 생성분포로 직접 계산하면 품질 0.8, 관측 품질 $c=0.1$의 이상치 가중은 $0.245/(0.245+0.581875)\simeq0.296$이고 $E[r\mid c=0.1]\simeq30.87$다. 표의 2.41은 평균 복원으로 해석할 수 없다. 주석에는 최소제곱을 쓴다고 되어 있지만 실제 구현은 12개 구간 중앙값이다. 앞선 최소제곱 실패 실행의 원본 스크립트·로그는 이 브랜치에서 미확인이다.
- 무정보 스윕은 매 품질 설정에서 보정 모델을 다시 학습한다. 이미 학습한 맵에 실제 품질이 갑자기 무정보가 되는 상황과 다르다. 유한 표본의 무정보 구간별 중앙값은 서로 같을 이유가 없고, 표본이 무한이어도 공통 중앙값이 기존 고정 R인 0.7과 같아야 할 이유가 없다. “절대 고정 R보다 나빠지지 않는다”는 결론은 철회한다.
- 새 비교에서는 점프와 품질 및 이상치를 독립 교차한다. 무정보·부호가 뒤집힌 품질·과신형 이상치·센서 장기 결측·새 방 보정을 사전 지정한다. 같은 보정 데이터에 최적으로 맞춘 고정 R/Q, StrongSORT식 confidence 맵, 라벨을 쓴 DICE식 공분산 예측, 혁신 가능도 기반 비지도 품질 맵, 원형 Student-t VB와 Q 조정/급변 전이 대조를 둔다. 품질 맵을 동결한 이동 실험과 새 도메인 재교정 실험을 각각 보고한다. 정상 RMSE, 실제 회복시간, 혁신 보정 오차, 최악 구간을 분리한다. 이득이 품질-점프 결합을 풀었을 때 사라지거나 튜닝된 비교군에서 사라지면 일반화 주장은 종료한다.

## 한계와 정리

**한계**. 문헌 비교는 확인한 원문 기준이며 독창성의 전수 보증 범위는 아니다. 두 toy의 출력과 위 행렬 반례는 실행했다. 음향 실측, 다차원 창 수축, 동결 품질 모델 이동 실험은 이 검수에서 미실행이다.

정리하면

- 음향 네 가지는 각각 가까운 선행연구가 있다. 차이는 실측으로 다음 합성을 바꿔 얻는 추가 이득으로 검증할 수 있다.
- #63은 합성 1차원 toy 수치가 재현됐다. 원 수축 식과 학습의 무손해·평균 복원 주장은 수정이 필요하다.
- 추천은 #63의 관측 품질 이동 및 공정한 비교 관문을 우선하고, 음향 확장 시 실제 배포에서 쓸 품질 신호와 평가 장면을 먼저 확정하는 것이다.

