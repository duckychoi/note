# #39~#42 독립 이론 반박 검수 (1차)

판정은 연구 관문 기준 **NOT READY**다. 기존 코드의 숫자는 재현됐고, 각 실험이 실제 증명하거나 비교한 범위는 원장 문장보다 좁다. 아래 결과는 원장 등재·개념 검토와 별개의 반박 검수다.

## #39 소산성 신경 KF 세계모델

- 원 코드 [0eabb37](https://github.com/duckychoi/note/blob/0eabb37/src/toy_dissipative_kf.py#L23), 로그 `theory_replays/toy_dissipative_kf.txt`. 120회, 18 스텝의 양 상태 관측에서 2×2 행렬을 최소제곱으로 적합하고 고유값 크기만 0.98로 잘랐다. 불안정 11%→0%, 150번째 norm 중 최대 16.0→0.5, 평균 0.6→0.2. '최대'는 전체 궤적의 최대가 아니다. 최종 norm >10이 1%→0%이며 유한시간 검사다. KF 예측·보정, 신경망, 입력–출력 확장 소산성 성능은 구현하지 않았다.
- 고정 행렬 A가 Schur 안정이면 P=Σ(A^j)ᵀA^j는 양의 정부호이고 AᵀPA−P=−I다. 따라서 무입력 V=xᵀPx가 감소한다. 유계 외란에는 x_t=A^t x_0+Σ A^j w_{t−1−j}이고 ||A^j||≤C r^j를 적용하면 유계다. 차원 2에서 A=[[.5,1],[0,.5]], B=Aᵀ는 둘 다 ρ=.5지만 ||A||₂≈1.207이고 ρ(AB)≈1.457로 교대하면 발산한다. 시간가변 백본은 공통 P나 별도로 증명된 모드 전환 제한이 필요하다. 고유벡터 역행렬을 쓰는 구현도 결함 행렬에 일반화할 수 없다.
- 잠재 선형가우시안 Mamba KF/스무딩과 제약 학습·관측기 선행이 있다. 조합 자체의 신규성은 미확인이다. 중단 관문은 안정 칼만/REN급 기준선보다 외란 응답 및 장기·폐루프 정확성에서 이득이 없거나, 보장에 쓰는 가정이 동작 중인 잠재 동역학을 덮지 못하는 경우다. [KalMamba](https://arxiv.org/html/2406.15131v1), [KalmanNet](https://arxiv.org/abs/2107.10043), [REN](https://arxiv.org/abs/2104.05942).

## 별도 #39 점유분포 모방

- [개념노트 §3](https://github.com/duckychoi/lab/blob/3aaa471/wiki/concepts/occupancy-matching-imitation.md#L25)의 '증명済'는 본가 [docs/05](https://github.com/duckychoi/note/blob/fm-reward-v2/docs/05-fm-residual-divergence-theorem.md#L3)에서 11-22에 이미 철회한 내용이다. 현 교정안 [34-audit §2](https://github.com/duckychoi/note/blob/fm-reward-v2/docs/v2/34-audit-20261122.md)는 t/(1−t) 가중의 정확한 marginal velocity discrepancy 등식이다. 고정된 전문가 잔차에 붙어 있는 정책별 조건부 분산 C(q)는 q에 따라 바뀐다. MDP occupancy 일치에는 초기분포·동역학·가능한 정책 집합·엔트로피 정의를 맞춘 별도 목적 등식이 필요하다. GAIL의 기본 대상은 할인 상태–행동 점유분포다.
- 조상 2D toy 두 건을 재실행했다. FM 점수와 log p 순위상관 .932, 완전/덜 학습 판별자는 .607/.537 평균이고 표준편차 .016/.069다. FM 분산 0은 고정 실행 한 번에서 나온 값이다. 정답 log p를 넣는 soft 최대화는 온도 1에서 모드 가중치를 회복했지만, 잔차로 학습한 MDP 정책은 아니다. R–KL 순위상관 .727이나 shift 1→1.5에서 KL 1.560→3.645, 잔차 7.393→5.407로 역전된다. `theory_replays/toy_fm_vs_amp.txt`, `theory_replays/toy_fm_divergence.txt`.
- 2026 FMP는 AMP 대신 온라인 rollout→expert OT 경로 FM으로 스칼라 모션 보상을 학습한다. SMP는 동결한 diffusion score prior를 실제 휴머노이드 보상에 재사용한다. SMILING은 전문가와 학습자의 score를 비교한다. IL-flOw는 전문가 상태전이 log density를 고정 보상으로 쓴다. SOIL-TDM은 비적대적 궤적 매칭을 실제 수행하고, EnergyFlow는 score로 soft-Q의 gradient를 복원한다. 방어 가능한 질문은 고정 expert-only 잔차가 이 보상보다 새 순서·희소 모드를 같은 정책/계산 예산에서 더 충실히 보존하는 특정 조건이다. 전문가 log density 오라클이 재현되는데 잔차가 못 따라가면 대리 목적은 종료한다. [FMP](https://arxiv.org/html/2609.15631v1), [SMP](https://arxiv.org/abs/2512.03028), [SMILING](https://arxiv.org/abs/2410.13855), [IL-flOw](https://arxiv.org/abs/2205.09251), [SOIL-TDM](https://doi.org/10.1007/s10489-023-05062-w), [EnergyFlow](https://proceedings.mlr.press/v306/ji26e.html), [GAIL](https://arxiv.org/abs/1606.03476), [AMP](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf).

## #40 복소·텐서네트워크 세계모델

- [개념노트](https://github.com/duckychoi/lab/blob/3aaa471/wiki/concepts/quantum-inspired-worldmodel.md)에 가설만 있다. 재실행할 독립 코드·데이터 미확인이다. 실수 두 채널로 a+be^{iφ}의 실수·허수 성분을 쓰면 같은 교차항을 얻는다. 동일 변수 쌍의 합은 비음 확률 혼합에서 상쇄되지 않지만 실제 비교는 동일 자원의 부호 있는 실수·복소 모형이어야 한다. Born MPS 생성과 제한된 rank에서의 양·실수·복소 분해 표현력 차이는 선행 공개됐다.
- 하나의 질문은 저결합 환경에서 같은 바이트·지연의 실수 상태공간 또는 비음 tensor train보다 다봉 예측 정확도·계획 성공률이 좋은지다. 동등한 실수 두 채널 또는 저랭크 실수 기준선과 결과가 같으면 복소 표기를 독립 기여로 세울 수 없다. [Born MPS](https://doi.org/10.1103/PhysRevX.8.031012), [텐서네트워크 표현력 비교](https://proceedings.neurips.cc/paper_files/paper/2019/hash/b86e8d03fe992d1b0e19656875ee557c-Abstract.html).

## #41 복소/텐서넷 KF

- 원 코드 [c33b50a](https://github.com/duckychoi/note/blob/c33b50a/src/toy_complex_vs_real_kf.py#L17). Part 1은 512점 균일 사전의 실수 grid likelihood 정규화 vs 이미 정확한 사후로 모멘트를 추출한 가우시안이며, 실제 KF 업데이트나 복소 연산이 없다. 모드질량 .25→.97은 단봉 근사의 손실이다. Part 2는 미리 정한 진폭을 합성했으며 위상 0, π/2, π에서 겹침질량 .782/.714/.494, 고정 혼합 .714가 그대로 재현됐다. 이는 관측이 위상을 알아내거나 예측·갱신에 쓰인 실험이 아니다. `theory_replays/toy_complex_vs_real_kf.txt`.
- 임의 grid 사후 p의 ψ=√p는 항상 실수로 가능하다. likelihood L에 대한 ψ⁺∝√Lψ는 실수 Bayes 보정과 같다. Markov 전이의 p⁻=T p는 경로 확률을 더하지만 복소 선형 진폭 전이는 일반적으로 교차항을 더한다. 같은 의미의 모델이 되려면 예측 법칙과 위상 식별 조건부터 필요하다. 복소 선형 Gaussian KF는 여전히 단봉이고, tensor network KF는 선형가우시안 계산의 저랭크 형식인 사례가 있다. 실수 grid/TT는 이미 샘플 없이 다봉 확률을 필터링한다.
- 중단 관문은 여러 시점의 상태전이·관측으로 식별되는 위상이나 구조가 없거나, 그리드·Gaussian sum·실수 TT·PF와 같은 정확도에서 시간/메모리 개선이 없는 경우다. [복소 KF](https://pubmed.ncbi.nlm.nih.gov/24806126/), [TNKF](https://www.sciencedirect.com/science/article/abs/pii/S0005109817303023), [실수 grid/TT Bayes](https://doi.org/10.1080/17415977.2020.1862109).

## #42 합성데이터 강건학습

- 원 코드 [f84e410](https://github.com/duckychoi/note/blob/f84e410/src/toy_kf_synthetic_collapse.py#L36), 로그 `theory_replays/toy_kf_synthetic_collapse.txt`. 고정 mix·KF·적응 KF의 최종 평균절대오차 .230/.146/.287, 변화 후 첫 평균 0.3 미만까지 35/7/3 인덱스, 누적오차 41.19/16.39/19.96. 첫 변화 시점을 오프셋 0으로 세며 개별 시드의 지속 회복을 측정하지 않았다. 분산 .12/.92/1.04는 표본 2차 모멘트이며 꼬리·모드 보존의 실증은 아니다.
- KF 두 팔이 매 세대 만든 합성 표본은 난수 소모 외 사용되지 않는다. 분산도 실측 표본분산을 계수 .5로 갱신한다. 원래 비교군은 실측 5/전체 55 고정이고 두 팔의 분산 추정법도 다르다. 추가 paired 진단은 원본의 난수 소모를 유지하되 실데이터 이동평균만 사용했다. gain .0909에서 .08485/24/31.94, .270156에서 .14630/7/16.38576으로 KF와 사실상 일치, .5에서 .21122/3/16.53, .75에서 .28060/2/19.92다. 같은 300 seed를 쓴 탐색 수치로 독립 사전등록 시험은 아니다. `theory_replays/independent.py`, `theory_replays/independent.txt`.
- 평균 오차는 e⁺=(1−K)e+Kη로 수축할 수 있다. 단 K가 양이라는 항목별 조건만으로 일반 균일 수축은 성립하지 않으며, 평균오차 유계는 꼬리 복원을 보장하지 않는다. 이 스칼라 코드의 고정 참평균 + Gaussian 실측에서는 2차 모멘트로 표현 가능한 정상 오차를 논할 수 있으나, 실현별 결정론적 유계와 보정된 coverage는 검증되지 않았다. 실제 naive의 E[V_t]=.8(.98)^t라 평균은 martingale이며 증분 분산의 합은 .8에 수렴한다. 초기 .2를 더하면 한계 평균제곱오차는 1이다. 평균이 무한히 발산한다는 코드 설명은 틀렸고, 표본분산 붕괴와 무작위 평균 편차의 정착이 관측된다.
- 2512.00757 §4.1에는 참평균으로 거리를 매겨 실험 학습 라벨을 붙인 사례가 있다. 이것으로 모든 수축 필터를 오라클 의존으로 묶을 수 없다. 실데이터 축적과 fresh-real injection 선행도 확보되어 있다. 같은 실데이터·계산·메모리 예산에서 real-only EMA/변화점 추정, 튜닝한 mix, 누적 실데이터/합성데이터와 비교하고, 합성 제거가 손해를 주는 문제에서 모드·꼬리 또는 forecast calibration의 증분이 있어야 재개한다. [수축 신경 필터](https://arxiv.org/html/2512.00757v1#S4), [Gerstgrasser 데이터 축적](https://arxiv.org/abs/2404.01413), [Shumailov 모델 붕괴](https://doi.org/10.1038/s41586-024-07566-y).

## 재현 범위

상응하는 주제의 최신 확인 브랜치에서 `git show origin/<branch>:src/<파일>.py`로 코드를 추출하고 로컬 NumPy로 실행했다. 모방의 두 결과는 `origin/fm-reward-v2`의 조상 toy이며 별도 #39의 정식 새 관문이 아니다. 원격 소스 자체는 변경하지 않았다. 근거 문헌은 비교에 직접 필요한 부분을 확인했으며, 전 분야 신규성이나 장난감 밖 성능은 이 검수 범위를 벗어난다.

