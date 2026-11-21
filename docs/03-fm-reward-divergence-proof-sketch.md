# 증명 스케치 — Flow-Matching 잔차 리워드와 분포매칭

목표: "FM 잔차 리워드를 최대화하면 정책분포 $p_\pi$ 가 데이터분포 $p_{data}$ 로 수렴하는가?"
결론(정직): **그냥 리워드 최대화로는 붕괴한다. 최대엔트로피(soft) RL과 결합해야 분포매칭이 성립한다.**

## 1. 재료
- 조건부 FM: 최적 속도장 $v^*(y,t)=\mathbb E[x_1-x_0\mid x_t=y]$, $x_t=(1-t)x_0+tx_1$, $x_0\sim\mathcal N(0,I)$, $x_1\sim p_{data}$.
- CNF 로그가능도(정확): $\log p_1^\theta(x)=\log p_0(x_0)-\int_0^1\nabla\!\cdot v^*(x_t,t)\,dt$.
- 잔차 에너지(대리): $\mathcal R(x)=\mathbb E_{t,x_0}\|v^*(x_t,t)-(x-x_0)\|^2$, 리워드 $r(x)=-\mathcal R(x)$.

## 2. 핵심 함정 — 리워드가 $q$에 선형이면 붕괴
정책유도분포 $q=p_\pi$ 에 대한 목적:
$$J(q)=\mathbb E_{x\sim q}[r(x)]=\int q(x)\,r(x)\,dx.$$
이것은 $q$에 **선형**. 제약 없는 최대화의 해는 $r$ 최댓값 지점의 **점질량(델타)** → **모드붕괴/리워드해킹**. (AMP 판별자 리워드도 동일: 고정 판별자면 선형 → 붕괴. 그래서 AMP는 판별자를 **적대적 공적응**시켜 이동표적으로 붕괴를 막음.)

## 3. 해결 — 최대엔트로피(soft) RL
엔트로피 정규화 목적(온도 $\alpha$):
$$\max_q\ \mathbb E_{x\sim q}[r(x)] + \alpha\,\mathcal H(q).$$
변분 최적해(깁스):
$$\boxed{\,q^*(x)\ \propto\ \exp\!\big(r(x)/\alpha\big)\,}$$
여기에 **정확 리워드 $r(x)=\log p_1^\theta(x)\approx\log p_{data}(x)$** 를 넣고 $\alpha=1$:
$$q^*(x)\propto\exp(\log p_{data}(x))=p_{data}(x).$$
→ **분포매칭 성립.** $\alpha\to0$ 이면 점질량(붕괴), $\alpha=1$ 이면 $p_{data}$, 그 사이는 $p_{data}^{1/\alpha}$.
(이는 에너지기반/GFlowNet/soft-RL 의 고정점 결과와 동일.)

## 4. 잔차 대리의 gap
$-\mathcal R(x)$ 는 정확한 $\log p$ 가 아니라 **대리(ELBO/경로 KL 유형)**:
- CFM 목적은 생성분포와 타깃 사이 KL의 **상한**을 준다(interpolant/stochastic-interpolants 결과).
- 고정 $v^*$ 에 대해 $\mathbb E_{x\sim q}\mathcal R(x)$ 는 "$x$ 에서 $v^*$ 가 자기일관적인가"를 측정 = **$p_{data}$ 의 모드/지지에서 작고 밖에서 큼** → in-distribution 점수로 유효.
- 열린 과제: $\mathcal R$ 과 $-\log p_{data}$ 의 차이(경로상 divergence·가중치)를 닫힌형으로 특성화 → 그래야 "리워드 타당성(soft 고정점 = $p_{data}$)"이 근사적으로 보장.

## 5. 설계 함의 (바로 쓸 수 있음)
1. **PPO 리워드최대화 ✗ → soft/max-ent RL(SAC류) ✓.** 안 그러면 모드붕괴.
2. 온도 $\alpha$ 가 "분포매칭↔성능" 손잡이. (파쿠르처럼 성공이 중요하면 $r=r_{task}+\beta\,r^{style}$ + 엔트로피.)
3. 위상조건 $v^*(\cdot;\phi)$ → soft 고정점 $q^*(x|\phi)\propto p_{data}(x|\phi)$ = **위상별 분포매칭** = 리듬.

## 6. 검증 (src/toy_fm_vs_amp.py)
- (C1) 정확 $\log p_{data}$(3-모드 GMM)로 **리워드최대화=1모드 붕괴, $\alpha=1$ soft=3모드 복원** 을 모드커버리지로 실측.
- (C2) **비모수 FM 잔차 에너지가 $\log p_{data}$ 와 높은 순위상관**(유효 점수) — 고정(안정) 모델로 1회 산출.
- (C3) **AMP 적대 에너지는 수렴 시 비슷하나 과정 분산(불안정)이 큼** — FM 대비 안정성 실측.
