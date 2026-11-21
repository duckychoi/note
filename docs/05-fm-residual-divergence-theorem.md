# 정리 — Flow-Matching 잔차가 분포 divergence를 상한한다

목표: "FM 잔차 리워드 $-\mathcal R(q)$ 를 (max-ent로) 최대화하면 $q\to p_{data}$" 를 **엄밀히** 뒷받침.
핵심 결과: **FM 잔차는 바이어스-분산 분해되고, 그 바이어스항이 $\mathrm{KL}(q\|p_{data})$(또는 $W_2^2$)을 상한한다.**

## 0. 세팅 (가정)
- 기저 $p_0=\mathcal N(0,I)$, 데이터 $p_1=p_{data}$. 선형 보간 $x_t=(1-t)x_0+t x_1$, 독립결합 $x_0\sim p_0,\ x_1\sim p_1$.
- 주변속도장(=CFM 최적장) $u_p(x,t)=\mathbb E[x_1-x_0\mid x_t=x]$ — 연속방정식 $\partial_t p_t+\nabla\!\cdot(p_t u_p)=0$ 으로 $p_t$ 생성.
- 학습장 $v^*=u_{p_{data}}$ (실현가능/잘학습 가정).
- 테스트(정책) 분포 $q$ 와 그 최적장 $u_q(x,t)=\mathbb E_q[x_1-x_0\mid x_t=x]$ (단, $x_1\sim q$).
- (A1) 작은 노이즈 $\sigma>0$ 의 확률보간(stochastic interpolant) SDE 사용. (A2) $v^*$ 가 $x$에 대해 Lipschitz. (A3) 2차 모멘트 유한.

## 1. 잔차의 바이어스-분산 분해 (정확)
정책 샘플 $x\sim q$ 에 대한 잔차 에너지:
$$\mathcal R(q)=\mathbb E_{t\sim U[0,1]}\,\mathbb E_{x\sim q,\,x_0\sim p_0}\big\|v^*(x_t,t)-(x-x_0)\big\|^2.$$
$y=x_t$ 고정 조건부에서 $u_q(y,t)=\mathbb E_q[(x-x_0)\mid x_t=y]$ 를 더하고 빼면(조건부평균 직교성, 교차항 0):
$$\boxed{\ \mathcal R(q)=\underbrace{\mathbb E_t\,\mathbb E_{y\sim p_t^q}\|v^*(y,t)-u_q(y,t)\|^2}_{\textstyle B(q)\ (\text{바이어스: 장 불일치})}+\underbrace{\mathbb E_t\,\mathbb E\big[\mathrm{Var}_q((x-x_0)\mid x_t)\big]}_{\textstyle C(q)\ (\text{기약 결합분산})}\ }$$
- $B(q)\ge0$, **$B(q)=0\iff u_q\equiv v^*$ (지지 위) $\iff q=p_{data}$** (장이 끝점주변을 결정). → **엄밀 proper discrepancy**.
- $C(q)$: 결합의 기약 잡음(리워드 상수성 offset). 정책경사엔 $B(q)$ 의 $q$-의존만 유효.

## 2. 주정리 — divergence 상한
**정리.** (A1–A3) 하에서
$$\mathrm{KL}(q\,\|\,p_{data})\ \le\ \frac{1}{2\sigma^2}\int_0^1\mathbb E_{y\sim \rho_t}\|v^*(y,t)-u_q(y,t)\|^2\,dt\ \asymp\ \frac{1}{2\sigma^2}\,B(q).$$
*증명 스케치.* 확률보간 SDE $dX=v^*(X,t)dt+\sigma dW$ 는 $p_0\!\to\!p_{data}$, $dX=u_q(X,t)dt+\sigma dW$ 는 $p_0\!\to\!q$ 를 수송(초기 동일). **Girsanov**: 두 경로측도의 KL $=\frac{1}{2\sigma^2}\int_0^1\mathbb E\|v^*-u_q\|^2dt$. **데이터처리 부등식**으로 끝점 주변의 KL이 경로 KL 이하. $\square$
- 결정론 극한 $\sigma\!\to\!0$: KL 상한은 발산 가능 → 대신 **연속방정식 안정성(Grönwall)** 으로 $W_2^2(q,p_{data})\lesssim \int_0^1\|v^*-u_q\|_{L^2(\rho_t)}^2dt \asymp B(q)$ (Lipschitz $v^*$). 즉 $\sigma{>}0$이면 KL, $\sigma{=}0$이면 $W_2$ 를 상한.

**따름정리 (리워드 타당성).** $\mathcal R(q)=B(q)+C(q)$ 이므로, $C(q)$ 를 상수화(또는 분산축소)하면 FM 잔차 리워드 $-\mathcal R$ 최대화는 $-\tfrac12\mathrm{KL}$ 의 하한을 밀어올림. max-ent(온도 $\alpha{=}1$) 고정점 $q^*\propto\exp(-\mathcal R/\alpha)$ 는 $B(q)$ 가 지배할 때 $p_{data}$ 로 수렴.

## 3. $-\log p$ 와의 gap (두 번째 질문)
- 정확 CNF NLL: $-\log p_1^\theta(x)=-\log p_0(x_0)+\int_0^1\nabla\!\cdot v^*(x_t,t)dt$ (정확 궤적). 이는 **발산(divergence) 적분** = 잔차와 다른 양.
- 관계: FM 잔차는 NLL이 아니라 **운동에너지($L^2$) 목적**. §2가 보이듯 **$L^2$ 장오차가 KL을 상한** → 잔차는 NLL의 **상계 대리**. gap = $C(q)$(기약분산) + 발산항과 운동항의 차. 점수매칭→KL 상한(Song 등)·stochastic interpolants(Albergo–Vanden-Eijnden)와 동형.

## 4. 남은 엄밀화(정직한 구멍)
1. **$C(q)$ 의 $q$-의존**: 독립결합에선 큼. **rectified/OT 결합**으로 축소 시도.
   - ⚠️ **검증결과(src/toy_rectified_coupling.py) — 순진한 OT는 역효과**: OT결합이 데이터 경로는 직선화(CFM잔차 4.36→0.82)하나, **eval에서 $x_0\sim\mathcal N$ 독립샘플**이면 학습·평가 **결합 불일치**로 $v^*$ 가 안 맞아 base R 3.19→4.51↑, corr(R,KL) 0.77→**0.25↓**.
   - **교훈**: $C$ 축소는 **train/eval 결합 일관성**이 필수. "OT로 학습"만으론 안 되고, eval도 일관된(결정론/매칭) 결합을 써야 함. 대안: **가능도(CNF 발산적분)·점수매칭 기반 리워드**는 $C$ floor 자체가 없음 → 더 유망.
2. **주변 불일치**: §2의 $L^2(\rho_t)$ 는 한 시스템의 $\rho_t$, $B(q)$ 는 $p_t^q$ — 변수변환(밀도비 유계) 또는 단측안정으로 봉합.
3. **실현가능성**: $v^*=u_{p_{data}}$ 근사오차(유한데이터·NN)의 전파.
→ 특히 1의 교훈(결합 일관성 or 가능도형 리워드)이 논문의 실질 설계 포인트. 셋을 닫으면 완전 정리.

## 5. 수치검증 (src/toy_fm_divergence.py)
GMM 데이터 + 비모수 $v^*$. 교란분포족 $q_\lambda$ 에 대해 **FM 잔차 $\mathcal R(q_\lambda)$ 가 $\mathrm{KL}(q_\lambda\|p_{data})$ 와 단조(강양상관)** 임을 확인 → 정리 방향(잔차가 divergence 상한·단조) 실증.
