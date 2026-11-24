# 월드 모델 · 월드 액션 모델(WAM) — 2026-11 기준 SOTA와 구현 수준 학습 절차

근거는 arXiv TeX 원문 32편이다. ✔ 표시는 원문에서 직접 다시 대조한 수치다. 벤치마크 수치는 모두 저자의 자체 보고이고, 같은 조건에서 서로 비교한 것은 아니다.

---

## 0. 결론과 정정

### 계열 지도

| 계열 | 무엇을 예측하나 | 대표 (2026) | 행동은 어떻게 나오나 |
|---|---|---|---|
| 생성형 WM | 다음 **픽셀/VAE 잠재** 프레임 | Cosmos 3 (2606.02800), Cosmos-Predict2.5 (2511.00062), Matrix-Game 2.0/3.0, Self-Forcing, LingBot-World, Genie 3(구조 비공개) | 행동을 **입력**으로 받아 다음 장면을 생성(시뮬레이터) |
| 잠재 WM (JEPA) | 다음 **인코더 특징** (픽셀 복원 없음) | V-JEPA 2 / 2-AC (2506.09985), V-JEPA 2.1, DINO-WM (2411.04983), LeWM (2603.19312), JEPA-WM 설계연구 (2512.24497) | 정책 없이 **CEM-MPC 계획**으로 행동을 탐색 |
| WAM | **미래 영상 + 행동**을 한 모델에서 같이 생성 | DreamZero (2602.15922), Cosmos Policy (2601.16163), LingBot-VA (2601.21998), Fast-WAM (2603.16666), GR00T 2(백서) | 행동을 **출력**으로 직접 디노이즈 |
| VLA (대비) | 행동 (미래 영상은 예측하지 않음) | π0.5, GR00T N1.x | VLM 백본 + 행동 전문가(flow) |

### 붙여넣은 정리의 정정
1. **V-JEPA 2-AC는 역동역학 모델(IDM)이 아니다.** 행동을 조건으로 받는 블록 인과 예측기 `P(z_t, a_t, s_t) → z_{t+1}`(순방향 동역학)다. 행동은 학습된 정책이 아니라 CEM으로 목표 이미지 임베딩과의 L1 거리를 최소화해서 고른다. ✔
2. **GR00T N1/N1.5/N1.6은 VLA다.** 원문에 "Vision-Language-Action (VLA) model with a dual-system"이라고 되어 있다. ✔ WAM으로 분류되는 것은 GR00T 2(백서)다.
3. **Genie 3는 구조·학습법이 공개되지 않았다.** 구현 참고로는 Genie 1(2402.15391)이나 공개 모델(Matrix-Game, LingBot-World)을 써야 한다.
4. **Cosmos 3는 존재한다.** 2606.02800, reasoner와 generator 두 타워로 된 MoT 옴니모달 구조이고, 행동을 하나의 모달리티로 다룬다. 행동 데이터는 8.4M 에피소드, 61.3K시간이다. ✔

---

## 1. 생성형 월드 모델

현재 구현 표준은 **3D-VAE 잠재 + flow-matching DiT**다. 먼저 양방향 사전학습을 하고, 행동 조건을 붙인 뒤, 인과(자기회귀) 모델로 바꾸고, Self-Forcing 계열 증류로 실시간화한다.

### 1.1 모델 구성

```
비디오 x[T,3,H,W] ──3D causal VAE (Wan2.1/2.2, 4×8×8 압축, 16~48ch)──▶ z[T',C,h,w]
           │                                    (첫 프레임은 단독 인코딩 → 이미지/영상 공용)
           ▼
     patchify 1×2×2 → 토큰열 [T'·h/2·w/2, D]
           ▼
  DiT 블록 ×N : self-attn(3D RoPE) → cross-attn(텍스트 T5/Qwen-VL 임베딩) → MLP
               adaLN(timestep 임베딩 + 행동 임베딩)  ← 행동 주입 지점
           ▼
     unpatchify → 속도장 v̂ (flow matching)
```

**행동 주입 방식(중요)**
Cosmos-Predict2.5의 Bridge 실험(행동 조건 모델 비교)에서 행동 임베딩을 **timestep 임베딩에 더하는(adaLN)** 방식이 가장 좋았다. ✔

| 방식 | PSNR | FVD |
|---|---|---|
| timestep 임베딩에 더하기 | 24.95 | 146 |
| cross-attn | 24.41 | 159 |
| 채널 concat | 23.11 | 267 |

구현은 다음과 같다.
```python
a_emb = MLP(action_chunk.flatten(-2))          # 잠재 프레임 1개당 행동 4개(시간압축 4배) → 묶어서 임베딩
cond  = t_emb(t)[:, None] + a_emb              # [B, T', D]  프레임별 조건
# 각 DiT 블록: shift, scale, gate = adaLN_mlp(cond)  (프레임 토큰마다 자기 프레임 cond)
```

Matrix-Game 2.0은 게임 입력을 다음처럼 넣는다.
- 키보드: 이산 원-핫을 cross-attn으로 넣는다.
- 마우스: 연속값을 프레임 특징에 concat한 뒤 self-attn으로 넣는다.
- 텍스트 조건은 빼서 순수 행동 조건으로 만든다.

Genie 1식으로 **행동 라벨이 없는 영상**을 쓸 때는 잠재 행동 모델(LAM)을 둔다.
- 인코더 `(x_{1:t}, x_{t+1}) → ã_t` 뒤에 VQ(|A|=8 ✔)를 둔다.
- 디코더 `(x_{1:t}, ã_t) → x̂_{t+1}`를 둔다.
- 손실은 재구성이다. 코드북이 작아야 프레임 간 "변화"만 담긴다.
- 학습 후에는 LAM 인코더를 버리고 코드를 사용자 입력으로 쓴다.

### 1.2 학습 절차

**단계 0. 토크나이저**
보통 Wan VAE를 동결해서 그대로 쓴다. 직접 학습한다면 다음과 같다.
- 손실: `L1 + LPIPS + KL(β≈1e-6)`, 이후 GAN을 추가한다.
- 인과 3D conv를 쓰고 첫 프레임은 독립 인코딩한다.

**단계 1. 양방향 텍스트/이미지→영상 사전학습 (rectified flow)**
```python
z1 = vae.encode(video); z0 = randn_like(z1)
t  = sigmoid(randn(B) * 1.0 + 0.0)            # logit-normal
t  = shift*t / (1 + (shift-1)*t)              # 해상도 클수록 shift↑ (Wan/SD3식, 예: 3~5)
zt = (1-t)*z0 + t*z1
loss = ((model(zt, t, text) - (z1 - z0))**2).mean()
```
커리큘럼은 저해상도 이미지에서 시작해 저해상도 이미지+영상, 고해상도로 올린다. 이미지와 영상의 비율을 따로 조정한다.

**단계 2. 행동 조건 미세조정**
- 데이터 쌍은 (영상, 프레임별 행동)이다. 로봇이면 DROID/Bridge의 EE 델타+그리퍼, 게임이면 키/마우스다.
- 조건 프레임은 1~몇 장의 **깨끗한 잠재**로 주고, 나머지를 생성한다.
- 문맥 프레임에는 작은 노이즈를 섞는다(GameNGen식 노이즈 증강). 자기회귀 누적 오차를 견디게 하기 위해서다.

**단계 3. 인과(자기회귀) 전환**
다음 두 방식 중 하나를 쓴다.

- **(a) Diffusion Forcing**
  - 프레임(청크)마다 **서로 다른 독립 노이즈 레벨** `t_k ~ U`를 뽑는다.
  - attention을 **블록 인과 마스크**로 바꾼다.
  - 손실은 위와 같고, `t`만 프레임별 벡터가 된다.

- **(b) Causal teacher forcing**
  - 과거 청크는 깨끗하게 두고 현재 청크만 노이즈를 넣는다.

```python
def block_causal_mask(n_chunks, tok_per_chunk):
    c = torch.arange(n_chunks).repeat_interleave(tok_per_chunk)
    return c[:, None] >= c[None, :]           # 자기 청크까지 양방향, 미래 금지
```

**단계 4. Self-Forcing 증류 (실시간화)**
teacher forcing으로 학습한 모델은 추론 때 자기 출력이 문맥으로 들어가서 노출 편향이 생긴다. Self-Forcing은 **학습 중에도 자기 출력으로 KV 캐시 롤아웃**을 하고, 영상 전체에 분포 정합 손실(DMD)을 건다.
```python
# G: 인과 few-step 생성기(초기값=단계3 모델), S_real: 동결 양방향 교사, S_fake: 학습되는 critic
kv = []; frames = []
for k in range(n_chunks):                     # 청크=잠재 3프레임
    x = randn(chunk_shape)
    s = randint(0, n_steps)                   # 무작위 하나의 디노이즈 스텝에서만 grad 유지
    for i, t in enumerate(timesteps):         # 예: 4스텝 [1000,750,500,250]
        with torch.set_grad_enabled(i == s):
            x0 = G(x, t, kv_cache=kv, action=a[k])
        if i == s: break
        x = renoise(x0, timesteps[i+1])
    frames.append(x0)
    kv = G.update_cache(x0.detach(), kv)      # 깨끗한 출력으로 캐시 갱신(t=0 재통과)
video = cat(frames)
# DMD: grad_G ≈ (S_fake(noised(video)) - S_real(noised(video)))
t = rand(); xn = add_noise(video, t)
g = (S_fake(xn, t) - S_real(xn, t)).detach()
loss_G = 0.5 * ((video - (video - g).detach())**2).mean()
# critic: S_fake를 G 출력분포에 대해 일반 flow/denoise 손실로 학습 (G 1스텝당 critic 여러 스텝)
```
보고된 속도는 다음과 같다.
- Self-Forcing: 단일 H100에서 17.0 FPS ✔
- Matrix-Game 2.0: 1.8B 파라미터, 25.15 FPS ✔
- Matrix-Game 2.0의 KV 창은 6이 9보다 나았다(긴 창은 오차 누적).

**단계 5. 장기 일관성(메모리)**
- 방문한 시점의 프레임을 메모리로 꺼내 문맥에 넣는다. Matrix-Game 3.0은 카메라 frustum이 겹치는 과거 프레임을 검색한다.
- 오차 버퍼를 쓴다. 모델이 실제로 만든 오차를 모아 두었다가 문맥에 주입하고, 거기서 회복하는 법을 학습시킨다.

### 1.3 추론
```python
kv = encode_context(first_frames)
while running:
    a = read_user_or_policy_action()
    x = randn(chunk); 
    for t in timesteps_4: x = step(G(x, t, kv, a), t)
    show(vae.decode(x)); kv = G.update_cache(x, kv)[-window:]
```

---

## 2. 잠재 월드 모델 (JEPA 계열)

픽셀을 복원하지 않는다. **인코더 특징 공간에서 다음 상태를 예측**하고, 계획은 그 공간에서 한다.

### 2.1 V-JEPA 2: 자기지도 사전학습 (행동 없음)

**구성**
- 인코더 `E_θ`: ViT-g(1B)를 쓰고 tubelet은 2×16×16, 3D RoPE를 적용한다.
- 타깃 인코더 `E_ξ`: `E_θ`의 EMA다. momentum은 0.99925 ✔
- 예측기 `P_φ`: 인코더보다 훨씬 좁은 ViT다(크기는 원문 표 확인, 여기서 미대조). 입력은 보이는 토큰 특징과, 가려진 위치의 mask 토큰(+위치 임베딩)이다.

**마스킹 (multiblock, 시간축 전체를 관통하는 tube)**
- 짧은 범위: 블록 8개, 각 공간 비율 0.15
- 긴 범위: 블록 2개, 각 공간 비율 0.70
- 합쳐서 약 90%를 가린다. 예측기는 가려진 위치 전부를 예측한다.

**손실**
```python
ctx_idx, tgt_idx = sample_multiblock_mask()
with torch.no_grad():
    h = E_xi(video)                           # 전체 영상
    h = F.layer_norm(h, h.shape[-1:])[:, tgt_idx]
z = E_theta(video, keep=ctx_idx)              # 보이는 토큰만
pred = P_phi(z, ctx_idx, tgt_idx)
loss = (pred - h).abs().mean()                # L1
loss.backward(); opt.step()
ema_update(E_xi, E_theta, m=0.99925)
```

**하이퍼파라미터 ✔(일부)**
- 배치 3072, 최대 LR 5.25e-4, warmup → 상수 → cooldown.
- 점진적 해상도 상향: 16프레임 256px로 학습한 뒤, cooldown에서 64프레임 384px로 늘린다.
- 데이터는 VideoMix22M(약 1M시간 영상)과 이미지다.

**대안: EMA 없이 붕괴 막기 (LeJEPA / LeWM)**
타깃은 stop-grad를 쓰지 않고 같은 인코더를 쓴다. 대신 임베딩 분포가 등방 가우시안이 되도록 SIGReg를 건다.
```python
def sigreg(z, M=1024, n_t=17):                # z:[N,D]
    u = F.normalize(randn(D, M), dim=0)       # 무작위 방향 M개
    p = z @ u                                 # 1D 투영 [N,M]
    t = linspace(-5, 5, n_t)
    ecf = torch.exp(1j * p[..., None] * t).mean(0)       # 경험적 특성함수
    return (((ecf - torch.exp(-t**2/2))).abs()**2 * torch.exp(-t**2/2)).mean()   # Epps–Pulley
loss = pred_loss + 0.1 * sigreg(z)            # LeWM λ=0.1, M=1024 ✔
```
- LeWM은 15M 파라미터이고 projector에 BatchNorm을 쓴다.
- 계획 속도는 DINO-WM보다 약 48배 빠르다고 보고했다.

### 2.2 V-JEPA 2-AC: 행동 조건 예측기 (사후학습)

**구성**
- 인코더는 **동결**한다. 프레임별로 `z_k = E(x_k) ∈ R^{256×1408}`다(16×16 패치).
- 예측기는 새로 만든다. 300M 파라미터, 24층, 16헤드, 은닉 1024, GELU ✔
- 매 스텝 토큰은 `[a_k(7: EE 델타 + 그리퍼), s_k(7: EE 상태), z_k(256 패치)]`이다. a와 s는 선형층으로 사상한다.
- attention은 **블록 인과**다. 시점 k의 토큰은 k 이하의 모든 토큰만 본다. 3D RoPE를 쓴다.

**손실: teacher forcing + 롤아웃**
```python
z = [E(x_k).detach() for k in range(T+1)]     # T=15 정도, 4fps 클립
# (1) teacher forcing: 정답 문맥으로 모든 다음 스텝 동시 예측
zhat = P(z[:-1], a[:-1], s[:-1])              # 블록인과 → zhat[k] ≈ z[k+1]
L_tf = (zhat - z[1:]).abs().mean()
# (2) 롤아웃: 첫 프레임에서 자기 예측을 문맥으로 2스텝
cur = [z[0]]
for k in range(2):                             # rollout T=2 ✔(요약)
    cur.append(P(cur, a[:k+1], s[:k+1])[-1])
L_roll = (cur[-1] - z[2]).abs().mean()
loss = L_tf + L_roll
```
데이터는 DROID 중 62시간 미만(행동 라벨, 보상과 과제 라벨 없음)이다. ✔

**계획: CEM-MPC (목표 이미지 조건)**
```python
z_goal = E(goal_img); z_now = E(obs)
mu, sigma = zeros(H, 7), ones(H, 7) * init_std
for it in range(10):
    A = mu + sigma * randn(800, H, 7)          # 샘플 800 ✔, H=1 ✔
    A = project_l1_ball(A[..., :3], r=0.075)   # 위치 델타를 L1 공 반경 0.075로 ✔
    cost = (rollout(P, z_now, s_now, A)[-1] - z_goal).abs().mean((-1, -2))
    elite = A[cost.topk(10, largest=False).indices]    # top-10
    mu, sigma = elite.mean(0), elite.std(0)
execute(mu[0]); replan()                       # RTX 4090에서 행동 하나당 약 16초 ✔
```
V-JEPA 2.1은 다음을 추가했다.
- 가려지지 않은 문맥 토큰에도 거는 dense 손실
- 중간층 deep supervision
- 계획 horizon을 8로 늘림(grasp 80%)

### 2.3 DINO-WM (가장 작은 재현 출발점)
- 인코더는 **동결 DINOv2-S/14** 패치 특징(224px 입력, 패치 256개 × 384차원)이다.
- 예측기는 약 19M ViT이고 프레임 단위 인과 마스크를 쓴다. 문맥은 H=1~3 프레임이다.
- 행동(그리고 proprio)은 10차원으로 MLP 사상한 뒤, **모든 패치 토큰에 concat**한다.
- 손실: `|| P(z_{t-H:t}, a_{t-H:t}) − z_{t+1} ||²`, teacher forcing이다.
- 디코더는 시각화용으로 따로 학습하고, 예측기로는 gradient를 보내지 않는다.
- 계획: CEM 샘플 100 × 최적화 10스텝 ✔, MPC로 다시 계획한다.
- 설계연구(2512.24497)의 권장 사항
  - 행동은 AdaLN으로 주입한다.
  - 비용은 L2를 쓴다.
  - 계획기는 CEM이 경사 기반보다 안정적이다.

### 2.4 Dreamer 4 (참고: 생성형 + RL)
- 잠재는 픽셀 토크나이저에서 얻는다.
- 동역학은 shortcut forcing으로 학습해 소수 스텝으로 샘플링한다.
- 상상 롤아웃 안에서 RL(정책+가치 헤드)을 한다. 오프라인 데이터만으로 Minecraft 다이아몬드를 달성했다고 보고했다.
- JEPA와 다른 점: 디코더와 픽셀 재구성이 있다.

---

## 3. 월드 액션 모델 (WAM)

공통 아이디어는 **사전학습된 영상 확산 모델을 백본으로 쓰고, 미래 영상과 행동을 같은 디노이즈 과정에서 같이 생성**하는 것이다. 영상 사전학습이 물리와 동역학 사전지식을 제공한다.

### 3.1 공통 뼈대
```
관측 이미지들 ─VAE─▶ 깨끗한 문맥 잠재
언어 ─텍스트 인코더─▶ cross-attn
[미래 영상 잠재 청크(노이즈)] + [행동 청크(노이즈)] ─▶ 공동 DiT (블록 인과) ─▶ v̂_video, v̂_action
```
모델마다 다른 점은 다음과 같다.

| 모델 | 백본 | 행동 표현 | 특징 |
|---|---|---|---|
| DreamZero | Wan2.1-I2V-14B | 행동 토큰(전용 enc/dec MLP) | 청크 단위 자기회귀, 모달리티별 timestep, Flash |
| Cosmos Policy | Cosmos-Predict2-2B | **행동·proprio·가치를 잠재 프레임 모양으로 변형해 끼워 넣음** | 구조 변경 없음, 하나의 모델이 정책+WM+가치 |
| LingBot-VA | Wan2.2-5B + 350M 행동 스트림 (총 5.3B ✔) | 별도 행동 트랜스포머(MoT, 공유 attention) | 영상은 부분 디노이즈만 (s=0.6) |
| Fast-WAM | 영상 DiT | 행동 헤드 | **학습 때만** 영상 공동 예측, 추론 때 영상 생성 생략 |

### 3.2 학습 절차

**단계 0. 백본 준비**
공개 영상 확산 모델(Wan2.1/2.2, Cosmos-Predict2) 가중치를 불러온다.

**단계 1. 데이터**
- (관측 영상, 언어 지시, 행동 시퀀스, proprio)를 묶는다.
- DreamZero는 여러 로봇의 **이질적 데이터**를 쓴다. 반복 시연이 아닌 다양한 행동 데이터에서 일반화가 왔다고 주장한다.
- 행동은 데이터셋별로 정규화(분위수 → [-1,1])하고, 로봇 형태별 차원은 패딩과 마스크로 맞춘다.

**단계 2. 공동 flow matching (모달리티별 timestep)**
```python
zv1 = vae.encode(future_video)          # [B, K, C, h, w]  K=청크 수
za1 = action_chunks                     # [B, K, n_act, d_act]
ctx = vae.encode(obs_history)           # 깨끗한 문맥
# DreamZero 기본: 영상·행동 같은 t, 청크마다 다른 t (diffusion forcing식)
t_v = rand(B, K); t_a = t_v.clone()
# UWM식 대안: t_v, t_a 독립 → 한 모델로 정책/역동역학/순동역학/영상생성 모두
zv0, za0 = randn_like(zv1), randn_like(za1)
zvt = interp(zv0, zv1, t_v); zat = interp(za0, za1, t_a)
vv, va = model(ctx, zvt, zat, t_v, t_a, text, attn_mask=block_causal)
loss = mse(vv, zv1 - zv0) + lam_a * mse(va, za1 - za0)
```
- 블록 인과는 다음과 같다. 청크 k의 영상과 행동 토큰은 문맥과 청크 ≤k만 보고, 같은 청크 안의 영상↔행동은 양방향이다.
- DreamZero는 청크 K=2를 쓴다(요약).

**단계 3. (DreamZero-Flash) 추론 가속용 timestep 분리** ✔
```python
eta = Beta(7, 1).sample((B, K)); t_v = 1 - eta    # 영상 t를 한쪽 끝에 몰고
t_a = rand(B, K)                                   # 행동 t는 균등
```
목적은 영상이 덜 디노이즈된 상태에서도 행동을 정확히 내도록 학습해서, 추론 때 영상 스텝을 줄이는 것이다. 위 해석과 t의 부호 규약은 원문 코드로 확인하고 쓸 것.

보고된 성능은 다음과 같다. ✔
- 지연 150ms, 행동 청크 약 7Hz
- 새 과제/환경 평균 과제 진척도 62.2%, 최고 VLA의 2배 이상
- 표 일부: DreamZero 52% → Flash 74%, 둘 다 150ms

**Cosmos Policy 방식 (구조를 건드리지 않는 버전)**
```python
# 행동 청크(예: 16×7)를 잠재 프레임 모양 [C,h,w]으로 정규화·reshape해 영상 잠재열 사이에 삽입
seq = [cur_obs_lat, proprio_lat, ACTION_lat, future_proprio_lat, future_obs_lat, VALUE_lat]
# 조건으로 줄 칸은 깨끗하게, 예측할 칸만 노이즈 → 같은 비디오 확산 손실
```
- 하나의 모델로 정책(행동 생성), 월드 모델(미래 상태), 가치 V(s')를 모두 학습한다.
- 추론 때는 후보 행동 여러 개를 뽑아 예측 가치로 고르는 계획이 가능하다.
- LIBERO 평균 98.5%, RoboCasa 67.1% ✔

**Fast-WAM의 교훈 (설계 결정용 절제)** ✔
- 추론 때 미래 영상을 생성하지 않아도 LIBERO 97.6이다.
- 학습에서 영상 공동 예측을 빼면 93.5로 떨어진다.
- 결론: WAM의 이득은 대부분 **"학습 목표로서의 영상 예측"**에서 온다. 테스트 시점의 상상은 그보다 덜 중요하다.
- 지연은 190ms다.

### 3.3 추론 (KV 캐시 + 부분 디노이즈)
```python
kv = model.prefill(obs_lat, text)
while True:
    zv, za = randn(chunk_v), randn(chunk_a)
    for t in steps:                               # 행동은 끝까지, 영상은 일부만(LingBot-VA s=0.6)
        vv, va = model(zv, za, t, kv)
        za = step(za, va, t)
        if t > s_video_stop: zv = step(zv, vv, t)
    robot.execute(decode_action(za))              # 실행 중 다음 청크 비동기 생성
    kv = model.append(vae.encode(new_obs), kv)    # 예측 영상은 버리고 실제 관측으로 캐시 갱신
```
마지막 줄이 핵심이다. 예측 프레임을 캐시에 남기지 않고 **실제 관측으로 교체**해야 오차가 누적되지 않는다.

---

## 4. VLA와의 차이 (구현 관점)

| | VLA (π0.5, GR00T N1) | WAM |
|---|---|---|
| 백본 | VLM (PaliGemma, Eagle-2) | 영상 확산 모델 |
| 사전지식 | 웹 이미지-텍스트 의미 | 영상의 물리·동역학 |
| 행동 헤드 | flow matching 행동 전문가 (π0: 300M) / DiT (GR00T N1) | 공동 디노이즈 |
| 미래 예측 | 없음 | 미래 영상 (학습 또는 추론) |
| 지연 | GR00T N1 63.9ms (16행동 청크, L40 bf16 ✔) | 150~190ms 수준 |

π0.5의 학습 순서는 다음과 같다.
1. VLM에 FAST 이산 행동 토큰으로 next-token 사전학습을 한다.
2. 행동 전문가를 붙여 flow 손실로 학습한다(가중치 α=10).
3. 추론은 연속 flow 행동으로 한다.

---

## 5. 축소 재현 제안 (A6000×2 / 5090)

| 트랙 | 무엇을 | 예산 | 양성 대조 | 1차 가설 |
|---|---|---|---|---|
| L (잠재 WM) | DINO-WM: 동결 DINOv2-S + 19M 예측기, PushT/PointMaze | 5090 1장, 1일 이내 | 무작위 행동 CEM / 정답 동역학 오라클 상한 | rollout 손실 추가 ≥ teacher forcing 단독 (V-JEPA 2-AC 주장 재현), SIGReg ≈ EMA |
| G (생성 WM) | 소형 DiT(~100M)를 Wan VAE 잠재로, 행동 조건 영상(게임/Bridge 저해상도) | A6000×2, 수일 | 행동 무시 모델(조건 섞기) | adaLN 행동 주입 > concat (Cosmos 2.5 순서 재현), Self-Forcing이 장기 FVD 개선 |
| W (WAM) | 같은 DiT에 행동 스트림 추가, LIBERO 일부 과제 | A6000×2 | 영상 공동학습 제거 팔(Fast-WAM 93.5 대조) | 영상 공동학습 이득이 시드 신뢰구간 밖인가 |

판정은 기존 12규칙을 따른다(사전등록, 시드 ≥3, 신뢰구간, 튜닝된 비교군, 오라클 금지). 이 문서는 학습법 정리이고 성능 판정이 아니다.

---

## 출처 (arXiv)
- 생성형: Genie 2402.15391 · Cosmos 2501.03575 · Cosmos-Predict2.5 2511.00062 · Cosmos 3 2606.02800 · Self-Forcing 2506.08009 · Matrix-Game 2.0 2508.13009 · Matrix-Game 3.0 2604.08995 · LingBot-World 2601.20540 · GameNGen 2408.14837 · DIAMOND 2405.12399 · Diffusion Forcing 2407.01392
- 잠재: V-JEPA 2 2506.09985 · DINO-WM 2411.04983 · JEPA-WM 설계 2512.24497 · LeJEPA 2511.08544 · LeWM 2603.19312 · Dreamer 4 2509.24527 · NWM 2412.03572 · V-JEPA 2.1 / RoboJEPA (2610.10515 외)
- WAM: DreamZero 2602.15922 · Cosmos Policy 2601.16163 · LingBot-VA 2601.21998 · Fast-WAM 2603.16666 · UWM 2504.02792 · UVA 2503.00200 · GE-Act 2508.05635 · VPP 2412.14803
- VLA: GR00T N1 2503.14734 · π0 2410.24164 · π0.5 2504.16054
