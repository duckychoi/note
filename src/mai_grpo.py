"""
MAI-Thinking-1 RL 레시피 — 프레임워크 독립 numpy 레퍼런스 구현
=================================================================
출처: Microsoft AI, "MAI-Thinking-1: Building a Hill-Climbing Machine" §3.1 (식 4–12, §3.1.5 하이퍼파라미터),
      §3.4.1(보상 결합), §3.6(Rocket), 그리고 "RL infrastructure at Microsoft AI"(Rocket 글).
목적: 손실·제어기·필터·보상 규칙을 정확히 옮겨, 어떤 학습 프레임워크(PyTorch/JAX)로도 그대로 이식할 수 있게 한다.
      여기의 모든 함수는 '토큰별 로그확률' 수준에서 동작한다(모델 무관).
주의: 논문에 명시되지 않은 선택은 코드에 `ASSUMPTION:` 으로 표시했다.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# 하이퍼파라미터 (§3.1.5)
# ---------------------------------------------------------------------------
@dataclass
class MAIConfig:
    # GRPO + 적응 엔트로피 제어 + 외부 클립
    eps: float = 0.6            # 기본 신뢰영역 폭 ε  → 하한 1-ε, 상한 (1-ε)^-1 + k
    k_max: float = 2.5          # 상한 완화 k 의 최대값
    delta: float = 0.25         # 적분 제어기 스텝 δ
    target_entropy: float = 0.3 # 목표 토큰 엔트로피 H* (nats)
    r_max: float = 50.0         # 외부 비율 클립 상한
    r_min: float = 0.0          # 하한은 사실상 비제약(=0)
    # 문제 샘플링
    G: int = 128                # 문제당 전체 롤아웃 수
    G_early: int = 16           # early-exit 단계 롤아웃 수
    early_range: tuple = (0.05, 0.8)
    full_range: tuple = (0.1, 0.8)
    top_p: float = 0.97
    # 보상 (식 10–12)
    w_lang: float = 0.5
    alpha_lang: float = 0.005
    w_len: float = 0.25         # 64k 단계까지, 128k 단계에서 0
    # 비동기
    grad_steps_per_update: int = 5   # 추론 가중치 갱신 사이 그래디언트 스텝
    max_stale_updates: int = 8       # 8 갱신(=40 스텝)보다 오래된 롤아웃 폐기
    # 최적화
    lr: float = 1e-6
    lr_long: float = 9e-7            # 긴 길이 단계에서 낮춤
    adam_betas: tuple = (0.95, 0.95)
    adam_eps: float = 1e-15
    weight_decay: float = 0.0
    global_batch_tokens_packed: int = 7040   # "global batch size 7040 after packing"
    max_unpacked_sequences: int = 12000
    length_schedule: tuple = (8192, 16384, 32768, 65536, 131072)


# ---------------------------------------------------------------------------
# 1) 그룹 상대 어드밴티지 (식 5 아래)
# ---------------------------------------------------------------------------
def group_advantages(rewards, eps: float = 1e-8) -> np.ndarray:
    """A_i = (R_i - mean(R)) / std(R). 분산 0 그룹은 0 (어차피 통과율 필터가 제거).
    ASSUMPTION: std 는 모집단 표준편차(ddof=0). 논문은 std(R_1:G)만 표기."""
    R = np.asarray(rewards, dtype=float)
    s = R.std()
    if s < eps:
        return np.zeros_like(R)
    return (R - R.mean()) / s


# ---------------------------------------------------------------------------
# 2) 토큰 레벨 대리목적함수 + 적응 상한 + 외부 비율 클립 (식 5, 6, 9)
# ---------------------------------------------------------------------------
def clip_bounds(eps: float, k: float) -> tuple[float, float]:
    """r^tr = clip(r, 1-ε, (1-ε)^-1 + k). k=0 이면 로그비 공간에서 대칭."""
    return 1.0 - eps, 1.0 / (1.0 - eps) + k


def token_surrogate(logp_new, logp_old, adv, k: float, cfg: MAIConfig):
    """토큰별 s = min(r_out·A, r_tr·A) 와 ds/d(logp_new).

    r = exp(logp_new - logp_old)          (식 6, logp_old = 롤아웃을 만든 정책)
    r_tr = clip(r, 1-ε, (1-ε)^-1 + k)     (적응 신뢰영역)
    r_out = clip(r, r_min, r_max)         (식 9, 모든 가지에 적용되는 외부 클립)
    해석: 표준 PPO가 열어 두는 두 가지(A<0·r>1, A>0·r<1)도 r_max 밖이면 상수가 되어 그래디언트 0.
    ASSUMPTION: 외부 클립을 PPO min 의 '비클립 가지'를 r_out 으로 바꾸는 방식으로 구현(이중클립 PPO와 같은 정신).
    """
    logp_new = np.asarray(logp_new, float)
    logp_old = np.asarray(logp_old, float)
    adv = np.asarray(adv, float)
    r = np.exp(logp_new - logp_old)
    lo, hi = clip_bounds(cfg.eps, k)
    r_tr = np.clip(r, lo, hi)
    r_out = np.clip(r, cfg.r_min, cfg.r_max)
    a = r_out * adv
    b = r_tr * adv
    s = np.minimum(a, b)
    # 활성 가지의 그래디언트: d(r·A)/dlogp = r·A (클립 안쪽일 때만), 클립되면 0
    ga = np.where((r >= cfg.r_min) & (r <= cfg.r_max), r * adv, 0.0)
    gb = np.where((r >= lo) & (r <= hi), r * adv, 0.0)
    g = np.where(a <= b, ga, gb)
    return s, g, r


def grpo_loss(logp_new, logp_old, adv_per_token, mask, k: float, cfg: MAIConfig):
    """전역 배치 토큰 정규화(식 5: 1/Σ|y_i| · ΣΣ). loss = -J.
    입력은 평탄화된 토큰 배열(모든 응답·모든 데이터병렬 랭크를 합친 것).
    adv_per_token: 응답 어드밴티지 A_i 를 그 응답의 모든 토큰에 방송한 값.
    반환: loss, d loss / d logp_new, 통계 dict."""
    mask = np.asarray(mask, float)
    n_tok = mask.sum()
    s, g, r = token_surrogate(logp_new, logp_old, adv_per_token, k, cfg)
    J = (s * mask).sum() / n_tok
    grad = -(g * mask) / n_tok
    lo, hi = clip_bounds(cfg.eps, k)
    m = mask > 0
    stats = {
        "J": J,
        "clipfrac_low": float(((r < lo) & m).sum() / n_tok),
        "clipfrac_high": float(((r > hi) & m).sum() / n_tok),
        "outer_clipfrac": float(((r > cfg.r_max) & m).sum() / n_tok),
        "ratio_mean": float((r * mask).sum() / n_tok),
    }
    return -J, grad, stats


# ---------------------------------------------------------------------------
# 3) 엔트로피 추정 + 적분 제어기 (식 7, 8)
# ---------------------------------------------------------------------------
def entropy_estimate(logp_new, logp_old, mask) -> float:
    """Ĥ(πθ) = (1/|T|) Σ -log πθ(y_t) · r_t  — πold 샘플로 πθ 엔트로피를 중요도 가중 추정."""
    mask = np.asarray(mask, float)
    logp_new = np.asarray(logp_new, float)
    r = np.exp(logp_new - np.asarray(logp_old, float))
    return float(((-logp_new) * r * mask).sum() / mask.sum())


def update_k(k: float, H_hat: float, cfg: MAIConfig) -> float:
    """k ← clip(k + δ·sign(H* - Ĥ), 0, k_max). 엔트로피 낮으면 상한을 넓혀 대안 토큰 확률을 더 올릴 수 있게."""
    return float(np.clip(k + cfg.delta * np.sign(cfg.target_entropy - H_hat), 0.0, cfg.k_max))


# ---------------------------------------------------------------------------
# 4) top-p 마스크 재생 (§3.1.3)
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def topp_mask(logits, p: float) -> np.ndarray:
    """누적확률이 p 에 처음 도달하는 최소 집합(경계 토큰 포함) = True."""
    probs = softmax(np.asarray(logits, float))
    order = np.argsort(-probs, axis=-1)
    sorted_p = np.take_along_axis(probs, order, axis=-1)
    csum = np.cumsum(sorted_p, axis=-1)
    keep_sorted = (csum - sorted_p) < p          # 이전 누적이 p 미만이면 포함
    mask = np.zeros_like(keep_sorted)
    np.put_along_axis(mask, order, keep_sorted, axis=-1)
    return mask.astype(bool)


def masked_log_softmax(logits, mask) -> np.ndarray:
    """롤아웃 때 저장한 nucleus 마스크를 학습 때 재생: 제외 토큰 logit = -inf 후 log-softmax.
    논문: 이를 안 하면 수 스텝 내 발산하는 '파국적 오프폴리시 불일치'."""
    x = np.where(mask, np.asarray(logits, float), -np.inf)
    m = np.max(x, axis=-1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
    return x - lse


# ---------------------------------------------------------------------------
# 5) 문제 샘플링: early-exit + 통과율 필터 (§3.1.3, §3.6.2)
# ---------------------------------------------------------------------------
def pass_rate(rewards) -> float:
    """통과 = 보상 > 0 (논문: 'fraction of responses with positive reward')."""
    return float((np.asarray(rewards) > 0).mean())


def sample_problem(rollout_fn, grade_fn, cfg: MAIConfig):
    """rollout_fn(n) -> n개 응답, grade_fn(responses) -> 보상 배열.
    1) G_early 개로 통과율 → early_range 밖이면 폐기(None)
    2) 나머지를 채워 G 개 → full_range 밖이면 폐기
    ASSUMPTION: §3.1.5 'G=128 total' 을 따라 early 16개를 포함해 총 128개로 채움
    (§3.6.2 는 'additional 128' 이라 적어 모호함 — 설정으로 바꿀 수 있게 extra_full 사용)."""
    early = rollout_fn(cfg.G_early)
    r_early = np.asarray(grade_fn(early), float)
    pe = pass_rate(r_early)
    if not (cfg.early_range[0] <= pe <= cfg.early_range[1]):
        return None, {"stage": "early", "pass_rate": pe}
    rest = rollout_fn(cfg.G - cfg.G_early)
    responses = list(early) + list(rest)
    R = np.concatenate([r_early, np.asarray(grade_fn(rest), float)])
    pf = pass_rate(R)
    if not (cfg.full_range[0] <= pf <= cfg.full_range[1]):
        return None, {"stage": "full", "pass_rate": pf}
    return (responses, R), {"stage": "kept", "pass_rate": pf}


# ---------------------------------------------------------------------------
# 6) 보상 (식 10–12) + 결합 규칙 (§3.4.1)
# ---------------------------------------------------------------------------
def lang_reward(n_non_english, alpha: float) -> np.ndarray:
    """R_lang = max(1 - α·n_non-english, 0)  (CoT 안의 비영어 단어 수)."""
    return np.maximum(1.0 - alpha * np.asarray(n_non_english, float), 0.0)


def length_penalty(lengths, pass_rate_q: float, l_max: int) -> np.ndarray:
    """R_len = ρ_q · |y| / ℓ_max. 쉬운 문제(통과율↑)일수록 강하게, 어려운 문제엔 약하게."""
    return pass_rate_q * np.asarray(lengths, float) / float(l_max)


def total_reward(r_task, n_non_english, lengths, pass_rate_q, l_max, cfg: MAIConfig, w_len=None):
    """R = R_task + w_lang·R_lang - w_len·R_len  (식 10).
    주의: ρ_q 는 과제 보상으로 계산한 통과율. ASSUMPTION: 같은 그룹의 R_task>0 비율 사용."""
    w_len = cfg.w_len if w_len is None else w_len
    return (np.asarray(r_task, float)
            + cfg.w_lang * lang_reward(n_non_english, cfg.alpha_lang)
            - w_len * length_penalty(lengths, pass_rate_q, l_max))


def lexicographic_reward(reward_columns, tie_tol: float = 1e-12) -> np.ndarray:
    """사전식 셰이핑: 열(우선순위 순) 중 그룹 안에서 '모두 같지 않은' 첫 열을 보상으로 사용.
    하위 보상은 상위 보상이 그룹 내 동률일 때만 그래디언트에 영향 → 스케일 보정 불필요."""
    M = np.asarray(reward_columns, float)  # [G, K]
    for j in range(M.shape[1]):
        col = M[:, j]
        if col.max() - col.min() > tie_tol:
            return col
    return np.zeros(M.shape[0])


def gated_reward(primary_ok, secondary, r_floor: float) -> np.ndarray:
    """게이트: 상위 기준(예: 안전) 불통과면 최소 보상, 통과한 것만 하위 기준(품질)으로 채점."""
    return np.where(np.asarray(primary_ok, bool), np.asarray(secondary, float), r_floor)


# ---------------------------------------------------------------------------
# 7) 비동기: 정책 버전·staleness (§3.1.5, Rocket 글)
# ---------------------------------------------------------------------------
def inference_version(learner_step: int, cfg: MAIConfig) -> int:
    """추론 가중치는 grad_steps_per_update(=5) 스텝마다 갱신."""
    return learner_step // cfg.grad_steps_per_update


def keep_rollout(token_versions, learner_step: int, cfg: MAIConfig) -> bool:
    """생성 정책이 8 갱신(=40 스텝)보다 오래되면 폐기.
    한 롤아웃이 여러 가중치 버전에 걸칠 수 있음(Rocket 글: 최장 롤아웃이 6개 버전 사용).
    ASSUMPTION: 가장 오래된 토큰 버전 기준(보수적)."""
    oldest = int(np.min(token_versions))
    return inference_version(learner_step, cfg) - oldest <= cfg.max_stale_updates


def staleness_steps(sampler_lag, t_train, t_gen, dt_step) -> float:
    """Rocket 글 근사식: (샘플러 지연 + 학습시각 - 생성시각) / 스텝 간격."""
    return (sampler_lag + t_train - t_gen) / dt_step


# ---------------------------------------------------------------------------
# 8) 길이 커리큘럼 (§3.1.3, §3.1.5)
# ---------------------------------------------------------------------------
def stage_settings(stage: int, cfg: MAIConfig) -> dict:
    """8k→16k→32k→64k→128k. w_len 은 64k 단계까지 0.25, 128k 에서 0.
    ASSUMPTION: lr 을 9e-7 로 낮추는 '긴 길이' 기준을 64k 이상으로 둠(논문 미기재)."""
    L = cfg.length_schedule[stage]
    return {
        "max_len": L,
        "w_len": 0.0 if L >= 131072 else cfg.w_len,
        "lr": cfg.lr_long if L >= 65536 else cfg.lr,
    }


# ---------------------------------------------------------------------------
# 9) 학습–추론 수치 불일치 모니터 (§3.6.4, Rocket 글 Fig.3)
# ---------------------------------------------------------------------------
def logprob_gap(logp_sampler, logp_learner, mask) -> float:
    """같은 토큰에 대한 샘플러와 학습기 로그확률의 평균 절대차. 0 이 목표(Rocket2 는 비트 단위 일치)."""
    mask = np.asarray(mask, float)
    return float((np.abs(np.asarray(logp_sampler) - np.asarray(logp_learner)) * mask).sum() / mask.sum())
