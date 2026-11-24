"""mai_grpo 단위테스트 — `python3 tests/test_mai_grpo.py` 로 실행 (pytest 없이도 동작)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import mai_grpo as M  # noqa: E402

cfg = M.MAIConfig()


def test_advantages():
    A = M.group_advantages([1, 0, 0, 1, 1])
    assert abs(A.mean()) < 1e-12 and abs(A.std() - 1) < 1e-9
    assert np.all(M.group_advantages([1, 1, 1]) == 0)


def test_clip_bounds_symmetric_logspace():
    lo, hi = M.clip_bounds(0.6, 0.0)
    assert abs(np.log(lo) + np.log(hi)) < 1e-12          # k=0 → 로그비 대칭
    lo, hi = M.clip_bounds(0.6, 2.5)
    assert abs(hi - (2.5 + 2.5)) < 1e-12                  # (1-0.6)^-1 + 2.5 = 5.0


def test_branch_cases():
    k = 0.0
    lo, hi = M.clip_bounds(cfg.eps, k)                    # 0.4, 2.5
    old = np.zeros(1)
    # (a) A<0, r>hi (PPO가 여는 가지) → 비클립 그래디언트 r·A
    r = 3.0; s, g, _ = M.token_surrogate([np.log(r)], old, [-1.0], k, cfg)
    assert abs(g[0] - r * -1.0) < 1e-12
    # 같은 경우 r > r_max 면 외부 클립으로 그래디언트 0
    r = 60.0; s, g, _ = M.token_surrogate([np.log(r)], old, [-1.0], k, cfg)
    assert g[0] == 0.0 and abs(s[0] - (-cfg.r_max)) < 1e-9
    # (b) A>0, r<lo → 비클립 가지, 그래디언트 r·A
    r = 0.2; s, g, _ = M.token_surrogate([np.log(r)], old, [1.0], k, cfg)
    assert abs(g[0] - r) < 1e-12
    # A>0, r>hi → 신뢰영역 클립, 그래디언트 0
    r = 3.0; s, g, _ = M.token_surrogate([np.log(r)], old, [1.0], k, cfg)
    assert g[0] == 0.0 and abs(s[0] - hi) < 1e-12
    # k 를 키우면 같은 r=3 도 신뢰영역 안 → 그래디언트 살아남
    s, g, _ = M.token_surrogate([np.log(3.0)], old, [1.0], 1.0, cfg)
    assert abs(g[0] - 3.0) < 1e-12


def test_finite_difference_gradient():
    rng = np.random.default_rng(0)
    n = 400
    logp_old = np.log(rng.uniform(0.05, 0.9, n))
    logp_new = logp_old + rng.normal(0, 0.9, n)          # 다양한 비율(클립 안팎 모두)
    adv = rng.normal(0, 1, n)
    mask = (rng.random(n) > 0.1).astype(float)
    k = 0.7
    loss, grad, _ = M.grpo_loss(logp_new, logp_old, adv, mask, k, cfg)
    h = 1e-6
    lo, hi = M.clip_bounds(cfg.eps, k)
    r = np.exp(logp_new - logp_old)
    checked = 0
    for i in range(n):
        # 경계 근처(미분 불가점)는 건너뜀
        if min(abs(r[i] - lo), abs(r[i] - hi), abs(r[i] - cfg.r_max)) < 1e-3:
            continue
        lp = logp_new.copy(); lp[i] += h
        lm = logp_new.copy(); lm[i] -= h
        fd = (M.grpo_loss(lp, logp_old, adv, mask, k, cfg)[0] - M.grpo_loss(lm, logp_old, adv, mask, k, cfg)[0]) / (2 * h)
        assert abs(fd - grad[i]) < 1e-6, (i, fd, grad[i])
        checked += 1
    assert checked > 300


def test_token_level_normalization():
    # 긴 응답의 토큰도 짧은 응답의 토큰과 같은 가중(응답 길이로 나누지 않음)
    logp = np.zeros(10); adv = np.ones(10); mask = np.ones(10)
    _, grad, _ = M.grpo_loss(logp, logp, adv, mask, 0.0, cfg)
    assert np.allclose(grad, -0.1)


def test_entropy_controller():
    k = 0.0
    k = M.update_k(k, H_hat=0.1, cfg=cfg); assert k == cfg.delta      # 엔트로피 낮음 → k↑
    k = M.update_k(k, H_hat=0.9, cfg=cfg); assert k == 0.0            # 높음 → k↓, 0에서 막힘
    for _ in range(100):
        k = M.update_k(k, 0.0, cfg)
    assert k == cfg.k_max                                             # k_max 에서 막힘


def test_entropy_estimate_onpolicy():
    # 온폴리시(r=1)에서 Ĥ = 표본 -log p 평균 → 범주분포 엔트로피의 불편 추정
    rng = np.random.default_rng(1)
    p = np.array([0.7, 0.2, 0.1])
    a = rng.choice(3, size=200000, p=p)
    lp = np.log(p[a])
    H = M.entropy_estimate(lp, lp, np.ones_like(lp))
    assert abs(H - float(-(p * np.log(p)).sum())) < 5e-3


def test_topp_mask_and_replay():
    logits = np.array([[3.0, 2.0, 1.0, -5.0, -6.0]])
    m = M.topp_mask(logits, 0.97)
    assert m[0, 0] and not m[0, 4]
    lp = M.masked_log_softmax(logits, m)
    assert np.isneginf(lp[0, 4]) and abs(np.exp(lp[0][m[0]]).sum() - 1) < 1e-12


def test_sample_problem_filters():
    rng = np.random.default_rng(2)
    mk = lambda pr: (lambda n: list(rng.random(n) < pr), lambda resp: np.asarray(resp, float))
    out, info = M.sample_problem(*mk(0.0), cfg); assert out is None and info["stage"] == "early"   # 다 틀림
    out, info = M.sample_problem(*mk(1.0), cfg); assert out is None and info["stage"] == "early"   # 다 맞음
    out, info = M.sample_problem(*mk(0.5), cfg); assert out is not None and len(out[1]) == cfg.G


def test_rewards_and_combination():
    assert np.allclose(M.lang_reward([0, 100, 1000], 0.005), [1.0, 0.5, 0.0])
    assert np.allclose(M.length_penalty([1000, 4000], 0.5, 8000), [0.0625, 0.25])
    col = M.lexicographic_reward([[1, 0.2], [1, 0.9], [1, 0.4]])   # 1순위 동률 → 2순위 사용
    assert np.allclose(col, [0.2, 0.9, 0.4])
    col = M.lexicographic_reward([[1, 0.2], [0, 0.9]])               # 1순위 차이 → 1순위만
    assert np.allclose(col, [1, 0])
    assert np.allclose(M.gated_reward([True, False], [0.8, 0.9], -1.0), [0.8, -1.0])


def test_staleness():
    # 학습 스텝 100 → 추론 버전 20. 버전 12 까지 허용(20-12=8), 11 은 폐기
    assert M.keep_rollout([12, 15], 100, cfg)
    assert not M.keep_rollout([11, 20], 100, cfg)


def test_stage_settings():
    assert M.stage_settings(0, cfg) == {"max_len": 8192, "w_len": 0.25, "lr": 1e-6}
    assert M.stage_settings(4, cfg)["w_len"] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
