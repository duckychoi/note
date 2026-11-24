"""rsi_core 테스트 — `python3 tests/test_rsi_core.py`. 논문에 적힌 수치로 HCI 식을 검산한다."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import rsi_core as R  # noqa: E402


def test_illustrative_endpoints_match_paper():
    # 논문 §2.1 Observation 3: 사이버보안 91.9→98.2, SWE 52.6→89.6, 검색·터미널 56.8→90.5, 도구 39.9→86.8
    for T, R_paper in [(91.9, 98.2), (52.6, 89.6), (56.8, 90.5), (39.9, 86.8)]:
        assert abs(round(R.illustrative_rsi_endpoint(T), 1) - R_paper) < 1e-9, (T, R.illustrative_rsi_endpoint(T))


def test_paper_gaps():
    # Observation 2: 대학원 과학 85.8 대비 SWE 52.6 / 검색 56.8 / 도구 39.9 → 33.2 / 29.0(논문 29.1) / 45.9
    assert abs((85.8 - 52.6) - 33.2) < 1e-9 and abs((85.8 - 39.9) - 45.9) < 1e-9
    assert abs((91.9 - 39.9) - 52.0) < 1e-9   # 사이버 vs 도구 52.0


def test_hci_bounds_and_consensus():
    assert R.hci(60.0, 60.0) == 0.0 and R.hci(100.0, 60.0) == 100.0
    s = R.consensus_score([70, 80], ["benchmark_owner", "model_author"], [False, True])
    # 가중 3 과 1×0.75 → (3·70 + 0.75·80)/3.75 = 72
    assert abs(s - 72.0) < 1e-12


def test_domain_trajectory_sqrt_weights():
    T = R.domain_trajectory({"a": 50.0, "b": 80.0}, {"a": 4, "b": 16})
    assert abs(T - (2 * 50 + 4 * 80) / 6) < 1e-12


def test_classify_levels():
    A = R.LoopAudit
    assert R.classify_level(A(False, False, False, False, False)).startswith("B0")      # Self-Refine
    assert R.classify_level(A(True, False, False, False, False)) == "L1"                 # FineWeb-Edu
    assert R.classify_level(A(True, True, False, False, False)) == "L2"                  # GEPA
    assert R.classify_level(A(True, True, True, False, False)) == "L3"                   # AZR
    assert R.classify_level(A(True, True, True, True, False)) == "L4"                    # PANDO
    assert R.classify_level(A(True, True, False, False, True, None)) == "L5 structural"  # 구조적
    assert R.classify_level(A(True, True, False, False, True, True)) == "L5 effective"   # STOP(전이 과제 우위)


def test_run_round_inheritance():
    class Imp:
        def propose(self, st, ex):
            return [R.Candidate("x", st.artifacts["x"] + 1, "증가"), R.Candidate("x", -99, "나쁜 후보")]
    class Ver:
        def judge(self, st, c):
            ok = c.new_value > st.artifacts["x"]
            return R.Verdict(ok, float(c.new_value))
    led = R.BudgetLedger()
    s1, recs = R.run_round(R.SystemState(0, {"x": 1}), Imp(), Ver(), [], {"suite": "v1"}, led)
    assert s1.version == 1 and s1.artifacts["x"] == 2 and s1.parent == 0
    assert [r.verdict.accept for r in recs] == [True, False] and recs[0].child_version == 1
    assert led.total() == 4


def test_paired_admission():
    rng = np.random.default_rng(0)
    good = lambda t, cand: t + (0.5 if cand else 0.0) + rng.normal(0, 0.1)
    bad = lambda t, cand: t + rng.normal(0, 0.1)
    stages = [list(range(5)), list(range(20))]
    assert R.paired_admission_test(good, stages, list(range(100, 130)))["admit"]
    assert not R.paired_admission_test(bad, stages, list(range(100, 130)))["admit"]


def test_skill_library_cap_and_retire():
    lib = R.SkillLibrary(cap=2, min_uses_before_retire=5, window=40)
    assert lib.admit("a", 1) and lib.admit("b", 2)
    for i in range(30):
        lib.log("a", used=(i % 2 == 0), success=(i % 2 == 0))   # a: 쓰면 성공, 안 쓰면 실패 → 기여 +1
        lib.log("b", used=(i % 2 == 0), success=(i % 2 == 1))   # b: 쓰면 실패 → 기여 −1
    assert lib.admit("c", 3) and "b" in lib.retired and "a" in lib.active
    lib.log("c", True, False)
    for _ in range(10):
        lib.log("c", True, False); lib.log("c", False, True)
    lib.maintain()
    assert "c" in lib.retired


def test_epoch_evaluator_invalidation():
    truth = lambda x: 2 * x
    ev = R.EpochEvaluator(evaluator=lambda x: x)               # 틀린 평가기(0.5배)
    s_old = ev.score("agent", 3.0)
    replaced = ev.epoch_boundary([truth], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert replaced and ev.epoch == 1
    assert ev.score("agent", 3.0) == 6.0 and s_old == 3.0      # 옛 점수는 재평가됨


def test_consistency_reward():
    assert R.consistency_difficulty_reward(["a", "a", "b", "b"]) == 1.0      # 확신 0.5 → 최대
    assert R.consistency_difficulty_reward(["a"] * 4) == 0.0                 # 다 같으면 0
    picked = R.select_next_experience([1, 2, 3, 4], lambda x: -abs(x - 2.5), 2)
    assert set(picked) == {2, 3}


def test_matched_budget_compare():
    r = R.matched_budget_compare(lambda b, s: 1.0 + 0.01 * s, lambda b, s: 1.2 + 0.01 * s, 100, list(range(5)))
    assert abs(r["mean_diff"] - 0.2) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
