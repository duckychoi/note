"""
RSI 서베이(arXiv 2609.11873, "The Last AI Built by Humans", v3) — 코드 수준 골격
================================================================================
논문은 서베이라 단일 알고리즘이 없다. 여기서는
  (1) 논문에 수식이 있는 Headroom-Closed Index(HCI) 계산을 그대로 구현하고,
  (2) 개선 루프의 9요소를 데이터 모델로,
  (3) 단계(B0, L1–L5) 판정 규칙을,
  (4) 단계별 대표 메커니즘(서베이가 서술한 것)을 실행 가능한 골격으로,
  (5) 서베이가 요구하는 평가 프로토콜(예산 장부, 구조적/효과적 L5)을
옮긴다. 서베이가 수식 없이 서술만 한 부분의 구체 식은 `OUR CHOICE:` 로 표시한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np


# =============================================================================
# 1. Headroom-Closed Index (§2.1, 식 1–4)  — 논문 수식 그대로
# =============================================================================
SOURCE_BASE_WEIGHT = {
    "benchmark_owner": 3.0,        # 벤치마크 소유자 표
    "independent_harness": 2.5,    # 독립 공통 하네스 평가
    "combined_report": 2.0,        # 벤치마크·모델 통합 보고
    "model_author": 1.0,           # 모델 저자 표
}
FIRST_PARTY_FACTOR = 0.75          # 1자 보고에 추가 곱


def consensus_score(scores, source_types, first_party):
    """식 1: s̄ = Σ w̃_i s_i / Σ w̃_i,  w̃ = 기본가중 × (0.75 if 1자 보고)."""
    w = np.array([SOURCE_BASE_WEIGHT[t] * (FIRST_PARTY_FACTOR if fp else 1.0)
                  for t, fp in zip(source_types, first_party)])
    s = np.asarray(scores, float)
    return float((w * s).sum() / w.sum())


def entry_frontier(entry_year_scores, q: float = 90.0) -> float:
    """F_{b,0} = 벤치마크가 데이터에 들어온 첫해 모델 점수의 90번째 백분위.
    OUR CHOICE: 백분위 보간법은 numpy 기본(linear). 논문 미기재."""
    return float(np.percentile(np.asarray(entry_year_scores, float), q))


def hci(score: float, F0: float) -> float:
    """식 2: H = 100·(s̄ − F0)/(100 − F0). 0=진입년 프런티어, 100=만점. (점수는 0–100 척도)"""
    return 100.0 * (score - F0) / (100.0 - F0)


def family_year_frontier(hci_values, q: float = 90.0) -> float:
    """Q_{b,y} = 벤치마크 계열·연도별 HCI 의 90번째 백분위."""
    return float(np.percentile(np.asarray(hci_values, float), q))


def domain_trajectory(Q_by_family: dict, n_models_by_family: dict) -> float:
    """식 3: T_{d,y} = Σ_b √n_{b,y} Q_{b,y} / Σ_b √n_{b,y}  (√ 가중: 큰 표의 독식 방지)."""
    num = sum(math.sqrt(n_models_by_family[b]) * Q_by_family[b] for b in Q_by_family)
    den = sum(math.sqrt(n_models_by_family[b]) for b in Q_by_family)
    return num / den


def illustrative_rsi_endpoint(T_2026: float) -> float:
    """식 4(예시용 가설): R_d = 100 − 0.22(100 − T_{d,2026}). 이득 = 0.78·(남은 헤드룸).
    주의: 논문도 '예시(illustrative)'라고 명시 — 예측이 아니라 가설 그림."""
    return 100.0 - 0.22 * (100.0 - T_2026)


# =============================================================================
# 2. 개선 루프 9요소 (§2.2.1) — 데이터 모델
# =============================================================================
@dataclass
class SystemState:
    """한 라운드 끝에 남아 다음 라운드가 물려받는 것(파라미터·메모리·스킬·하네스·개선기·평가기 등)."""
    version: int
    artifacts: dict                       # 예: {"prompt": ..., "skills": [...], "improver_code": ..., "evaluator": ...}
    parent: Optional[int] = None


@dataclass
class Experience:
    """이전 상호작용에서 얻어 다음 수정을 이끄는 정보(테스트 실패, 환경 결과, 검토자 수정 등)."""
    payload: Any
    source: str                           # "test" | "env" | "human" | "judge" ...


@dataclass
class Candidate:
    target: str                           # 이번 라운드에 직접 수정하는 대상(artifacts 의 키)
    new_value: Any
    rationale: str = ""                   # 동기 증거 요약(상속 기록용)


@dataclass
class Verdict:
    accept: bool
    score: float
    details: dict = field(default_factory=dict)


class Improver(Protocol):
    """현재 상태+경험 → 후보 수정. 내부의 '전략'도 artifacts 에 두면 L5 에서 대상이 될 수 있다."""
    def propose(self, state: SystemState, experience: list[Experience]) -> list[Candidate]: ...


class Verifier(Protocol):
    """후보를 평가하고 수용 규칙을 적용."""
    def judge(self, state: SystemState, cand: Candidate) -> Verdict: ...


@dataclass
class LineageRecord:
    """향후 과제 §'재현 가능한 상속 인프라'가 요구한 필드: 부모 상태, 제안 변경, 동기 증거,
    평가 구성, 수용 결정, 이후 사용."""
    parent_version: int
    child_version: Optional[int]
    candidate: Candidate
    verdict: Verdict
    eval_config: dict
    later_uses: list = field(default_factory=list)


def run_round(state: SystemState, improver: Improver, verifier: Verifier,
              experience: list[Experience], eval_config: dict, ledger: "BudgetLedger") -> tuple[SystemState, list[LineageRecord]]:
    """개선 루프 1라운드: 제안 → 평가·수용 → 수용된 변경을 상태에 보존 → 후계자 반환."""
    records, new_artifacts, accepted_any = [], dict(state.artifacts), False
    for cand in improver.propose(state, experience):
        ledger.charge("proposal", 1)
        v = verifier.judge(state, cand)
        ledger.charge("verification", 1)
        if v.accept:
            new_artifacts[cand.target] = cand.new_value
            accepted_any = True
        records.append(LineageRecord(state.version, None, cand, v, eval_config))
    succ = SystemState(state.version + 1, new_artifacts, parent=state.version) if accepted_any else state
    for r in records:
        if r.verdict.accept:
            r.child_version = succ.version
    return succ, records


# =============================================================================
# 3. 단계 판정 (§3, 교차 단계 정리) — 인접 단계 경계를 규칙으로
# =============================================================================
@dataclass
class LoopAudit:
    """감사 질문: 루프는 어디서 닫히나 / 무엇이 상속되나 / 어떤 결정이 외부에 남나."""
    persistent_change: bool               # B0→L1: 수용된 변경이 현재 과제를 넘어 남는가
    ai_selects_intervention: bool         # L1→L2: 어떤 개입을 시도할지 AI가 고르는가
    learner_conditioned_acquisition: bool # L2→L3: 학습자 상태가 다음 경험 획득을 바꾸고 그 결과가 다시 되먹임되나
    deployment_adaptation: bool           # L3→L4: 배포 상호작용 피드백이 지속 상태를 바꾸나
    revises_improvement_mechanism: bool   # L4→L5: 이후 개선을 지배하는 메커니즘(개선기·평가기·연구정책)을 고치고 재사용하나
    mechanism_better_under_matched_budget: Optional[bool] = None  # '효과적 L5' 증거


def classify_level(a: LoopAudit) -> str:
    """가장 높은 충족 경계까지. 논문: 높은 단계 ≠ 더 좋은 개선 과정(자율성과 품질은 별개 축).
    주의: 논문은 각 단계가 앞 단계를 엄격히 포함한다고 요구하지 않고 '해당 루프'를 분류 — 여기선 최고 충족 경계."""
    if not a.persistent_change:
        return "B0 (비-RSI 기준: 출력만 개선)"
    if a.revises_improvement_mechanism:
        # 구조적 L5: 고친 메커니즘이 상속·호출됨 / 효과적 L5: 같은 예산·독립 평가에서 더 나은 후계자를 낳음
        return "L5 effective" if a.mechanism_better_under_matched_budget else "L5 structural"
    if a.deployment_adaptation:
        return "L4"
    if a.learner_conditioned_acquisition:
        return "L3"
    if a.ai_selects_intervention:
        return "L2"
    return "L1"


# =============================================================================
# 4. 단계별 메커니즘 골격
# =============================================================================

# --- L4 입장 검사 (HDSO 방식): 같은 과제를 대조(현 저장소) vs 처치(+후보)로 두 번 실행, 단계적 확대 ---
def paired_admission_test(run_task: Callable[[Any, bool], float], task_stages: list[list[Any]],
                          confirm_tasks: list[Any], z: float = 1.96) -> dict:
    """run_task(task, with_candidate) -> 점수. 서베이: '차이가 가설을 지지할 때만' 승인, 점점 큰 단계 + 독립 과제 확인.
    OUR CHOICE: '지지' = 짝 차이 평균의 95% 신뢰구간 하한 > 0 (각 단계와 최종 확인 모두)."""
    def stage_ok(tasks):
        d = np.array([run_task(t, True) - run_task(t, False) for t in tasks])
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("inf")
        lo = d.mean() - z * se
        return lo > 0, float(d.mean()), float(lo)
    log = []
    for i, tasks in enumerate(task_stages):
        ok, m, lo = stage_ok(tasks)
        log.append(("stage", i, len(tasks), m, lo))
        if not ok:
            return {"admit": False, "log": log}
    ok, m, lo = stage_ok(confirm_tasks)
    log.append(("confirm", len(confirm_tasks), m, lo))
    return {"admit": ok, "log": log}


# --- L4 스킬 라이브러리 수명관리 (Library Drift) ---
@dataclass
class SkillLibrary:
    """서베이: 추가만 하는 증거 로그 / 기여가 바래면 은퇴 / 활성 스킬 수 상한(새 스킬은 기존과 경쟁) /
    메타스킬로 작성 지침. 너무 공격적인 은퇴는 무지도 방치보다 나빴다는 보고도 있음.
    OUR CHOICE: 기여 = 최근 window 개 사용에서 (사용 시 성공률 − 미사용 기준 성공률)의 이동평균."""
    cap: int = 20
    window: int = 50
    retire_below: float = 0.0
    min_uses_before_retire: int = 20
    evidence: list = field(default_factory=list)      # append-only: (skill, used, success)
    active: dict = field(default_factory=dict)        # name -> skill body
    retired: dict = field(default_factory=dict)

    def log(self, skill: str, used: bool, success: bool):
        self.evidence.append((skill, used, success))

    def contribution(self, skill: str) -> tuple[float, int]:
        ev = [e for e in self.evidence if e[0] == skill][-self.window:]
        used = [s for _, u, s in ev if u]
        unused = [s for _, u, s in ev if not u]
        if not used or not unused:
            return 0.0, len(used)
        return float(np.mean(used) - np.mean(unused)), len(used)

    def admit(self, name: str, body: Any) -> bool:
        if len(self.active) < self.cap:
            self.active[name] = body
            return True
        worst = min(self.active, key=lambda s: self.contribution(s)[0])
        if self.contribution(worst)[0] < 0:          # 새 스킬은 가장 약한 기존 스킬과 경쟁
            self.retired[worst] = self.active.pop(worst)
            self.active[name] = body
            return True
        return False

    def maintain(self):
        for s in list(self.active):
            c, n = self.contribution(s)
            if n >= self.min_uses_before_retire and c < self.retire_below:
                self.retired[s] = self.active.pop(s)


# --- L5 후계자 평가기 공진화 (RQGM 방식): 에포크 내 동결 + 독립 앵커로 교체 검증 + 무효화 ---
@dataclass
class EpochEvaluator:
    """서베이: 평가기는 에포크 안에서 고정, 경계에서 도전 평가기를 독립 정답 앵커와 비교해 교체,
    교체된 평가기에 의존한 점수는 폐기하고 재방문 시 재평가. 안정성 논증은 에포크 내부에만 적용."""
    evaluator: Callable[[Any], float]
    epoch: int = 0
    scores: dict = field(default_factory=dict)        # agent_id -> (epoch, score)

    def score(self, agent_id, agent) -> float:
        hit = self.scores.get(agent_id)
        if hit and hit[0] == self.epoch:
            return hit[1]
        s = self.evaluator(agent)                     # 옛 에포크 점수는 재평가
        self.scores[agent_id] = (self.epoch, s)
        return s

    def epoch_boundary(self, challengers: list[Callable[[Any], float]],
                       anchor_items: list[Any], anchor_truth: list[float]) -> bool:
        """앵커 일치도(여기선 음의 MSE)가 가장 높은 평가기를 다음 에포크에 사용. OUR CHOICE: 일치도=−MSE."""
        def agree(ev):
            return -float(np.mean([(ev(x) - y) ** 2 for x, y in zip(anchor_items, anchor_truth)]))
        best = max([self.evaluator] + challengers, key=agree)
        replaced = best is not self.evaluator
        if replaced:
            self.evaluator = best
            self.epoch += 1                           # 이전 점수는 자동으로 무효(에포크 불일치)
        return replaced


# --- L3 학습자 조건부 경험 획득: 출제자 보상이 현재 풀이자에 의존 ---
def consistency_difficulty_reward(solver_answers: list[Any]) -> float:
    """서베이(R-Zero·VisPlay 서술): 출제자 보상은 현재 풀이자의 여러 응답 일치도에 의존,
    다수답 확신이 '중간'에 가까울수록 보상(VisPlay). 일치도는 난이도 대리지표일 뿐 정답성·학습이득 증거 아님.
    OUR CHOICE: conf = 다수답 비율, reward = 1 − 2|conf − 0.5|  (0.5에서 1, 0 또는 1에서 0)."""
    vals, counts = np.unique(np.asarray(solver_answers, dtype=object), return_counts=True)
    conf = counts.max() / counts.sum()
    return float(1.0 - 2.0 * abs(conf - 0.5))


def select_next_experience(pool: list[Any], learner_signal: Callable[[Any], float], k: int,
                           diversity_key: Optional[Callable[[Any], Any]] = None) -> list[Any]:
    """학습자 신호로 다음 과제 k개 선택(+선택적 다양성: 같은 key 중복 억제). 고정 스케줄과 비교해야 함(§L3 평가)."""
    ranked = sorted(pool, key=learner_signal, reverse=True)
    if diversity_key is None:
        return ranked[:k]
    out, seen = [], set()
    for x in ranked:
        key = diversity_key(x)
        if key in seen:
            continue
        out.append(x); seen.add(key)
        if len(out) == k:
            break
    return out


# --- L5 자기 적용 개선기 (STOP 방식) ---
def meta_utility(improver: Callable[[str, Callable[[str], float]], str],
                 downstream: list[tuple[str, Callable[[str], float]]]) -> float:
    """서베이: 후계 개선기는 '그것이 만든 프로그램의 하위 과제 품질'로 채점.
    improver(program, utility) -> 개선된 program. downstream = [(seed_program, utility), ...]."""
    return float(np.mean([u(improver(p, u)) for p, u in downstream]))


def stop_round(improver_src: str, compile_improver: Callable[[str], Callable],
               downstream: list, transfer: list) -> dict:
    """현재 개선기가 자기 소스를 개선 대상으로 받는다. 효용 = meta_utility.
    서베이 경고: 약한 모델에선 평균 퇴행, 일부 프로그램이 소프트 예산 우회·평가 버그 악용 → 하드 예산과 보호 평가 필수.
    compile_improver 는 샌드박스에서 문자열을 함수로 만드는 사용자 제공 함수(여기선 구현하지 않음)."""
    cur = compile_improver(improver_src)
    meta_u = lambda src: meta_utility(compile_improver(src), downstream)
    new_src = cur(improver_src, meta_u)
    return {
        "new_src": new_src,
        "meta_utility_old": meta_u(improver_src),
        "meta_utility_new": meta_u(new_src),
        # '효과적 L5' 판정은 자기개선에 쓰지 않은 전이 과제에서
        "transfer_old": meta_utility(cur, transfer),
        "transfer_new": meta_utility(compile_improver(new_src), transfer),
    }


# =============================================================================
# 5. 평가 프로토콜 (§L2–L5 평가, 향후 과제)
# =============================================================================
@dataclass
class BudgetLedger:
    """서베이: 비교는 총 예산(후보 생성·평가·메커니즘 개발·인간 검토 포함)을 맞춰야 함."""
    spent: dict = field(default_factory=dict)

    def charge(self, kind: str, amount: float):
        self.spent[kind] = self.spent.get(kind, 0.0) + amount

    def total(self) -> float:
        return sum(self.spent.values())


def matched_budget_compare(run_a: Callable[[float], float], run_b: Callable[[float], float],
                           budget: float, seeds: list[int]) -> dict:
    """같은 총 예산에서 두 메커니즘을 반복 실행해 짝 차이와 95% 구간. run_x(budget, seed)->점수."""
    d = np.array([run_b(budget, s) - run_a(budget, s) for s in seeds])
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("inf")
    return {"mean_diff": float(d.mean()), "ci95": (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se))}
