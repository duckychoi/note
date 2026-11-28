"""Summarize the PF route experiments from scene-level outputs."""
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "research-sources/encoder42_eval/pf42-trials"
SETS = {"hammer": "HAMMER", "ibims": "iBims-1", "diode": "DIODE 실내", "diode_outdoor": "DIODE 실외"}
TRAINED = {
    "synthetic_equal": "합성 직접 CE, 660회",
    "loss_exact": "A 정확 우도",
    "loss_pf16": "A 부트스트랩 16",
    "loss_pf64": "A 부트스트랩 64",
    "loss_defensive": "A 보호 제안 64",
    "hard": "B 실사 중앙값",
    "teacher_iid": "B 독립 관측 사후",
    "teacher_exact": "B 정확 순차 사후",
    "teacher_hard": "B 정확 사후의 결정값",
    "teacher_pf": "B 부트스트랩 사후",
    "teacher_defensive": "B 보호 제안 사후",
    "adamw_map": "C AdamW 점 추정",
    "exact_map": "C 격자 정확 MAP",
    "exact_posterior": "C 격자 정확 사후예측",
    "smc": "C 입자 256 사후예측",
}


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf8"))


def mean_groups(sequence):
    groups = list(sequence[0])
    assert all(set(r) == set(groups) for r in sequence)
    return {g: float(np.mean([r[g] for r in sequence])) for g in groups}


def compare(a, b):
    assert set(a) == set(b)
    x, y = np.array([[r[g] for g in b] for r in (a, b)])
    ids = np.random.default_rng(421330).integers(len(x), size=(10000, len(x)))
    ci = np.quantile(100*(1-x[ids].mean(1)/y[ids].mean(1)), [.025, .975])
    return 100*(1-x.mean()/y.mean()), ci


def check_decisions():
    # A single RGB can have two possible depth states and two sensor outcomes.
    prior = np.array([.6, .4])
    observation = np.array([[.9, .1], [.2, .8]])
    joint = prior[:, None]*observation
    mass = joint.sum(axis=0)
    teacher = joint/mass
    np.testing.assert_allclose(teacher @ mass, prior, atol=1e-14)
    entropy = -(prior*np.log(prior)).sum()
    np.testing.assert_allclose(-(teacher*np.log(prior[:, None])).sum(axis=0) @ mass, entropy, atol=1e-14)

    # One fixed comparison distribution for the KL update and the cumulative bound.
    q = np.array([.3, .7])
    losses = np.array([[.2, .9], [.9, .1], [.5, .6]])
    p = prior.copy()
    val = 0.
    for loss in losses:
        unnormalized = p*np.exp(-loss)
        c = unnormalized.sum()
        post = unnormalized/c
        np.testing.assert_allclose(q @ loss + q @ np.log(q/p),
                                   q @ np.log(q/post)-np.log(c), atol=1e-14)
        val -= np.log(c)
        p = post
    np.testing.assert_allclose(val, -np.log(prior @ np.exp(-losses.sum(axis=0))), atol=1e-14)
    assert val <= q @ losses.sum(axis=0) + q @ np.log(q/prior)


def make(name="results.json"):
    check_decisions()
    result = load(name)
    datasets = list(result["baseline"])
    runs = result["loss_and_teacher"]
    c = result["parameter"]
    arms = {}
    for a in TRAINED:
        if a in runs[0]["models"]:
            arms[a] = {d: mean_groups([r["models"][a][d] for r in runs]) for d in datasets}
        elif a == "smc":
            arms[a] = {d: mean_groups([r["results"][d] for r in c["runs"]]) for d in datasets}
        else:
            arms[a] = c[a]

    yield "외부 지표는 **표본 격자 AbsRel이며 작을수록 좋다**. 영상마다 점 평균을 내고 그룹마다 영상 평균을 냈다. 세 시드를 쓰는 팔은 그룹 안에서 시드를 평균했다."
    yield ""
    yield "| 구분 | 그룹 | 영상 | 점 |"
    yield "|---|---:|---:|---:|"
    for name, d in result["sizes"].items():
        yield f"| {SETS.get(name, name)} | {len(d['groups'])} | {d['frames']} | {d['points']:,} |"
    yield ""
    yield "| 비교군 | " + " | ".join(SETS[k] for k in datasets) + " |"
    yield "|---|" + "---:|" * len(datasets)
    ordered = {"Depth Pro 2-S (baseline)": result["baseline"], **{TRAINED[k]: v for k, v in arms.items()}}
    for k, v in ordered.items():
        yield f"| {k} | " + " | ".join(f"{np.mean(list(v[d].values())):.6f}" for d in datasets) + " |"
    yield ""
    yield "| 비교 | " + " | ".join(SETS[k] for k in datasets) + " |"
    yield "|---|" + "---:|" * len(datasets)
    for arm, ref in (("loss_defensive", "loss_exact"), ("teacher_exact", "hard"), ("teacher_exact", "teacher_hard"),
                     ("exact_posterior", "baseline"), ("smc", "exact_posterior")):
        base = result["baseline"] if ref == "baseline" else arms[ref]
        contrast = []
        for d in datasets:
            delta, (lo, hi) = compare(arms[arm][d], base[d])
            contrast.append(f"{delta:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
        yield f"| {TRAINED[arm]} / {TRAINED.get(ref, ref)} | " + " | ".join(contrast) + " |"
    yield ""
    yield "| A 우도 계산 | 후보별 평균 로그우도 절대오차 | RGB 사전으로 주변화한 로그 증거 절대오차 | 계산 시간(s) |"
    yield "|---|---:|---:|---:|"
    for lab, key, marginal in (("정확", "0", None), ("부트스트랩 16", "16", "pf16"),
                               ("부트스트랩 64", "64", "pf64"), ("보호 제안 64", "defensive64", "defensive64")):
        exact_error = np.mean([r["errors"][key] for r in runs])
        evidence_error = np.mean([r["evidence_errors"][marginal] for r in runs]) if marginal is not None else 0.
        seconds = np.mean([r["times"][key] for r in runs])
        yield f"| {lab} | {exact_error:.6g} | {evidence_error:.6g} | {seconds:.4f} |"
    yield ""
    yield "| B 보류 Middlebury 교사의 관측 사용 점수 | 로그손실 ↓ | 표본 AbsRel ↓ |"
    yield "|---|---:|---:|"
    for lab, key in (("실사 중앙값", "direct_median"), ("독립 사후", "iid"), ("정확 순차 사후", "exact"),
                     ("부트스트랩 사후", "pf"), ("보호 제안 사후", "defensive")):
        seq = [r["teachers"][key] for r in runs]
        log = "—" if seq[0]["nll"] is None else f"{np.mean([r['nll'] for r in seq]):.6f}"
        absrel = np.mean([np.mean(list(r["absrel"].values())) for r in seq])
        yield f"| {lab} | {log} | {absrel:.6f} |"
    yield ""
    math = result["math"]
    yield (f"열거한 A 수치 예제의 Jensen 간극은 {math['jensen_gap']:.6f}다. 첫 로짓 경사는 "
           f"기대 입자 손실 {math['sampled_gradient'][0]:+.6f}, 정확 손실 {math['exact_gradient'][0]:+.6f}였다.")
    yield (f"C 입자 256 최종 고유 좌표는 {min(r['unique'] for r in c['runs'])}–{max(r['unique'] for r in c['runs'])}개였다. "
           f"최소 ESS는 {min(min(r['ess']) for r in c['runs']):.2f}였다. "
           f"정확 MAP는 {c['map'][0]:+.4f}, {c['map'][1]:+.4f}이고 AdamW 좌표는 {c['adamw'][0]:+.4f}, {c['adamw'][1]:+.4f}였다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(make(sys.argv[1] if len(sys.argv) > 1 else "results.json")))
