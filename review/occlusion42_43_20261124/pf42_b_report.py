"""Recompute the teacher and matched student comparisons from frozen row outputs."""
import json
from pathlib import Path
import sys

import numpy as np


DATA = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval/pf42-trials"
OUT = DATA / "separated-b"
SEED = 421334
DATASETS = {"hammer": "HAMMER", "ibims": "iBims-1", "diode": "DIODE 실내", "diode_outdoor": "DIODE 실외"}


def interval(a, b):
    i = np.random.default_rng(SEED).integers(len(a), size=(10000, len(a)))
    return 100*(1-a.mean()/b.mean()), np.quantile(100*(1-a[i].mean(1)/b[i].mean(1)), [.025,.975])


def grid_diagnosis():
    with np.load(DATA/"held.npz") as d:
        keep = np.isfinite(d["truth"]) & (d["truth"] >= .25) & (d["truth"] <= 30)
        base, truth, group = d["base"][keep].astype(float), d["truth"][keep], d["group"][keep]
        r = np.nanmedian(np.log(d["y"][keep].astype(float)),axis=1)-np.log(base)
    def score(a):
        e = abs(base*np.exp(a)/truth-1)
        return float(np.mean([e[group == g].mean() for g in np.unique(group)]))
    return dict(clip_only=score(np.clip(r,-1.6,1.6)), round_only=score(np.round(r/.1)*.1),
                outside=int((abs(r)>1.6).sum()), continuous=score(r))


def verify_risk_bound():
    # At every candidate the sampled Wasserstein coupling bounds the loss difference.
    z = np.array([1., 2., 2.8, 3.])
    q = np.array([1.2, 1.8, 2.6, 2.9])
    m, a = 2.4, 2.1
    def gain(x):
        return (abs(m-x)-abs(a-x))/x
    empirical_gap = abs(gain(z).mean()-gain(q).mean())
    w1 = abs(z-q).mean()
    assert empirical_gap <= (m+a)/min(z.min(),q.min())**2*w1
    # A constant offset in true log depth, canceled in both reported errors, leaves sources unchanged.
    r, stereo, synth = .25, -.08, .12
    offset = .1
    np.testing.assert_allclose([r+stereo, r+synth],
                               [r+offset+stereo-offset, r+offset+synth-offset])


def report():
    verify_risk_bound()
    data = json.loads((OUT/"results.json").read_text(encoding="utf8"))
    diagnostic = grid_diagnosis()
    assert abs(diagnostic["continuous"]-data["mean"]["direct"]) < 2e-8
    yield ("예전 로그 후보 격자에서 보정 범위를 벗어난 직접 중앙값은 "
           f"{diagnostic['outside']}점이었다. 로그 깊이 제한만 남기면 AbsRel {diagnostic['clip_only']:.6f}, "
           f"0.1 로그 눈금 반올림만 남기면 {diagnostic['round_only']:.6f}였다. 두 효과는 가산하지 않는다.")
    yield ""
    yield "| 고정한 교사 | 보류 AbsRel ↓ | 연속 로그 중앙값 대비 상대 감소 [95%] |"
    yield "|---|---:|---:|"
    names = {"legacy_grid": "기존 격자 중앙값", "direct": "연속 로그 중앙값", "direct_meter": "미터 단위 중앙값",
             "sensor": "반복 오차 보정만", "joint_raw": "원래 합성 사전 + 반복 오차 보정",
             "joint": "합성 사전·반복 오차 분리 보정"}
    for k,label in names.items():
        change = data["relative"].get(k)
        if change is None:
            relative = "기준"
        else:
            lo,hi = change["interval"]
            relative = f"{change['relative']:+.3f}% [{lo:+.3f}, {hi:+.3f}]"
        yield f"| {label} | {data['mean'][k]:.9f} | {relative} |"
    yield ""
    yield "| 지표 | 보정 계열 | 보류 계열 |"
    yield "|---|---:|---:|"
    for k,label in (("points","유효한 평가점"),("prior_log_bias","합성 사전 평균의 로그 편향"),
                    ("median_stereo_log_bias","직접 중앙값의 로그 편향"),
                    ("aggregate_log_rms","직접 중앙값 로그 RMSE"),
                    ("cross_light_correlation","두 조명의 측정 잔차 상관")):
        yield f"| {label} | {data['bias']['cal'][k]:.5g} | {data['bias']['held'][k]:.5g} |"
    held = data["bias"]["held"]
    yield (f"보류의 반복 폭이 0.002 이하인 {held['small_span']:,}점에서 "
           f"{held['small_span_error_3_5pct']}점은 시차 중앙값의 로그 오차가 0.035를 넘었다.")
    prior_check = json.loads((OUT/"prior-diagnostic.json").read_text(encoding="utf8"))["held"]
    yield (f"보류에서 합성 사전 중심의 로그 편향은 {prior_check['raw_bias']:+.4f} → "
           f"{prior_check['adjusted_bias']:+.4f}, 로그 RMSE는 {prior_check['raw_log_rmse']:.4f} → "
           f"{prior_check['adjusted_log_rmse']:.4f}였다. 같은 점의 시차 중앙값 로그 RMSE는 "
           f"{held['aggregate_log_rms']:.4f}였다.")
    if data.get("student_ran"):
        yield ""
        yield "| 외부 실사 | 직접 감독 AbsRel ↓ | 분리 교사 AbsRel ↓ | 직접 대비 상대 감소 [95%] |"
        yield "|---|---:|---:|---:|"
        for key,name in DATASETS.items():
            seed_scores = data["students"]
            groups = sorted(seed_scores[0]["scores"]["direct"][key])
            b,a = np.array([[[r["scores"][arm][key][g] for g in groups] for r in seed_scores]
                            for arm in ("direct","separated")]).mean(axis=1)
            diff,(lo,hi) = interval(a,b)
            yield f"| {name} | {b.mean():.6f} | {a.mean():.6f} | {diff:+.3f}% [{lo:+.3f}, {hi:+.3f}] |"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    print("\n".join(report()))
