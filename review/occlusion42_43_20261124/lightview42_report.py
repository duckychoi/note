"""Independently recalculate scale-relative test summaries from recorded image scores."""
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "research-sources/encoder42_eval/lightview42"
SETS = (("hammer", "HAMMER"), ("diode", "DIODE 실내"), ("ibims", "iBims-1"), ("diode_outdoor", "DIODE 실외"))


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf8"))


def verify_excess():
    rng = np.random.default_rng(421131)
    lo = rng.uniform(.01, 4, 512)
    hi = lo+rng.uniform(.0, 3, 512)
    k = np.exp(rng.uniform(-.40, .20, size=(512, 1)))
    k0 = np.exp(-.20)
    points = np.column_stack((lo, hi, np.clip(1/k0, lo, hi)))
    exact = (abs(k*points-1)-abs(k0*points-1)).max(axis=-1)
    t = lo[:, None]+(hi-lo)[:, None]*np.linspace(0, 1, 4097)
    samples = (abs(k*t-1)-abs(k0*t-1)).max(axis=-1)
    delta = exact-samples
    assert np.all(delta >= -1e-12)
    assert np.all(delta <= (k.ravel()+k0)*(hi-lo)/8192+1e-12)
    return float(delta.max())


def result():
    teachers = read("teachers.json")
    history = read("training.json")["history"]
    before = teachers["original"]
    assert len(history) == 2190 and sum(x["kind"] == "s" for x in history) == 1460
    assert all(r["lightview_metric"] == 0 for r in history if r["kind"] == "s")
    yield f"구간 최악 오차는 512개 임의 구간에서 검산했다. 4,097점 촘촘한 탐색과의 최대 차는 {verify_excess():.3g}다."
    yield ""
    yield "| 이전 목표 | 업데이트 | 배율 증가 | 배율 감소 | 목표 중앙값 | 슬롯 경사 합 |"
    yield "|---|---:|---:|---:|---:|---:|"
    for kind, title in (("s", "Hypersim 절대 배율"), ("r", "Middlebury 시차 구간")):
        d = before[kind]
        yield f"| {title} | {d['n']} | {d['up']} | {d['down']} | {d['median']:.4f} | {d['gradient_sum']:+.3f} |"
    syn = [r["old_pred"] for r in teachers["synthetic"] if r["old_pred"] is not None]
    real = [r["old_pred"] for r in teachers["real"].values()]
    yield ""
    yield (f"이전 체크포인트의 배율 중앙값은 Hypersim 기록 {len(syn)}장에 {np.median(syn):.4f}, "
           f"Middlebury 학습 {len(real)}장에 {np.median(real):.4f}였다. "
           f"이번 확장 범위에서 실사 학습 사진으로 고른 상수는 {teachers['fixed']:.6f}다.")
    grid = np.exp(np.linspace(-.40, .20, 601))
    k0 = np.exp(-.20)
    best = []
    for row in teachers["real"].values():
        t = np.array(row["candidates"])
        values = (abs(grid[:, None, None]*t-1)-abs(k0*t-1)).max(axis=-1).mean(axis=-1)
        best.append(grid[np.argmin(values)])
    best = np.array(best)
    down = (best < k0-1e-10).sum()
    up = (best > k0+1e-10).sum()
    yield (f"같은 확장 범위의 실사 교사 도출값에서 {down}장은 0.819보다 더 작은 배율, {up}장은 더 큰 배율, "
           f"{len(best)-down-up}장은 같은 배율을 선호했다. 최소 허용 배율에 걸린 영상은 {(best == grid[0]).sum()}장이다.")
    yield ""
    yield "| 외부 실사 | 영상/그룹 | Depth Pro 2-S (baseline) | Hypersim 대조 | 고정 0.819 | 새 상수 | LightView-Depth (이전 시차) | 합성 절대 배율 포함 | LightView-Depth |"
    yield "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    for dataset, title in SETS:
        value = read(f"{dataset}.json")
        m = value["means"]
        labels = ("apple_s", "hypersim", "fixed_0819", "train_fixed", "previous_lightview", "mixed_metric", "lightview")
        values = [f"{m[a]:.6f}" if a in m else "미측정" for a in labels]
        yield f"| {title} | {value['n']}/{len(value['groups'])} | " + " | ".join(values) + " |"
        groups = {}
        for row in value["rows"]:
            groups.setdefault(row["group"], []).append(row["scores"])
        for key, score in m.items():
            calc = np.mean([np.mean([row[key] for row in group]) for group in groups.values()])
            np.testing.assert_allclose(score, calc, atol=1e-12)
    yield ""
    yield "| 외부 실사 | Hypersim 대조 대비 감소 | 고정 0.819 대비 감소 | 새 상수 대비 감소 | 상대 학습의 배율 범위 | 혼합 학습의 배율 중앙값 |"
    yield "|---|---:|---:|---:|---:|---:|"
    for dataset, title in SETS:
        value = read(f"{dataset}.json")
        cmp = value["comparisons"]["lightview"]
        def outcome(ref):
            c = cmp[ref]
            lo, hi = c["interval"]
            precision = 4 if ref == "train_fixed" else 2
            return f"{c['relative']:+.{precision}f}% [{lo:+.{precision}f}, {hi:+.{precision}f}]"
        scale = np.array([r["scales"]["lightview"] for r in value["rows"]])
        mixed = np.median([r["scales"]["mixed_metric"] for r in value["rows"]])
        yield f"| {title} | " + " | ".join(outcome(ref) for ref in ("hypersim", "fixed_0819", "train_fixed")) + \
              f" | {scale.min():.5f}–{scale.max():.5f} | {mixed:.4f} |"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(result()))
