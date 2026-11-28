"""Regenerate the numerical report from saved per-image rows."""
import json
import numpy as np

from parallax42_data import GRID, OUT, SEED, robust_delta


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf8"))


def verify_bound():
    rng = np.random.default_rng(SEED)
    lo = rng.uniform(.01, 4., size=512)
    hi = lo + rng.uniform(.0, 3., size=lo.shape)
    k = rng.uniform(GRID.min(), GRID.max(), size=lo.shape)
    exact = robust_delta(k, lo, hi)
    t = lo[:, None] + (hi-lo)[:, None]*np.linspace(0, 1, 4097)
    nearest = (abs(t-1)-abs(k[:, None]*t-1)).min(axis=1)
    gap = nearest-exact
    assert (gap >= -1e-12).all()
    assert (gap <= (1+k)*(hi-lo)/8192 + 1e-12).all()
    np.testing.assert_allclose(robust_delta(1.1, np.array([2/3.2]), np.array([2/2.8])), [.0625])
    np.testing.assert_allclose(robust_delta(.9, np.array([2/1.2]), np.array([2/.8])), [1/6])
    return len(lo), float(gap.max())


def audit(name):
    rows = read(name)["rows"]
    used = [r for r in rows if r["bound"] > 0]
    groups = {r["group"] for r in rows}
    out = f"{len(groups)} | {len(rows)} | {sum(r['inside'] for r in rows):,}/{sum(r['valid'] for r in rows):,} "
    out += f"| {sum(r['all_inside'] for r in rows)} | {len(used)}"
    for direction in (1, -1):
        s = [r for r in used if np.sign(r["robust"]-1) == direction]
        same = sum(np.sign(r["full_oracle"]-1) == direction for r in s)
        gains = sum(r["full_robust"] > 0 for r in s)
        out += f" | {same}/{len(s)} | {gains}/{len(s)}"
    if any(r["all_inside"] and r["support_robust"] + 1e-9 < r["bound"] for r in rows):
        raise ValueError("Guaranty violated on a verified interval")
    return out


def report():
    training = read("training.json")["history"]
    stereo = [r for r in training if r["kind"] == "r"]
    fit = read("real-teachers.json")
    assert len(training) == 2190 and len(stereo) == 730
    mismatched = {r["i"] for r in fit["rows"] if r["robust"] != GRID[np.argmin(r["risk"])]}
    n, gap = verify_bound()
    yield f"구간 최솟값을 {n}개 무작위 사례에서 4,097점 격자와 대조했다. 격자값과의 최대 차는 {gap:.3g}다."
    yield ""
    yield (f"각 모델을 {len(training):,}회 갱신했다. 혼합군의 합성 슬롯은 1,460개, 실사 슬롯은 {len(stereo)}개다. "
           f"실사 입력은 {len({r['stereo'] for r in stereo})}장, {len(set(r['group'] for r in fit['rows']))}개 계열이다. "
           f"고유 사진에서 고른 시차점은 {sum(r['n'] for r in fit['rows']):,}개, "
           f"반복을 포함한 관측 사용 횟수는 {sum(r['stereo_points'] for r in stereo):,}회다. "
           f"구간 교사가 배율을 변경한 슬롯은 {sum(r['robust_changed'] for r in stereo)}/730개다. "
           f"학습 자료에서 고른 고정 배율은 {fit['fixed']:.6f}다.")
    yield (f"같은 9개 배율 후보를 사용할 때 점 교사와 구간 교사가 다른 배율을 고른 비율은 "
           f"{len(mismatched)}/{len(fit['rows'])} 고유 입력과 {sum(r['stereo'] in mismatched for r in stereo)}/730 반복 슬롯이다.")
    yield ""
    yield "| 실사 관측 검증 | 계열 | 영상 | 2 px 이내 점 | 모든 점 포함 영상 | 수정 제안 | 증가 부호 일치 | 증가 후 개선 | 감소 부호 일치 | 감소 후 개선 |"
    yield "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for title, f in (("Middlebury 보류", "held-stereo.json"), ("ETH3D 별도 촬영", "eth3d-audit.json")):
        yield f"| {title} | {audit(f)} |"
    yield ""
    yield "| 외부 평가 | 공개 S | 동결 D2 | 추가 합성 | 시차 점 | 시차 구간 | 고정 배율 | 전체 NYU 감독 |"
    yield "|---|---:|---:|---:|---:|---:|---:|---:|"
    for title, d in (("HAMMER", "hammer"), ("DIODE 실내", "diode"), ("iBims-1", "ibims")):
        result = read(f"{d}.json")
        public = read(f"official-{d}.json")
        assert set(result["groups"]) == set(public["groups"])
        ref = public["means"]["public_s"]
        m = result["means"]
        names = ("frozen", "synthetic_replay", "stereo_point", "stereo_robust", "fixed_stereo", "nyu_full")
        yield f"| {title} | {ref:.6f} | " + " | ".join(f"{m[a]:.6f}" for a in names) + " |"
    yield ""
    yield "| 외부 평가 | 시차 구간 vs 동결 D2 [95%] | 시차 점 vs 동결 D2 [95%] | 시차 구간 vs 시차 점 [95%] | 고정 배율 vs 동결 D2 [95%] | 정답 배율 진단 |"
    yield "|---|---:|---:|---:|---:|---:|"
    for title, d in (("HAMMER", "hammer"), ("DIODE 실내", "diode"), ("iBims-1", "ibims")):
        result = read(f"{d}.json")
        c = result["comparisons"]
        def fmt(value):
            lo, hi = value["interval"]
            return f"{value['relative']:+.3f}% [{lo:+.3f}, {hi:+.3f}]"
        def diagnostics(key):
            groups = {r["group"] for r in result["rows"]}
            return np.mean([np.mean([r[key] for r in result["rows"] if r["group"] == g]) for g in groups])
        frozen = np.array([g["frozen"] for g in result["groups"].values()])
        fixed = np.array([g["fixed_stereo"] for g in result["groups"].values()])
        idx = np.random.default_rng(421127).integers(len(frozen), size=(10000, len(frozen)))
        draws = 100 * (1-fixed[idx].mean(1)/frozen[idx].mean(1))
        control = dict(relative=100 * (1-fixed.mean()/frozen.mean()), interval=np.quantile(draws, [.025, .975]))
        entries = [fmt(c["stereo_robust"]["frozen"]), fmt(c["stereo_point"]["frozen"]),
                   fmt(c["stereo_robust"]["stereo_point"]), fmt(control), f"{diagnostics('oracle_absrel'):.6f}"]
        yield f"| {title} | " + " | ".join(entries) + " |"
    yield ""
    yield "| 외부 평가 | 구간 학생 배율 중앙값 | 정답 배율 중앙값 | 증가·감소 방향 일치 |"
    yield "|---|---:|---:|---:|"
    for title, d in (("HAMMER", "hammer"), ("DIODE 실내", "diode"), ("iBims-1", "ibims")):
        rows = read(f"{d}.json")["rows"]
        k = np.array([r["scales"]["stereo_robust"] for r in rows])
        oracle = np.array([r["oracle"] for r in rows])
        same = (np.sign(k-1) == np.sign(oracle-1)).sum()
        yield f"| {title} | {np.median(k):.4f} | {np.median(oracle):.4f} | {same}/{len(k)} |"


if __name__ == "__main__":
    print("\n".join(report()))
