"""Regenerate #42 diagnostics and evaluation from the per-image ledgers."""
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval"
OUT = SOURCE / "preserve42"
HERE = Path(__file__).resolve().parent
NAMES = {"hammer": "HAMMER", "diode": "DIODE 실내", "ibims": "iBims-1", "nyu": "NYU 평가"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def numeric():
    rng = np.random.default_rng(421129)
    base, truth = rng.uniform(.3, 30, (2, 10000))
    tau = .22
    candidate = base * np.exp(rng.uniform(-tau, tau, len(base)))
    mixture = .5 * (base + candidate)
    risk = lambda x: np.abs(x / truth - 1).mean()
    convex_slack = .5 * (risk(base) + risk(candidate)) - risk(mixture)
    cap_slack = np.expm1(tau) * (1 + risk(base)) - (risk(candidate) - risk(base))
    assert convex_slack >= 0 and cap_slack >= 0
    lower, upper = .5 * (1 + np.exp(np.array([-tau, tau])))
    assert np.all((mixture / base >= lower) & (mixture / base <= upper))
    assert abs((.5 * (2. + 2.2)) / 2. - 1) > 0
    (OUT / "theory-check.json").write_text(json.dumps(dict(n=len(base), convex_slack=convex_slack,
        cap_slack=cap_slack, output_ratio=[float(lower), float(upper)]), indent=2), encoding="utf8")


def main():
    numeric()
    rows = {tag: read(OUT / f"diagnose-{tag}.json") for tag in NAMES}
    doc = [
        "# #42 동결 D2 대비 배율·형상과 보존형 학습",
        "",
        "로그 배율 제곱과 평균을 제거한 로그 형상 제곱, 참 깊이비가 1.05 이내인 8픽셀 점쌍의 로그 깊이비 "
        "제곱오차를 같은 유효 영역에서 집계했다. 표에는 각 통계의 제곱근을 표시한다. "
        "AbsRel 기여는 배율과 형상을 두 순서로 바꿨을 때의 차이를 평균한다.",
        "",
        "| 자료 | 로그 배율 RMSE | 배율 제거 형상 RMSE | 국소 깊이비 RMSE | 배율 기여 (%p) | 형상 기여 (%p) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag, label in NAMES.items():
        row = rows[tag]["means"]
        np.testing.assert_allclose(row["delta_scale"] + row["delta_shape"], row["old"] - row["frozen"], atol=1e-10)
        arrows = " | ".join(f"{row['frozen_'+k]**.5:.6f} → {row['old_'+k]**.5:.6f}"
                            for k in ("scale2", "shape2", "surface2"))
        doc.append(f"| {label} | {arrows} | {100*row['delta_scale']:+.3f} | {100*row['delta_shape']:+.3f} |")
    doc.extend(["", "| 자료 | 동결 AbsRel | 기존 입자 배율만 교환 | 기존 형상만 교환 | 기존 입자 전체 |",
                "|---|---:|---:|---:|---:|"])
    for tag, label in NAMES.items():
        row = rows[tag]["means"]
        doc.append(f"| {label} | {row['frozen']:.6f} | {row['scale_from_old']:.6f} | "
                   f"{row['shape_from_old']:.6f} | {row['old']:.6f} |")
    doc.extend(["", "두 표는 정답을 사용한 진단이다. 학습 배치의 어떤 원인이 이 변화를 일으켰는지는 이 분해로 정해지지 않는다.",
                "", "[정리·규약](PRESERVE42_PROTOCOL.md) · [점별 보존 조건](PRESERVE42_GUARANTEE.md) · "
                "[실행기](preserve42.py) · [원자료·가중치](../../../research-sources/encoder42_eval/preserve42)"])
    if all((OUT / f"evaluate-{tag}.json").is_file() for tag in NAMES):
        files = {tag: read(OUT / f"evaluate-{tag}.json") for tag in NAMES}
        ledger = read(OUT / "training.json")["history"]
        manifest = read(SOURCE / "sequential-pf42/manifest.json")
        assert [[entry["kind"], entry["index"]] for entry in ledger] == manifest["order"]
        assert all({"bounded_direct", "bounded_particle"} <= entry.keys() for entry in ledger)
        real_points = sum(entry["samples"] for entry in ledger if entry["kind"] == "r")
        doc.extend([
            "", "## 같은 학습 예산의 재검증", "",
            f"Hypersim 1,460장과 NYU 730장, 희소 관측 {real_points:,}점, 2,190회의 갱신을 이전 순차 시험과 맞췄다. "
            "이번 두 방법의 직접 감독, 합성 감독, 입력이 같고, 순차 팔에만 입자 우도가 추가된다. "
            "동일 D2 초기값에서 마지막 인코더 4블록을 갱신했다. RGB 한 장과 동일 초점 추정기로 평가했다. "
            "동결과 적응 경로가 인코더 앞 8블록을 공유하고 뒤 4블록과 깊이 디코더를 각각 실행한다. "
            "이미지·관측·갱신 수는 동일하지만 이전 실험과 연산량이 같지는 않다.",
            "",
            "| 자료 | 동결 D2 | 기존 PF | 제한 직접 감독 | 제한 PF | 동결 대비 제한 PF 감소 [95%] |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for tag, label in NAMES.items():
            report = files[tag]
            m = report["means"]
            np.testing.assert_allclose(m["frozen"], rows[tag]["means"]["frozen"], atol=2e-6)
            c = report["comparison"]["frozen"]
            lo, hi = c["interval"]
            ci = f" [{lo:+.3f}, {hi:+.3f}]" if tag != "nyu" else ""
            doc.append(f"| {label} | {m['frozen']:.6f} | {m['old_pf']:.6f} | {m['bounded_direct']:.6f} | "
                       f"{m['bounded_particle']:.6f} | {c['relative']:+.3f}%{ci} |")
        doc.extend(["", "| 자료 | 배율 보정만 | 형상 보정만 | 형상만 vs 동결 상대 감소 [95%] | 동일 영상 밀집 감독 | 전체 NYU 지도 |",
                    "|---|---:|---:|---:|---:|---:|"])
        for tag, label in NAMES.items():
            m = files[tag]["means"]
            groups = list(files[tag]["groups"].values())
            idx = np.random.default_rng(421127).integers(len(groups), size=(10000, len(groups)))
            scores = np.array([[g["frozen"], g["shape_only"]] for g in groups])
            draws = scores[idx].mean(1)
            low, high = np.quantile(100 * (1 - draws[:, 1] / draws[:, 0]), [.025, .975])
            ci = "" if tag == "nyu" else f" [{low:+.3f}, {high:+.3f}]"
            change = 100 * (1 - m["shape_only"] / m["frozen"])
            doc.append(f"| {label} | {m['scale_only']:.6f} | {m['shape_only']:.6f} | {change:+.3f}%{ci} | "
                       f"{m['dense']:.6f} | {m['nyu_full']:.6f} |")
        doc.extend(["", "| 자료 | 제한 PF 로그 배율 RMSE | 배율 제거 형상 RMSE | 국소 깊이비 RMSE |",
                    "|---|---:|---:|---:|"])
        for tag, label in NAMES.items():
            m = files[tag]["means"]
            formatted = " | ".join(f"{m['frozen_'+k]**.5:.6f} → {m['bounded_particle_'+k]**.5:.6f}"
                                   for k in ("scale2", "shape2", "surface2"))
            doc.append(f"| {label} | {formatted} |")
        doc.extend([
            "",
            "대괄호는 장면 계열 단위 짝 재표집이다. NYU는 장면 ID가 없어 구간을 표시하지 않았다. "
            "같은 영상의 밀집 기준은 같은 갱신 수에서 더 많은 실사 거리점을 봤다. 전체 NYU 모델은 별도의 초기화·학습 예산이다.",
            "",
            "두 팔은 고정 혼합 가중치와 배율·형상 보정 제한을 사용했다. 보존 보장의 입력인 독립 실사 보정 장면과 "
            "목표 분포 차이 상한은 확보되지 않았다. 현재 값은 경험적 반복 시험이다. 형상만 남기는 대조는 산술 평균 이후 "
            "로그 전체 평균을 다시 제거한다. 배율/형상 대조는 결합 출력의 정확한 AbsRel 가산 분해와 다르다. "
            "학습은 단일 시드이며 과거에 평가한 외부 자료를 재사용했다. 이전 PF 대비로는 변경 제한, 무관측 읽기와 "
            "직접 지도 목적이 함께 달라졌으므로 차이 전체를 어느 하나의 선택으로 귀속하지 않는다.",
            "",
            "[교차 검수](codex_preserve42_최종_sol.txt)",
        ])
    (HERE / "PRESERVE42_RESULT.md").write_text("\n".join(doc) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
