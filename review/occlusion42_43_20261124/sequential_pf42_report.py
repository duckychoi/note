"""Regenerate the real-depth comparison from per-image scores and the train ledger."""
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "research-sources/encoder42_eval/sequential-pf42"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf8"))


def display():
    data = read("training.json")["history"]
    manifest = read("manifest.json")
    order = manifest["order"]
    assert [[step["kind"], step["index"]] for step in data] == order
    real = [step for step in data if step["kind"] == "r"]
    synth = [step for step in data if step["kind"] == "s"]
    arms = ("frozen", "synthetic", "direct", "static", "exact", "particle", "nyu_full")
    updates = {arm: sum(arm in step for step in data) for arm in arms[1:-1]}
    assert updates == dict(synthetic=len(synth), **{a: len(data) for a in arms[2:-1]})
    assert len({step["index"] for step in real}) == len(real)
    assert all(step["samples"] <= 32 for step in real)
    math = read("numerical.json")
    gradient = read("gradient-check.json")
    lines = [
        "# #42 순차 입자 학습 실험",
        "",
        f"Hypersim 365장면에서 {len(synth):,}장, NYU 훈련 샤드에서 {len(real):,}장과 "
        f"희소 거리 {sum(step['samples'] for step in real):,}개를 사용했다. "
        "네 실사 관측 조건의 사진, 위치, 거리값, 갱신 순서를 맞췄다. 실사 정답 깊이의 유효성 마스크에서 표집했다. "
        f"실사 사용 조건은 각 {len(data):,}회, 합성 D2는 {len(synth):,}회 갱신했다. "
        f"이 중 합성 {sum(step['samples'] == 0 for step in synth)}장은 사용 거리 범위에 정답이 없어서 자료 손실이 0이었다. "
        f"유효 관측이 있는 합성 장면은 {len({manifest['synthetic'][x['index']][0] for x in synth if x['samples']})}개다. "
        "고정 시작점은 이전에 Hypersim 전체 59,543장을 사용해 학습했다. "
        "전체 NYU 지도 모델은 이전의 47,584장 학습과 다른 초기화·예산을 쓰는 참고선이다. "
        "이 모델은 장면 내 공통 초점 추정기를 사용해 다른 모델과 같은 단안 입력으로 평가했다.",
        "",
        f"숫자 검산에서 동일 상태·우도의 고정 혼합 {math['static']:.8f}과 정확 전이 {math['exact']:.8f}가 달랐다. "
        f"조상 50,000개에서 PF 추정값은 {math['particle']:.8f}였다. 본 학습은 64개다.",
        f"독립 수치 검산의 전이 로짓 경사 최대 오차는 {gradient['gradient_max_abs_error']:.3e}다. "
        f"{gradient['trials']:,}번의 64입자 실험에서 정확 결합 우도 {gradient['exact_joint']:.8f}, "
        f"입자 우도 평균 {gradient['particle_joint_mean']:.8f}, 평균의 표준오차 {gradient['standard_error']:.8f}였다.",
        "",
        "[전이의 증명](SEQUENTIAL_PF42_PROTOCOL.md) · [학습 경사와 표본 근사](SEQUENTIAL_PF42_DERIVATION.md) · "
        "[실행기](sequential_pf42.py) · [장면별 수치·체크포인트](../../../research-sources/encoder42_eval/sequential-pf42)",
        "",
        "## 평가 결과",
        "",
        "| 자료 | 영상 / 채점 그룹 | 동결 D2 | 합성 D2 | 희소 직접 | 고정 혼합 | 순차 정확 | 입자 학습 | 전체 NYU 지도 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = dict(hammer="HAMMER", diode="DIODE 실내", ibims="iBims-1", nyu="NYU 평가 분할")
    reports = {name: read(name + ".json") for name in labels}
    for name, label in labels.items():
        report = reports[name]
        assert report["n"] == len(report["rows"])
        for arm in arms:
            overall = np.mean([group[arm] for group in report["groups"].values()])
            np.testing.assert_allclose(overall, report["means"][arm])
        cells = " | ".join(f"{report['means'][arm]:.6f}" for arm in arms)
        lines.append(f"| {label} | {report['n']} / {len(report['groups'])} | {cells} |")
    lines.extend([
        "",
        "| 자료 | 동결 D2 대비 | 합성 D2 대비 | 희소 직접 대비 | 고정 혼합 대비 | 순차 정확 대비 | 전체 NYU 지도 대비 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, label in labels.items():
        info = reports[name]["compare"]

        def format_percent(k):
            score, (lo, hi) = info[k]["relative"], info[k]["interval"]
            return f"{score:+.3f}%" if name == "nyu" else f"{score:+.3f}% [{lo:+.3f}, {hi:+.3f}]"

        lines.append("| " + label + " | " + " | ".join(format_percent(k) for k in arms if k != "particle") + " |")
    lines.extend(["", "| 전이만 추가한 효과 | 고정 혼합 → 정확 순차 AbsRel | 상대 감소 |",
                  "|---|---:|---:|"])
    for name, label in labels.items():
        values = reports[name]["means"]
        lines.append(f"| {label} | {values['static']:.6f} → {values['exact']:.6f} | "
                     f"{100*(1-values['exact']/values['static']):+.3f}% |")
    if all((OUT / f"dense-{name}.json").exists() for name in labels):
        dense = read("dense-training.json")["records"]
        assert [[v["kind"], v["index"]] for v in dense] == order
        real_depth_count = sum(v["valid"] for v in dense if v["kind"] == "r")
        lines.extend(["", "## 영상 수를 맞춘 밀집 실사 감독", "",
                      f"같은 2,190회와 같은 실사 730장에 밀집 지도학습을 적용했다. "
                      f"4픽셀 격자에서 유효한 실사 거리 {real_depth_count:,}개를 주었다. "
                      "[추가 대조 규약](SEQUENTIAL_PF42_DENSE_CONTROL.md)", "",
                      "| 자료 | 밀집 지도 | 입자 학습 | 밀집 지도 대비 입자 상대 감소 |",
                      "|---|---:|---:|---:|"])
        for name, label in labels.items():
            result = read(f"dense-{name}.json")
            np.testing.assert_allclose(result["means"]["particle"], reports[name]["means"]["particle"])
            gain, (lo, hi) = result["comparison"]["relative"], result["comparison"]["interval"]
            uncertainty = "" if name == "nyu" else f" [{lo:+.3f}, {hi:+.3f}]"
            lines.append(f"| {label} | {result['means']['dense']:.6f} | {result['means']['particle']:.6f} | "
                         f"{gain:+.3f}%{uncertainty} |")
    lines.extend([
        "",
        "양수는 입자 학습의 AbsRel 감소다. 대괄호는 같은 장면 묶음을 10,000번 짝지어 재표집한 95% 구간이다. "
        "그룹이 적은 자료의 구간은 장면 간 변동을 제한적으로 보여 준다. "
        "NYU의 배포 아카이브에서는 평가 분할의 프레임 순서가 섞였고 채점기에 장면 식별자가 없어 구간을 표시하지 않는다. "
        "해당 평가 분할은 전체 NYU 감독 참고선의 체크포인트 선택에도 앞서 사용되었다. 한 시드의 반복 시험이다.",
        "",
        "## 구현과 주장 범위",
        "",
        "- 배포에는 관측을 주지 않았다. 입자를 쓰는 것은 훈련 경로다. 배포는 RGB 한 장에서 9개 전이를 정확히 합산한다.",
        "- 네 영역 점수로 전체 첫 분포를 만든다. 이후 전이는 목적 영역 점수와 고정 거리 벌점이다. 공간 순서를 사용했다. 카메라 시간 전이나 새 입자 상태의 생성 연산을 검증한 시험이 아니다.",
        "- 합성의 보조 학습도 표집 깊이의 가우스 주변우도다. 별도 상태 교차엔트로피를 쓰지 않았다. 순차 음의 로그우도는 유효한 점 개수로 나눈 뒤 .5로 가중했다. 마지막 4블록은 학습률 3e−6, 점수층은 5e−4를 사용했다.",
        "- 직접 감독은 후보 로그 깊이의 가중 평균에 제곱오차를 적용한다. 모든 후보형 모델의 평가는 AbsRel의 가중 중앙값이다. 합성 D2 대조는 표준 지도 손실·영역 헤드 없음·실사 관측 없음의 기준이다. 입자 표집의 추가 효과는 정확 전이와의 대조에 한정된다.",
        "- 기존 규약의 ‘구조적 교차엔트로피’ 표기는 실행 코드를 설명하지 않는다. 코드에 맞춘 합성 보조항의 정의는 위와 같다.",
        "",
        "[NYU RGB-D 원자료](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) · [사용한 NYU 배포 분할](https://huggingface.co/datasets/adams-story/nyu-depthv2-wds) · [정확 경사 및 64입자 검산 코드](sequential_pf42_verify.py) · [독립 검수](codex_sequential_pf42_최종_sol.txt)",
        "",
    ])
    (HERE / "SEQUENTIAL_PF42_RESULT.md").write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    display()
