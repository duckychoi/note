"""Rebuild the human-readable experiment report from per-frame measurements."""
import json
from pathlib import Path

import numpy as np

from light_encoder_score import summary
from light_encoder_study import ARMS, OUT, PREVIOUS, manifest


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def gain(ref, student):
    return 100*(1-student/ref)


def checked(path):
    report = load(path)
    rebuilt = summary(report["rows"])
    assert rebuilt["groups"] == report["groups"]
    assert rebuilt["means"] == report["means"]
    assert len({(r["key"],r.get("orientation")) for r in report["rows"]}) == report["n"]
    return report


def main():
    cal = load(OUT/"calibration.json")
    external = load(OUT/"external-recipes.json")
    studies = {s:checked(OUT/f"final-{s}.json") for s in ("synthetic","diode","ibims")}
    mid = checked(OUT/"middle-synthetic.json")
    assert all(studies[s]["checkpoint"] == studies["synthetic"]["checkpoint"] for s in studies)
    fit = [r for r in manifest() if r[0]=="fit"]
    nh = studies["synthetic"]
    names = {"diode":"DIODE 실내","ibims":"iBims-1","synthetic":"Hypersim 검증"}
    table = ["# #42·43 조명 관측을 실제 단안 깊이 인코더에 학습한 결과","",
             "2026-11-26. 정지 장면의 깊이 비와 미터 깊이를 평가했다. 동적 장애물과 움직이는 카메라의 추적은 별도 연구로 남긴다.","",
             f"출발 가중치는 Hypersim 59,543개 앵커로 두 패스를 진행한 Depth Pro 2-S D2 실험에서 합성 검증으로 선택한 92,000회 시점 모델이다. 이번에는 학습 분할 {len(set(r[1] for r in fit))}장면에서 {len(fit):,}장을 읽고 실제 유효 깊이가 있는 {nh['updates']:,}장에서 세 학생을 각각 갱신했다. 마지막 네 블록 {nh['parameters']:,}개 파라미터를 학습했다. 출력 공통 배율을 보존하려고 동일한 RGB의 앞쪽 여덟 블록을 공유하고, 원래와 학습된 뒤쪽 블록·디코더를 각각 통과시켰다. 초점도 RGB에서 추정했다. 합성 학습에는 렌더러의 카메라 계수와 깊이 정답을 사용했다.","",
             "## 주 결과","",
             "원본 해상도의 장면 가중 AbsRel. 낮을수록 좋다. iBims는 파일명으로 묶은 공간 유형을 균등 가중한다. 배율은 합성에서 이미 고정한 1.112972다.","",
             "| 자료 | 영상 / 그룹 | 공통 배율만 | 지도 계속학습 | 고정 게인 | 조건별 게인 | 조건별 vs 배율 | 조건별 vs 지도 | 조건별 vs 고정 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s,report in studies.items():
        m = report["means"]
        table.append(f"| {names[s]} | {report['n']} / {len(report['groups'])} | "+
                     " | ".join(f"{m[a]:.6f}" for a in ("base",*ARMS))+" | "+
                     " | ".join(f"{gain(m[a],m['conditional']):+.3f}%" for a in ("base","d2","fixed"))+" |")
    table += ["",f"합성 중간 {mid['updates']}회와 최종 {nh['updates']}회의 조건별 학생 AbsRel은 {mid['means']['conditional']:.6f} → {nh['means']['conditional']:.6f}였다. 사전에 지정한 마지막 체크포인트로 실사를 열었다. 한 시드·한 패스의 추가 적응 결과다. 실제 파라미터 수와 업데이트 수가 충분한 최종 학습 예산을 증명하지 않는다.","",
              "## 미터 배율과 깊이 형상의 분해","",
              "| 시험 | 공통 배율 → 조건별, 영상 평균 로그 오차의 제곱 | 공통 배율 → 조건별, 영상 내부 로그 오차의 분산 | 배율 1에서 공통 → 조건별 AbsRel |",
              "|---|---:|---:|---:|"]
    for s in ("diode","ibims"):
        m = studies[s]["means"]
        table.append(f"| {names[s]} | {m['base_scale_mse']:.6f} → {m['conditional_scale_mse']:.6f} | "
                     f"{m['base_shape_mse']:.6f} → {m['conditional_shape_mse']:.6f} | "
                     f"{m['base_k1']:.6f} → {m['conditional_k1']:.6f} |")
    drift = max(r["max_mean_log_update"] for r in studies.values())
    table += ["",f"전체 RGB에서 평균 로그 보정의 최대 절댓값은 {drift:.2g}였다. 위 표의 두 오차 항은 정답이 유효한 픽셀에서 계산해 배율 항이 소폭 달라질 수 있다. 모형 분할 전후 기준 역깊이 최대 차이는 첫 입력에서 {nh['official_max_error']:.2g}이었다.","",
              "## 강한 공개·실사 학습 참조","",
              "공개 체크포인트 본래 출력과 비교하기 위해 모든 모델에 배율 1을 쓴다. 같은 평가 이미지와 마스크로 재집계했다. NYU 실사 지도 모델은 실사 47,584장을 학습 풀로 사용한 별도 119,086회 실행에서 선택한 체크포인트다. 이 모델의 성적은 해당 실험의 참조값이다. 초기 공개 S 가중치에는 실사 선행 학습도 들어 있다.","",
              "| 시험 | 조건별 학생 | 기존 합성 D2 | 합성 SiLog | NYU 실사 지도 | 공개 Depth Pro 2-S | 공개 DA V2 Hypersim 소형 / 대형 |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("diode","ibims"):
        report = studies[s]
        ids = {r["key"] for r in report["rows"]}
        vals = []
        for label in ("silog","nyu_real"):
            ref = external[f"{label}-{s}"]
            assert ids == {r["key"] for r in ref["rows"]}
            vals.append(ref["means"][label+"_k1"])
        if s=="diode":
            folder = OUT.parent/"full-comparison-4243"
            apple = load(folder/"diode_frozen_scale_diagnostic.json")["rows"]["frozen"]
            depth_anything = load(folder/"diode_da2_hypersim_reference.json")
            def matched(rows):
                assert set(rows)==ids
                return summary([dict(key=r["key"],group=r["group"],scores={"value":rows[r["key"]]})
                                for r in report["rows"]])["means"]["value"]
            publ = f"{matched({k:v['rgb'][0] for k,v in apple.items()}):.6f}"
            da = " / ".join(f"{matched({k:v[0] for k,v in depth_anything[size]['rows'].items()}):.6f}"
                            for size in ("small","large"))
        else:
            published = load(OUT/"published-ibims.json")
            for ref in published.values():
                assert ids == {r["key"] for r in ref["rows"]}
            publ = f"{published['official']['means']['official_k1']:.6f}"
            da = " / ".join(f"{published[a]['means'][a+'_k1']:.6f}" for a in ("small","large"))
        m = report["means"]
        table.append(f"| {names[s]} | {m['conditional_k1']:.6f} | {m['base_k1']:.6f} | "
                     f"{vals[0]:.6f} | {vals[1]:.6f} | {publ} | {da} |")
    table += ["","## 어느 조명이 깊이 판단에 도움을 주었나","",
              f"합성 보정 {cal['scenes']}장면 {cal['frames']}장에서 맞춘 단일 게인은 {cal['fixed']:.4f}, 조건별 게인은 "+
              ", ".join(f"{k:.4f}" for k in cal["conditional"])+"였다. 관측량은 가로·세로로 16픽셀 떨어진 두 점의 로그 깊이 차이다. 두 점이 같은 물체인지 보장하지 않는다.","",
              "| 교사 진단 | 장면 계열 / 영상 방향 | 무보정 두 점 MSE | 두 사진 평균 개선 | 단일 게인 개선 | 조건별 개선 | 조건별 vs 단일 | 평가 정답으로 4개를 다시 맞춘 진단 vs 단일 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("synthetic","physical"):
        report = checked(OUT/f"teacher-{name}.json")
        m,diag = report["means"],report["diagnostic"]
        table.append(f"| {name} | {len(report['groups'])} / {report['n']} | {m['original']:.7f} | "+
                     " | ".join(f"{gain(m['original'],m[a]):+.3f}%" for a in ("half","fixed","conditional"))+
                     f" | {gain(m['fixed'],m['conditional']):+.3f}% | {gain(m['fixed'],diag['retrospective_risk']):+.3f}% |")
        assert abs(diag["uncorrected_risk"]-m["original"])<1e-7
    table += ["","Middlebury 실제 조명은 같은 카메라와 같은 셔터 시간의 14장면·11계열에서 조명 순서를 뒤집어 28방향을 평가했다. 같은 점과 가중치를 두 순서에 주면 최적 단일 게인이 0.5인 것은 대칭성의 결과다. 이 표의 MSE를 단안 미터 AbsRel과 직접 비교할 수 없다. 합성 감광은 원본 RGB 노출을 공간적으로 바꾼 것이며 실제 광원과 반사면을 재렌더링한 실험은 별도로 남아 있다.","",
              "## 두 사진을 넣는 순서가 만든 오차","",
              "주 시험 후 같은 게인과 장면에서 두 순서의 교사 출력을 평균했다. 이 방법은 정답을 보지 않고 두 순서 간 불일치에서 생기는 제곱오차를 정확히 제거한다. 표는 양방향 표본에서 계산했다.","",
              "| 자료 | 순서별 조건 게인 MSE | 순서 독립 조건 게인 MSE | 반반 평균 MSE | 원래 반반 평균과의 격차 중 제거된 비율 |",
              "|---|---:|---:|---:|---:|"]
    order = load(OUT/"order-diagnostic.json")["datasets"]
    for s,r in order.items():
        m = r["means"]
        assert np.isclose(m["ordered"]-m["symmetric"],m["removable"],atol=1e-12)
        table.append(f"| {s} | {m['ordered']:.8f} | {m['symmetric']:.8f} | {m['half']:.8f} | "
                     f"{100*m['removable']/(m['ordered']-m['half']):.1f}% |")
    table += ["","합성 184쌍·46장면과 실제 조명 14쌍·11계열을 모두 양방향으로 평가했다. 앞 절의 합성 표는 같은 184쌍을 지정된 한 방향으로 평가했다. 두 자료 모두 순서에 따라 달라지는 보정이 격차의 일부를 설명했다. 그 오차를 제거한 뒤에도 밝기 조건별 선택은 두 사진의 깊이 차 예측을 반반 섞은 결과보다 오차가 컸다. 이 실험의 다음 검증 대상은 실제로 불리한 관측을 골라낼 수 있는 단서다. 실제 물체 경계, 재질과 광원 방향을 따로 조작한 장면에서 반반 평균 대비 이득이 있는지 먼저 측정한다.","",
              "## 재현","",
              "torch 환경에서 학습은 `light_encoder_study.py both`, 중단 후 재개는 `light_encoder_study.py train`, 채점은 `light_encoder_score.py middle final teachers references published`, 추가 진단은 `light_encoder_order.py`, 표 재생성은 `light_encoder_report.py`이다. 학습 순서와 감광은 장면·프레임에서 결정하며 최신 체크포인트에 세 모델과 각 옵티마이저 상태가 들어 있다.",
              "", "[이론 및 고정한 절차](LIGHT_ENCODER_PROTOCOL.md) · [학습 코드](light_encoder_study.py) · [단안 채점과 교사 진단](light_encoder_score.py) · [프레임 원자료·체크포인트](../../../research-sources/encoder42_eval/light-encoder-native) · [독립 반박 검수 2회](codex_light_encoder_최종_sol.txt) · [DIODE 공개 자료 설명](https://diode-dataset.org/) · [MoGe 평가용 배포 설명](https://github.com/microsoft/MoGe/blob/main/docs/eval.md)",""]
    Path(__file__).with_name("LIGHT_ENCODER_RESULT.md").write_text("\n".join(table),encoding="utf8")


if __name__ == "__main__":
    main()
