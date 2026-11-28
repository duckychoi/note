"""Generate the numerical tables directly from the preserved video and training ledgers."""
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
LABELS={"uniform":"전체 표본 균등","source":"원본 장면 균등","replay":"과거 사례 누적",
        "gain":"누적 + 산식 게인","learned":"누적 + 학습한 게인","visible":"누적 + 확인 여부",
        "shuffle":"누적 + 게인 위치 섞기"}
BASELINES=("uniform","source","replay")


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf8"))


def outcomes(lines, title, prefix):
    d=read(prefix+"summary.json")
    lines.extend([f"## {title}",""])
    for key,kname in (("whole","전체"),("thin","가는 경계"),("inner","경계 안쪽"),("outer","경계 바깥쪽")):
        m=d[key]
        lines.append(f"- {kname} 평가 영상 {len(m['scenes'])}개, 프레임 {sum(m['n_frames'])}개, "
                     f"유효 위치 {m['n_pixels']:,}개. 해당 영상 번호 {m['scenes']}.")
    lines.extend(["","단위는 mm이며 낮을수록 좋다. 프레임을 영상 안에서 평균하고 영상과 훈련 시드는 각각 같은 비중으로 집계했다.","",
                  "| 방법 | 실측 0 · 전체 | 실측 0 · 가는 경계 | 소량 · 전체 | 소량 · 가는 경계 |",
                  "|---|---:|---:|---:|---:|"])
    for arm,label in LABELS.items():
        cells=[d[key]["results"][budget]["means"][arm] for budget in ("zero","few") for key in ("whole","thin")]
        lines.append("| "+label+" | "+" | ".join(f"{v:.2f}" for v in cells)+" |")
    lines.extend(["","산식 게인 − 각 기준. 각 칸은 평균 차이와 괄호 안 단측 95% 상한이다. 음수는 산식 게인의 오차 감소를 뜻한다.","",
                  "| 비교 기준 | 실측 0 · 전체 | 실측 0 · 가는 경계 | 소량 · 전체 | 소량 · 가는 경계 |",
                  "|---|---:|---:|---:|---:|"])
    for arm in BASELINES+("visible","shuffle","learned"):
        cells=[d[k]["results"][b]["gain_minus"][arm] for b in ("zero","few") for k in ("whole","thin")]
        lines.append("| "+LABELS[arm]+" | "+" | ".join(f"{v['delta']:+.2f} ({v['upper95']:+.2f})" for v in cells)+" |")
    for b,name in (("zero","실측 0"),("few","소량 실측")):
        passed=sum(d[k]["results"][b]["gain_minus"][arm]["upper95"]<0 and
                   d[k]["results"][b]["gain_minus"][arm]["delta"]<0
                   for arm in BASELINES for k in ("whole","thin"))
        lines.extend(["",f"{name}에서 세 기준 × 두 지표 중 {passed}/6개가 통계 기준을 만족했다."])
    lines.extend(["","가는 경계의 안쪽·바깥쪽을 각각 점검했다. 괄호는 같은 누적 방법에 산식 게인을 붙이면서 바뀐 오차다.","",
                  "| 실측 | 물체 안쪽 | 물체 바깥쪽 |","|---|---:|---:|"])
    for b,name in (("zero","0개"),("few","소량")):
        cells=[d[k]["results"][b] for k in ("inner","outer")]
        lines.append("| "+name+" | "+" | ".join(f"{v['means']['gain']:.2f} ({v['gain_minus']['replay']['delta']:+.2f})" for v in cells)+" |")
    lines.append("")


def main():
    lines=["# #42 × #43 수치 장부",""]
    outcomes(lines,"기본 시험", "")
    outcomes(lines,"별도 촬영 영상의 보조 시험","extra_")
    ledgers=[json.loads(f.read_text()) for f in sorted(HERE.glob("ledger_*.json"))]
    assert len(ledgers)==5
    lines+=["## 고정한 결합식의 합성 점검","",
            "별도 원본 16쌍씩에서 같은 확인 픽셀로 역깊이 MSE(m⁻²)를 쟀다. 합성 원본과 초기값을 함께 바꾼 다섯 실행의 단순 평균이다.","",
            "| 단안 | 스테레오 | 고정 게인 | 산식 게인 | 학습 게인 |","|---:|---:|---:|---:|---:|",
            "| "+" | ".join(f"{mean(r['audit']['mse'][k] for r in ledgers):.5f}" for k in ("mono","stereo","constant","gain","learned"))+" |",""]
    g=read("geometry_check.json")
    for k,label in (("thin_boundary","가는 물체 근처"),("away_from_boundary","그 밖")):
        lines.append(f"- 합성 {label}에서 가시성·좌우 일치 검사를 통과한 비율은 "
                     f"{100*mean(r[k]['accepted'] for r in g.values()):.1f}%다. 그중 시차가 정답에서 10% 넘게 빗나간 비율은 "
                     f"{100*mean(r[k]['bad_disparity_among_accepted'] for r in g.values()):.1f}%다.")
    lines.extend(["","## 재현","",
      "학습·평가에 사용한 공개 BOP 아카이브는 ycbv_test_bop19.zip, SHA-256 c0c11251849877e7b2f373f6c7acf54739dc69d0fb2649c20050ca977bf5513d다. research-sources/gain42_video 아래 원본을 둔다. 보조 자료는 extra_data.py가 고정한 리비전에서 내려받아 CRC를 확인한 뒤 선택 파일 SHA-256을 기록한다. 이전 studio43_first_20261124의 학생 모델 정의와 입력 전처리를 사용한다.","",
      "CUDA PyTorch, OpenCV, NumPy, SciPy, requests 환경에서 CUBLAS_WORKSPACE_CONFIG를 :4096:8로 지정한다. experiment.py, geometry_check.py, extra_data.py, evaluate.py, evaluate.py --extra, report.py를 실행한다. frames.json과 extra_frames.json은 영상·프레임·모델별 오차를 담고, summary.json 및 extra_summary.json은 장면별 평균과 시드별 짝 차이를 담는다. 각 ledger 파일은 실제 배치 순서 해시와 개발 깊이 조회 위치를 보관한다. 70개 학생 및 5개 게인망 가중치는 같은 캐시의 checkpoints에 있다.",""])
    (HERE/"TABLES.md").write_text("\n".join(lines),encoding="utf8")


if __name__=="__main__":
    main()
