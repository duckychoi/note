"""Build the result table directly from the locked optical and monocular experiments."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photometric_teacher import MATERIALS, OUT, SOURCE, STRIDE, rgb


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def teacher_summary(rows, split):
    reports = [read(OUT / "renders" / row["scene"] / "teacher.json") for row in rows if row["split"] == split]
    result = {}
    for material in MATERIALS:
        summed = {key: sum(row[material][key] for row in reports) for key in reports[0][material]}
        result[material] = dict(**summed,
            coverage=summed["accepted"] / summed["valid"],
            degrees=summed["angular_sum"] / summed["accepted"],
            ratio_rmse=(summed["log_ratio_squared"] / summed["covered_edges"])**.5,
            flat_rmse=(summed["flat_squared"] / summed["covered_edges"])**.5)
    return result


def illustration(rows):
    scene = max((r["scene"] for r in rows if r["split"] == "val"), key=lambda s:
                read(OUT / "renders" / s / "teacher.json")["matte"]["accepted"])
    root = OUT / "renders" / scene
    with np.load(root / "geometry.npz") as geo, np.load(root / "teacher.npz") as est:
        valid = geo["valid"][::STRIDE, ::STRIDE]
        normal = geo["normal"][::STRIDE, ::STRIDE].astype(np.float32)
        solved = est["matte_normal"].astype(np.float32)
        trust = est["matte_trust"] & valid

    def colors(value, mask):
        display = (127.5 * (1 + value * [1, -1, -1])).clip(0, 255).astype(np.uint8)
        display[~mask] = [24, 28, 35]
        return display

    imgs = [rgb(root / f"matte-{index}.png") for index in (0, 2, 4)]
    imgs += [colors(normal, valid), colors(solved, trust), colors(solved, trust & valid)]
    last = imgs[-1]
    last[:] = 24
    last[trust] = [219, 194, 91]
    width, height, gap, label = 288, 216, 9, 29
    canvas = Image.new("RGB", (3 * width + 4 * gap, 2 * (height + label) + 3 * gap), "#151921")
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    names = ["조명 1", "조명 3", "조명 5", "렌더 정답 법선", "밝기 8장으로 복원한 법선", "채택된 위치"]
    for i, (name, pixels) in enumerate(zip(names, imgs)):
        x = gap + i % 3 * (width + gap)
        y = gap + i // 3 * (height + label + gap)
        resampling = Image.Resampling.LANCZOS if i < 3 else Image.Resampling.NEAREST
        canvas.paste(Image.fromarray(pixels).resize((width, height), resampling), (x, y))
        draw.text((x, y + height + 3), name, font=font, fill="#F6F7FB")
    canvas.save(OUT / "photometric-teacher.jpg", quality=91)
    return scene


def main():
    rows = read(SOURCE / "scenes.json")
    teacher = teacher_summary(rows, "val")
    scene = illustration(rows)
    raw = read(OUT / "teacher-diligent.json")["objects"]
    arms = ("least_squares_96", "least_squares_96_same_pixels", "robust_8")
    avg = {arm: np.mean([result[arm]["mean"] for result in raw.values()]) for arm in arms}
    coverage = sum(result["robust_8"]["n"] for result in raw.values()) / sum(r["pixels"] for r in raw.values())
    results = [
        "# #43 광도 스테레오 관측과 #42 단안 전이 결과",
        "",
        "보정 광원 여러 장은 직접 관측한 표면의 기울기를 복원했다. 같은 RGB 한 장으로 깊이를 예측할 때에는 검사한 실사 세 자료 모두에서 광도 감독의 상대 AbsRel 개선이 3%에 미달했다. 재렌더링으로 적응한 세 조건 모두 실사에서 동결 시작점보다 AbsRel이 높았다.",
        "",
        "## 두 단계의 데이터와 코드",
        "",
        "48개 Hypersim 방의 카메라 고정 재렌더를 사용했다. 방마다 8개 평행광에서 두 재질을 렌더하고 주변광만 있는 영상을 하나씩 생성했다. 전체 768개 단일 광원 렌더와 48개 주변광 프레임을 생성했다. 학습 32가족, 합성 검증 16가족이다. 광도 교사는 참 깊이를 모른 채 RGB와 보정 광원으로 법선을 구했다. 참 렌더 ID와 참 깊이 연속성은 이후 보조 손실의 위치를 정하는 데만 사용했다.",
        "",
        "[실험 규약](PHOTOMETRIC_STEREO_PROTOCOL.md) · [렌더](photometric_render.py) · [복원](photometric_teacher.py) · [실사 법선](photometric_real.py) · [학습 및 평가](photometric_student.py) · [원자료](../../../research-sources/encoder42_eval/photometric-stereo-43)",
        "",
        "## 여러 장으로 직접 복원한 형상",
        "",
        "| 합성 검증 16장면 | 유효 점 중 법선 채택률 | 채택점 평균 각도 오차 | 정답 표면 쌍 채택률 | 같은 쌍 로그 깊이비 RMSE (제로 기울기 → 광도) |",
        "|---|---:|---:|---:|---:|",
    ]
    for material, values in teacher.items():
        results.append(f"| {material} | {values['coverage']:.2%} | {values['degrees']:.3f}° | "
                       f"{values['covered_edges']/values['surface_edges']:.2%} | "
                       f"{values['flat_rmse']:.7f} → {values['ratio_rmse']:.7f} |")
    results.extend([
        "",
        f"[검증 렌더 예](../../../research-sources/encoder42_eval/photometric-stereo-43/photometric-teacher.jpg)에서 합성 무광 신뢰점이 가장 많았던 {scene}을 예시로 사용했다. 결과 수치는 16장면 전체에서 계산했다. 재질별 채택 위치가 다르다.",
        "",
        f"공개 DiLiGenT 실사 10개 물체에서 보정 광원 방향만으로 8개를 골랐다. 4픽셀 간격에서 픽셀 합산 채택률은 {coverage:.2%}였다. 16비트 선형 PNG를 광원 세기로 정규화했다. 주변광 단독 촬영은 사용하지 않았다. 평균 각도 오차는 물체에 같은 비중을 주면 {avg['robust_8']:.3f}°다. 공식 자료 96장으로 같은 픽셀에서 직접 계산한 최소제곱 오차의 평균은 {avg['least_squares_96_same_pixels']:.3f}°다. 영상 수, 조명 배치, 손실 방식이 함께 달라지는 기준 참고값이다. 이 시험은 스캔 법선과 비교한 각도 시험이다.",
        "",
        "| 실사 물체 | 8장 채택률 | 8장 평균 각도 | 같은 점 96장 최소제곱 |",
        "|---|---:|---:|---:|",
    ])
    for obj, row in raw.items():
        sparse, ref = row["robust_8"], row["least_squares_96_same_pixels"]
        results.append(f"| {obj.removesuffix('PNG')} | {sparse['n']/row['pixels']:.2%} | "
                       f"{sparse['mean']:.2f}° | {ref['mean']:.2f}° |")
    results.extend([
        "",
        "## 한 장짜리 모델에 전달한 결과",
        "",
        "| 자료 | 영상 수 / 그룹 수 | 동결 시작점 | 조명 증강 | 광도 기울기 | 참 깊이비 | 광도 상대 개선 [95%] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    short = {"synthetic": "재렌더링 검증", "hammer": "HAMMER 실사", "diode": "DIODE 실내 실사", "ibims": "iBims-1 실사"}
    for dataset, label in short.items():
        data = read(OUT / f"scores-1024-{dataset}.json")
        means = data["means"]
        compare = data["comparison"]["photometric"]["absrel"]
        low, high = compare["interval"]
        results.append(f"| {label} | {data['n']} / {len(data['groups'])} | {means['frozen']:.6f} | "
                       f"{means['augmentation']:.6f} | {means['photometric']:.6f} | {means['ground_truth']:.6f} | "
                       f"{compare['relative']:+.3f}% [{low:+.3f}, {high:+.3f}] |")
    results.extend([
        "",
        "| 자료 | 로그 깊이 오차에서 배율을 뺀 분산, 증강 → 광도 | 광도 상대 개선 [95%] |",
        "|---|---:|---:|",
    ])
    for dataset, label in short.items():
        data = read(OUT / f"scores-1024-{dataset}.json")
        means = data["means"]
        compare = data["comparison"]["photometric"]["_shape"]
        low, high = compare["interval"]
        results.append(f"| {label} | {means['augmentation_shape']:.6f} → {means['photometric_shape']:.6f} | "
                       f"{compare['relative']:+.3f}% [{low:+.3f}, {high:+.3f}] |")
    results.extend(["", "| 합성 동일 광도 채택 쌍의 로그 깊이비 MSE | 조명 증강 | 광도 기울기 | 참 깊이비 |",
                    "|---|---:|---:|---:|"])
    for step in (512, 1024):
        means = read(OUT / f"scores-{step}-synthetic.json")["means"]
        values = [means[k + "_local_sum"] / means[k + "_local_n"]
                  for k in ("augmentation", "photometric", "ground_truth")]
        results.append(f"| {step}회 | {values[0]:.8f} | {values[1]:.8f} | {values[2]:.8f} |")
    train = read(OUT / "training.json")
    local_groups = list(read(OUT / "scores-1024-synthetic.json")["groups"].values())
    base = np.array([row["augmentation_local_sum"] for row in local_groups])
    alternative = np.array([row["photometric_local_sum"] for row in local_groups])
    picks = np.random.default_rng(261127).integers(len(base), size=(10000, len(base)))
    local_interval = np.quantile(100 * (1 - alternative[picks].sum(1) / base[picks].sum(1)), [.025, .975])
    results.extend([
        "",
        f"최종 합성 학습의 지원영역 내 국소 깊이비 MSE 상대 감소는 {100*(1-alternative.sum()/base.sum()):.3f}%이고, 방 계열 재표집 95% 구간은 [{local_interval[0]:+.3f}, {local_interval[1]:+.3f}]%다. 이 사후 구간은 8픽셀 깊이비에만 적용된다.",
        "",
        f"모델은 전체 Hypersim에서 훈련한 기존 D2 체크포인트를 사용했다. 세 조건은 원해상도 RGB {train['rgb']}장으로 마지막 4블록을 같은 순서로 {train['steps']:,}회 갱신했다. 시드는 소스에 고정했다. 학습 JSON에는 사진, 손실, 갱신 전 그래디언트 크기, 코드·자료 해시를 보관한다. 보조 쌍의 MSE는 전체 방의 면적·배율 오차와 다른 지표다.",
        "",
        "## 해석 범위와 독립 검수",
        "",
        "- 독립 검수 1차에서 최소제곱과 핀홀 깊이비 증명을 검산했다. 2차에서 렌더와 학생 코드를 대조했다. 사전 규약의 ‘양 끝 법선을 평균’ 문구를 정정한다. 실제 코드는 양 끝 법선 각각의 로그 깊이비를 계산하고 그 값을 평균한다. 리지를 사용하지 않는 정확 해를 정리로 다루며 구현은 수치적으로 10⁻⁶ 리지를 사용한다. 3차에서 장면별 원자료로 표와 재표집 구간을 독립 재집계했다. [검수 이력](codex_photometric_stereo_최종_sol.txt)",
        "- 실사는 이번 초매개변수 선택이나 학습에 쓰지 않았고, 과거 별도 후보에서 이미 평가한 자료다. 한 시드와 32개 장면으로 한정된 적응 시험이다. 부트스트랩 구간의 변동은 장면 간 차이만 반영하며 학습을 다시 시작할 때의 변동은 포함하지 않는다.",
        "- 합성 무광 수치는 노이즈가 적고 실제 법선과 같은 반사 모델을 써서 유리한 검산이다. 코팅·실사에서 그림자·반사 영향이 남고, 채택점으로 한정된 오차는 전체 표면 오차를 대표하지 않는다.",
        "- 본 설정의 비교는 강한 기존 Hypersim D2 시작점 위의 조명 증강 대조다. 모든 공개 SOTA 학습법, 실사 지도학습과의 본시험까지 포함하는 비교는 별도다.",
        "",
    ])
    Path(__file__).with_name("PHOTOMETRIC_STEREO_RESULT.md").write_text("\n".join(results), encoding="utf8")


if __name__ == "__main__":
    main()
