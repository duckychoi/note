"""Reconstruct quantitative tables and create a contact sheet from the actual RGB renders."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from material_light_assets import OUT,digest


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf8"))


def gain(b,x):
    return 100*(1-x/b)


def figure():
    # A furnished living room, selected to show the floor, upholstery and cast shadows.
    sid = "ai_046_005"
    folder = OUT/"renders"/sid
    w,h = 380,285
    fig = Image.new("RGB",(3*w+32,3*h+123),"#f7f6f3")
    draw = ImageDraw.Draw(fig)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",17)
    heading = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf",22)
    y0 = 70
    draw.text((8,6),"같은 방, 같은 카메라에서 재질과 광원만 바꾼 재렌더링",font=heading,fill="#16191c")
    for i in range(3):
        x = 8+(w+8)*i
        draw.text((x,42),f"광원 {i+1}",font=font,fill="#20252a")
        with np.load(folder/"passes.npz") as f:
            shadow,glint = f[f"shadow{i}"],f[f"glint{i}"]
        for j,mat in enumerate(("matte","coat","passes")):
            pic = Image.open(folder/f"{'coat' if j==2 else mat}-{i}.png").convert("RGB")
            if j==2:
                arr = np.asarray(pic).astype(float)
                arr *= .67
                for mask,color in ((shadow,(61,170,255)),(glint,(255,152,32))):
                    arr[mask] = .18*arr[mask]+.82*np.asarray(color)
                pic = Image.fromarray(arr.astype(np.uint8))
            pic = pic.resize((w,h),Image.Resampling.LANCZOS)
            y = y0+j*(h+5)
            fig.paste(pic,(x,y))
            label = ("무광","유광 코팅","광선 가림 · 반사 성분")[j]
            width = int(draw.textlength(label,font=font))
            draw.rectangle((x+5,y+5,x+16+width,y+31),fill="#182128")
            draw.text((x+10,y+6),label,font=font,fill="white")
    draw.text((8,3*h+85),"파랑 = 광원까지 가려진 표면     주황 = 직접 거울반사 성분이 큰 표면     Hypersim ai_046_005 · CC BY-SA 3.0",font=font,fill="#20252a")
    fig.save(OUT/"material-light-figure.jpg",quality=91)


def main():
    valid,real,lock,diag = [read(n) for n in ("validation.json","middlebury.json","gain-lock.json","mechanisms.json")]
    assert real["gain_lock"]==diag["lock"]==digest(OUT/"gain-lock.json")
    assert read("depth-run.json")["model_sha256"]==lock["model"]
    for results,scale in ((valid,diag["synthetic"]),(real,diag["real"])):
        for key in ("half","conditional"):
            assert abs(results["means"][key]-scale["original"][key])<1e-8
    items=read("scenes.json")
    tris=[read(f"renders/{r['scene']}/render.json")["triangles"] for r in items]
    text=["# #43 카메라·표면을 고정한 재질 × 광원 방향 검증","",
          "합성에서 정한 게인을 Middlebury의 같은 카메라·같은 셔터 두 조명에 적용했다. 두 장의 단안 예측을 결합하는 시험이다. 이전 단안 학생의 가중치는 갱신하지 않았다.","",
          "## 통제한 양과 재렌더링","",
          f"Hypersim 32개 적합 장면, 16개 검증 장면을 사용했다. 전체 48개 장면의 장면 ID 및 장면명 접두어가 서로 다르다. 공식 G-buffer의 위치(미터 광선거리와 카메라로 복원), 확산 반사율, 카메라 법선과 렌더 노드 ID를 사용했다. 각 카메라에서 불투명 무광과 같은 바탕색의 두께 없는 GGX 유광 코팅, 세 점광원을 완전 교차해 1024×768 RGB 288장을 만들었다. 위치와 정답 깊이 배열을 여섯 조건이 공유한다. 광원 위치는 카메라를 기준으로 같은 반지름·같은 광축 위치이며 같은 세기를 썼다. 장면당 인접 시점을 포함한 삼각형 {min(tris):,}~{max(tris):,}개의 교차로 그림자를 계산했다. 주변광, 노출과 톤 매핑을 고정했다. 복원에서 보이지 않는 가림면, 간접 반사와 실제 도막 내부 수송은 모델 범위 밖이다.",
          "","![실제 재렌더링 예](../../../research-sources/encoder42_eval/material-light-43/material-light-figure.jpg)","",
          "원본은 Mike Roberts 등, Apple Hypersim이다. 위 이미지는 공개 깊이·재질·법선을 사용한 수정 렌더이며 원자료와 같은 CC BY-SA 3.0으로 제공한다. [공식 데이터·라이선스](https://github.com/apple-aiml-research/ml-hypersim) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)","",
          "## 미터 깊이의 주 결과","",
          "AbsRel은 각 픽셀의 절대 오차를 정답 미터 깊이로 나누어 평균한 값이다. 두 독립 RGB 추론 모두 원해상도와 모델 추정 초점을 사용했다. 양수 개선은 오차 감소를 뜻한다. 95% 구간은 동일 계열을 짝지어 10,000회 재표집했다.","",
          "| 시험 | 반반 | RGB 후보 전신뢰¹ | 공통 강도 | 조건별 강도 | 조건별 상대 개선 [95%] | 나아진 계열 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for label,r in (("합성 16장면·96쌍",valid),("실사 14장면·11계열",real)):
        m=r["means"]
        text.append(f"| {label} | {m['half']:.6f} | {m['full_candidate']:.6f} | {m['common']:.6f} | {m['conditional']:.6f} | {r['gain']:+.3f}% [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['wins']}/{len(r['groups'])} |")
    text += ["","¹ RGB 후보가 없는 점은 모든 방법이 반반을 사용한다. 후보 전신뢰는 후보가 있는 점에서 해당 사진만 사용하는 고정 대조다. 공개 S에서 출발해 기존 연구에서 Hypersim으로 추가 학습한 Depth Pro 2-S D2를 사용했다. 실사는 앞 단계에서 이미 사용한 공개 자료다.","",
             "## 어느 사진을 얼마만큼 더 믿었는가","",
             "| 두 RGB에서 검출한 조건 | 후보 | 후보 사진에 준 최종 가중치 | 실사 면적 | 실사 후보의 끝점 승률² | 실사 반반 → 조건별 AbsRel |",
             "|---|---|---:|---:|---:|---:|"]
    for i,k,label,cand in ((1,"highlight","반사광 후보","덜 밝은 쪽"),(2,"shadow","그림자 후보","더 밝은 쪽"),(3,"both","두 조건 동시","더 밝은 쪽"),(0,"quiet","약한 변화","반반")):
        s=real["regions"][k]
        win=f"{100*s['candidate_wins']:.1f}%" if i else "해당 없음"
        text.append(f"| {label} | {cand} | {50*(1+lock['coefficients'][i]):.1f}% | {100*s['fraction']:.1f}% | {win} | {s['half']:.6f} → {s['conditional']:.6f} |")
    text += ["","² 두 예측 중 정답에 더 가까운 사진의 비율이며 동률은 반점이다. 절대 오차 크기도 성적에 반영되므로 승률만으로 반반 대비 개선을 결정할 수 없다. 실사의 검출 조건은 RGB 대용 지표다. 촬영한 물체의 BRDF 또는 조명별 그림자 라벨은 제공되지 않는다. [Middlebury 촬영 설명](https://vision.middlebury.edu/stereo/data/scenes2006/)","",
             "![Middlebury Books의 두 사진·RGB 선택·실제 깊이 우열](../../../research-sources/encoder42_eval/material-light-43/middlebury-books.jpg)","",             "## 렌더러가 확인한 그림자와 반사","",
             "한쪽 사진에서만 광원까지의 광선이 차단된 경우에는 차단 라벨이 없는 사진, 한쪽에서만 거울반사 라벨이 켜진 경우에는 그 라벨이 없는 사진을 후보로 택했다. 두 마스크가 동시에 바뀐 점은 제외했다. 다른 마스크의 상태가 양쪽에서 같을 수 있다. 거울반사 성분은 유광 RGB 선형 복사휘도 평균의 20%를 초과하며 절대 성분이 0.015보다 큰 점이다. 다음 표의 물리 강도는 합성 적합 32장면의 렌더 라벨로 추가 적합한 진단이다. 실사에서 이 라벨에 접근할 수 없다.","",
             "| 한쪽에서만 변한 물리 조건 | 유효점 비중 | 후보의 끝점 승률 | 반반 | 후보 전신뢰 | 후보 75% | 물리 적합 후보 비중 → 오차 | RGB 검출 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,tag,label in (("ray_shadow","shadow","광선 차단"),("specular","glint","거울반사")):
        s=valid["physical"][k]
        text.append(f"| {label} | {100*s['fraction']:.2f}% | {100*s['candidate_wins']:.2f}% | {s['half']:.6f} | {s['full_candidate']:.6f} | {s['half_step']:.6f} | {50*(1+diag['physics_lambda'][tag]):.1f}% → {diag['physics_scores'][tag]['fitted']:.6f} | {100*s['proxy_recall']:.2f}% |")
    text += ["","## 재질만 바꾼 대응 실험","",
             "각 광원 열에서는 카메라·표면·광원이 동일하고 재질만 바뀐다. 한 칸은 16개 보류 장면에서 사진 하나로 추정한 AbsRel의 평균이다.","",
             "| 재질 | 광원 1 | 광원 2 | 광원 3 |","|---|---:|---:|---:|"]
    for mat,label in (("matte","무광"),("coat","유광")):
        vals=[np.mean([r["single"][f"{mat}-{i}"] for r in valid["factorial"]]) for i in range(3)]
        text.append("| "+label+" | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    text+=["",f"동일 장면에서 재질 교체가 일으킨 AbsRel 차이는 광원에 따라 달랐다. 세 방향에서 재질 효과의 최대−최소 범위는 장면 평균 {np.mean([r['interaction_range'] for r in valid['factorial']]):.6f}이었다. 반반 대비 조건별 게인의 개선은 무광 {valid['by_material']['matte']['gain']:+.3f}%, 유광 {valid['by_material']['coat']['gain']:+.3f}%였다.","",
          "## 공통 배율과 인접 표면 형상","",
          "카메라는 실제로 고정했지만 신경망은 사진마다 다른 초점과 전체 깊이 배율을 낼 수 있다. 추가 진단에서 두 예측의 초점을 기하평균에 맞추거나 두 전체 영상의 평균 로그 깊이를 같게 맞췄다. 두 조정 모두 RGB 예측만 사용하고 같은 조건별 게인을 적용했다. 별도 카메라 보정값 진단은 Hypersim 내부행렬과 Middlebury 1/3 크기 영상의 3740/3 px를 두 사진에 공통 적용했다. [Middlebury 2005 광학 보정](https://vision.middlebury.edu/stereo/data/scenes2005/) · [2006 광학 보정](https://vision.middlebury.edu/stereo/data/scenes2006/)","",
          "| 자료 | 원래 미터 개선 | 두 추정 초점 통일 후 | 알려진 카메라 초점 사용 | 영상 전체 로그 배율 통일 후 | 이웃 로그 깊이 차 MSE, 반반 → 조건별 |",
          "|---|---:|---:|---:|---:|---:|"]
    for k,r,label in (("synthetic",valid,"합성"),("real",real,"실사")):
        d=diag[k]
        text.append(f"| {label} | {d['original']['gain']:+.3f}% | {d['common_focal']['gain']:+.3f}% | {d['calibrated_focal']['gain']:+.3f}% | {d['common_geomean']['gain']:+.3f}% | {r['shape']['half']:.7f} → {r['shape']['conditional']:.7f} |")
    text+=["","같은 실사 픽셀과 고정된 RGB 판정에서 초점만 바꾼 부분집합 진단이다. 각 칸은 해당 부분에서의 반반 → 조건별 AbsRel이다.","",
           "| 실사 RGB 조건 | 사진별 추정 초점 | 두 추정 초점의 기하평균 | 알려진 고정 카메라 초점 |",
           "|---|---:|---:|---:|"]
    for tag,label in (("shadow","그림자 후보"),("highlight","반사광 후보"),("both","겹침")):
        cells = [diag["real"][key]["regions"][tag] for key in ("original","common_focal","calibrated_focal")]
        assert abs(cells[0]["half"]-real["regions"][tag]["half"])<1e-8
        text.append("| "+label+" | "+" | ".join(f"{c['half']:.6f} → {c['conditional']:.6f}" for c in cells)+" |")
    text+=["","형상 진단은 16픽셀 떨어진 두 점을 4픽셀 격자에서 검사한다. 합성은 같은 렌더 노드, 실사는 참 깊이가 5% 안인 이웃이다. 어느 조건도 같은 평면의 정답 라벨은 아니다.","",
          "## 해석","",          "실사에서는 RGB 그림자 후보에서 밝은 사진으로 이동한 것이 주된 원천이었다. 반사광 후보에서는 덜 밝은 쪽에 준 가중치가 오차를 키웠다. 합성 보정의 검증 개선 구간에는 0이 포함되며, 실사에서 후보 전신뢰가 조건별 강도보다 더 낮은 오차를 냈다. 실사 추정 초점 또는 전체 로그 배율을 같게 만들면 원래의 이득이 사라졌고, 인접 형상 오차는 오히려 증가했다. 국소 표면 형상의 개선이나 신경망 학습으로의 전이는 입증되지 않았다. 알려진 카메라 초점을 고정해도 그림자 후보 부분은 소폭 개선됐다. 반사광 후보와 겹침 영역은 악화했다. 다음 확인 대상은 광원을 움직이면 이동하는 그림자 경계와 같은 위치에 남는 실제 표면 경계다.","",          "## 재현","",
          "공식 원본 취득, 재렌더링, 동일 모델 추론, 합성 게인 고정과 시험, 배율·물리 진단, 이 보고서 작성 순서는 각각 다음 파일에 담았다. 렌더는 realsense 환경의 Open3D 0.19와 실험 폴더의 h5py, 추론은 기존 torch 환경을 사용한다.",
          "", "[수식과 사전 규약](MATERIAL_LIGHT_PROTOCOL.md) · [원자료 취득](material_light_assets.py) · [광선 추적·GGX](material_light_render.py) · [단안 추론](material_light_depth.py) · [게인 적합·평가](material_light_test.py) · [추가 진단](material_light_diagnostics.py) · [독립 검수 3회](codex_material_light_최종_sol.txt) · [밝기·재질별 데이터와 288장](../../../research-sources/encoder42_eval/material-light-43)",""]
    Path(__file__).with_name("MATERIAL_LIGHT_RESULT.md").write_text("\n".join(text),encoding="utf8")
    figure()


if __name__=="__main__":
    main()
