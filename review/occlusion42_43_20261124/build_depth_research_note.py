"""Build the portable Korean research notebook and the linked experimental record."""
import base64
from io import BytesIO
import json
from pathlib import Path
import re
from PIL import Image


HERE=Path(__file__).resolve().parent
NOTE=HERE.parents[1]
ROOT=NOTE.parent
DATA=ROOT/"research-sources/encoder42_eval"
E=DATA/"edge-motion-43"
R=DATA/"material-light-43"
REFERENCE=ROOT.parent/"바이브꿀팁/FIR_LoRA/core/docs/FIR_LoRA_research_brief.html"
DOC=NOTE/"docs/v2/42_43_depth_research_note.html"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def photo(path,width=1000):
    im=Image.open(path).convert("RGB")
    im.thumbnail((width,width*2),Image.Resampling.LANCZOS)
    data=BytesIO()
    im.save(data,format="JPEG",quality=87,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(data.getvalue()).decode()


def gain(m,k="moving"):
    return 100*(1-m[k]/m["half"])


def build(update_edge_report=True):
    result=read(E/"result.json")["sets"]
    lock=read(E/"lock.json")["strength"]
    diag=read(E/"label-diagnostic.json")
    q=result["val"]["quality"]
    realq=result["real"]["quality"]
    # The original's layout, type scale, color system, source badges and print palette.
    css=re.search(r"<style>(.*?)</style>",REFERENCE.read_text(encoding="utf8"),re.S)[1]
    html=TEMPLATE.replace("{{CSS}}",css)
    for name,path in {"RENDER":R/"material-light-figure.jpg","BOOKS":E/"Books.jpg",
                      "EDGE":E/"ai_046_005.jpg","OVER":E/"overlap.jpg",
                      "A":R/"renders/ai_046_005/coat-0.png","B":R/"renders/ai_046_005/coat-2.png"}.items():
        html=html.replace(f"{{{{{name}}}}}",photo(path,720 if name in ("A","B") else 1080))
    rows=[]
    for k,name in (("val","Hypersim · 16장면"),("real","Middlebury · 11계열")):
        s=result[k]["surface"]
        m=s["mean"]
        rows.append(f"<tr><th>{name}</th><td>{m['half']:.8f}</td><td>{m['moving']:.8f}</td><td>{gain(m):+.4f}%</td>"
                    f"<td>{s['interval'][0]:+.4f} ~ {s['interval'][1]:+.4f}%</td><td>{s['wins']}/{len(s['grouped'])}</td></tr>")
    html=html.replace("{{ROWS}}","".join(rows))
    table=[]
    for key,title in (("surface","표면 안쪽"),("all","모든 유효 쌍"),("jump","5% 이상 깊이차")):
        a,b=result["val"][key]["mean"],result["real"][key]["mean"]
        table.append(f"<tr><th>{title}</th><td>{gain(a):+.4f}%</td><td>{gain(b):+.4f}%</td><td>{gain(a,'unique'):+.4f}%</td><td>{gain(b,'unique'):+.4f}%</td></tr>")
    html=html.replace("{{AUX}}","".join(table))
    headroom=f"{diag['val_gain']:+.4f}%"
    html=html.replace("{{ORACLE}}",headroom).replace("{{LAMBDA}}",f"{lock['moving']:.4f}")
    html=html.replace("{{MOTIONP}}",f"{100*q['matched_shadow']/q['matched']:.2f}%")
    html=html.replace("{{EDGERECALL}}",f"{100*realq['depth_near_fixed']/realq['gt_depth']:.2f}%")
    same_strength={}
    for split,filename in (("val","val-rows.json"),("real","real-rows.json")):
        by_family={}
        for row in read(E/filename):
            v=row["moments"]["surface"]
            if v["n"]:
                a,c=v["shift"]
                harm=-2*lock["moving"]*a+lock["moving"]**2*c
                by_family.setdefault(row["group"],[]).append(harm)
        delta=sum(sum(v)/len(v) for v in by_family.values())/len(by_family)
        same_strength[split]=-100*delta/result[split]["surface"]["mean"]["half"]
    placebo=f"이동 방법과 동일한 강도 {lock['moving']:.4f}를 공간 이동 대조에 강제한 사후 검산은 합성 {same_strength['val']:+.4f}%, 실사 {same_strength['real']:+.4f}%였다."
    html=html.replace("{{PLACEBO}}",placebo)
    model_rows=[]
    for dataset,title in (("hammer","HAMMER"),("diode","DIODE 실내"),("ibims","iBims-1"),("diode_outdoor","DIODE 실외")):
        scores=read(DATA/"lightview42"/f"{dataset}.json")["means"]
        values=[scores.get(name) for name in ("apple_s","hypersim","fixed_0819","previous_lightview","lightview")]
        cells="".join(f"<td>{v:.4f}</td>" if v is not None else "<td>미측정</td>" for v in values)
        model_rows.append(f"<tr><th scope='row'>{title}</th>{cells}</tr>")
    html=html.replace("{{LIGHTVIEW_ROWS}}","".join(model_rows))
    if "{{" in html:
        raise ValueError("Notebook still contains a template marker")
    DOC.write_text(html,encoding="utf8")
    if not update_edge_report:
        return
    report=["# #43 움직이는 그림자와 고정 깊이 경계","",
            "고정된 카메라 보정 초점에서 반반 평균의 로그 깊이 지도를 기준으로 삼았다. 각 RGB는 동일한 Hypersim 대조 모델에서 따로 추론했다. 모델 및 게인 학습·검증 분할은 사전 규약에 적었다.","",
            "| 시험 | 반반 표면 MSE | 이동 방법 | 상대 개선 | 짝 재표집 95% 구간 | 개선 계열 | 한쪽만 보이는 경계 대조 | 강도 1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,name in (("val","합성 96쌍·16장면"),("real","실사 14쌍·11계열")):
        s=result[k]["surface"]; m=s["mean"]
        report.append(f"| {name} | {m['half']:.9f} | {m['moving']:.9f} | {gain(m):+.5f}% | [{s['interval'][0]:+.5f}, {s['interval'][1]:+.5f}]% | {s['wins']}/{len(s['grouped'])} | {m['unique']:.9f} | {m['unit']:.9f} |")
    report+=["",f"합성 32장면에서 적합한 강도는 이동 {lock['moving']:.6f}, 한쪽 경계 {lock['unique']:.6f}, 공간을 128픽셀 순환 이동한 대조 {lock['shift']:.6f}였다. 마지막 대조는 강도를 0으로 선택했다. 합성의 같은 렌더 노드 쌍과 실사의 참 깊이비 1.05 미만 쌍은 서로 다른 표면 근사다. 부트스트랩은 장면 계열을 짝지어 10,000회 수행했다.","",
            f"합성에서 이동 후보의 양 끝과 두 축을 센 {q['matched']:,}건 중 {q['matched_shadow']:,}건({100*q['matched_shadow']/q['matched']:.2f}%)이 양 끝의 차단 경계 및 원위치 소실 조건을 충족했다. 두 위치에 물리 경계가 있는지 확인하는 지표이며 같은 3D 차폐물의 신원을 추적한 정답은 아니다. 광원별 깊이 단차의 2픽셀 이내에 차단 경계가 함께 있는 깊이 경계 픽셀은 양쪽 조명·쌍 합계 {q['overlap']:,}/{2*q['gt_depth']:,}였다. 중복 집계 단위는 개별 물체가 아니다.",
            "",f"실사에는 그림자 GT가 없다. 이동 후보 위치 {realq['moving']:,}개 중 {realq['moving_near_depth']:,}개가 참 깊이 단차에 가까웠다. 고정 RGB 후보 위치 {realq['fixed']:,}개 중 {realq['fixed_near_depth']:,}개가 깊이 단차에 가까웠고 참 깊이 단차 {realq['gt_depth']:,}개 중 {realq['depth_near_fixed']:,}개 근처에서 고정 RGB 후보를 찾았다. 같은 후보가 여러 조명 쌍에서 다시 집계될 수 있다.","",
            "## 표면과 깊이 단차의 차이","",
            "| 점쌍 | 합성 이동 개선 | 실사 이동 개선 | 합성 한쪽 경계 개선 | 실사 한쪽 경계 개선 |","|---|---:|---:|---:|---:|"]
    for key,title in (("surface","표면 안"),("all","전체"),("jump","GT 5% 이상 단차")):
        a,b=result["val"][key]["mean"],result["real"][key]["mean"]
        report.append(f"| {title} | {gain(a):+.5f}% | {gain(b):+.5f}% | {gain(a,'unique'):+.5f}% | {gain(b,'unique'):+.5f}% |")
    report+=["",placebo,"","## 렌더 라벨을 사용한 추가 진단","",
            f"실사 주 시험 뒤 합성에서만 광선 차단의 실제 경계가 한쪽 영상에만 있는 띠를 선택하고 참 깊이 경계 주변을 제외했다. RGB 경계의 실제 이동거리를 요구하지 않는 특권 정보 진단이다. 합성 학습 장면에서 적합한 강도 {diag['strength']:.6f}, 별도 합성 검증 표면 MSE {diag['val_half']:.9f} → {diag['val_corrected']:.9f}, 개선 {headroom}. 사용 띠가 표준 격자에서 차지한 비율은 {100*diag['val_fraction']:.2f}%였다. 접근 가능한 모든 방법의 이론적 상한은 아니다.","",
            "현재 두 데이터 모두 주 형상 지표의 재표집 구간이 0을 포함한다. 후속 단안 학생에게 효과가 전달됐다고 판단할 근거는 없다.","",
            "[위치 조건·반례·형상 증명](EDGE_MOTION_PROTOCOL.md) · [구현](edge_motion43.py) · [렌더 라벨 진단](edge_motion_diagnostic.py) · [개별 관측·게인·해시](../../../research-sources/encoder42_eval/edge-motion-43) · [독립 검수 2회](codex_edge_motion_최종_sol.txt) · [HTML 연구노트](../../docs/v2/42_43_depth_research_note.html)",""]
    (HERE/"EDGE_MOTION_RESULT.md").write_text("\n".join(report),encoding="utf8")


TEMPLATE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>빛이 바뀌어도 깊이는 같다 · #42·43 연구노트</title>
<style>{{CSS}}
body{margin:0}.top-inner{padding-top:36px}.wrap{padding-bottom:52px}.studynav .wrap{padding-bottom:0}.studybar{align-items:center}
.studybtn[aria-selected="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}.studybtn{letter-spacing:0}.themer{margin-left:auto}
.sheet{padding-top:4px}.sheet[hidden]{display:none}.sheet section{margin-top:34px}.lead{font-size:17px;color:var(--ink-2);max-width:76ch}.lead strong{color:var(--ink)}
.mat{font:20px/1.65 Cambria,"Cambria Math",Georgia,serif;background:var(--surface-2);border-left:3px solid var(--accent);padding:12px 16px;overflow-x:auto;color:var(--ink);border-radius:0 4px 4px 0}
.mat math{font-size:1.08em}.wrap figure{margin:0}.wrap figure img{width:100%;height:auto;display:block;border-radius:5px;border:1px solid var(--rule)}figcaption{font-size:13px;color:var(--ink-2);margin:7px 0 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair figure{min-width:0}.diagram{overflow:auto;padding:16px;background:var(--surface);border:1px solid var(--rule);border-radius:6px}
.diagram svg{width:100%;height:auto;display:block;min-width:610px}.diagram text{font-family:"Malgun Gothic",system-ui,sans-serif;fill:var(--ink)}.tone{color:var(--accent)}
.flowbox{fill:var(--surface-2);stroke:var(--rule-strong)}.flowline{stroke:var(--accent);fill:none;stroke-width:2}.diagram .sub{fill:var(--ink-2);font-size:12px}
.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.controls label{font-size:14px;font-weight:600}.controls select,.controls button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--rule-strong);padding:5px;border-radius:4px}.controls input{vertical-align:middle;accent-color:var(--accent)}
.sources{font-size:13px;color:var(--ink-2)}.sources a{margin-right:10px}.studybar button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.skip{position:absolute;left:-200vw}.skip:focus{left:1rem;top:.5rem;background:var(--surface);padding:.4rem;z-index:10}.toclocal{display:flex;flex-wrap:wrap;gap:6px 16px;padding:8px 0 2px;font-size:13px}
footer{border-top:1px solid var(--rule);color:var(--ink-2);padding:20px 0;font-size:12px}.sheet :target{scroll-margin-top:100px}.compact{font-size:13px;color:var(--ink-2)}
@media(max-width:600px){.studynav{position:static}.studybar{gap:4px}.studybtn{padding:7px 8px;font-size:12px}.pair{gap:6px}.top-inner{padding-top:24px}.kv{grid-template-columns:repeat(2,1fr)}.diagram{padding:9px}.mat{font-size:17px}.sheet section{margin-top:25px}}
@media print{.sheet[hidden]{display:block!important}.studynav,.controls,.skip{display:none!important}.sheet{break-before:page}.sheet:first-of-type{break-before:auto}.spec,figure,tr{break-inside:avoid}.top-inner,.wrap{max-width:none}.wrap{padding-bottom:0}details>.fold-body{display:block}}
</style></head><body><a href="#overview" class="skip">본문으로</a>
<header class="top"><div class="top-inner"><div class="eyebrow">바이브연구 · SYNTHETIC DEPTH · RESEARCH NOTE</div>
<h1>빛이 바뀌어도 깊이는 같다</h1>
<p class="standfirst">같은 소파를 왼쪽과 오른쪽에서 비추면 그림자가 옮겨 간다. 소파의 모서리와 카메라까지의 거리는 그대로다. 이 차이를 이용해 <strong>어느 깊이 예측을 믿을지</strong> 판단하고, 궁극적으로는 사진 한 장만 보는 모델에 그 판단을 가르치는 연구다.</p>
<div class="meta-row"><span>#42 LightView-Depth</span><span>#43 조명 통제 스튜디오</span><span>2026.11.26 · 단안 배율 검증</span></div>
<div class="legend"><div><span class="badge b-d">유도</span>가정 아래 증명한 식</div><div><span class="badge b-c">실측</span>모델과 영상으로 잰 값</div><div><span class="badge b-u">미확인</span>남은 연구 질문</div></div></div></header>
<div class="studynav"><div class="wrap"><div class="studybar" role="tablist" aria-label="연구노트 장">
<button class="studybtn" id="t-overview" role="tab" aria-controls="overview">철학·파이프라인</button><button class="studybtn" id="t-model" role="tab" aria-controls="model">모델·학습</button><button class="studybtn" id="t-theory" role="tab" aria-controls="theory">경계 이론·반례</button><button class="studybtn" id="t-results" role="tab" aria-controls="results">실험·판정</button></div></div></div>
<main class="wrap">
<article class="sheet" id="overview" role="tabpanel" aria-labelledby="t-overview"><section><div class="sec-head"><span class="sec-num">01</span><h2>움직인 것은 빛인가, 물체인가</h2></div>
<div class="pair"><figure><img src="{{A}}" alt="첫 광원으로 비춘 동일한 소파와 바닥"><figcaption>같은 카메라와 유광 소파 · 광원 A</figcaption></figure><figure><img src="{{B}}" alt="광원만 옮겨 소파 쿠션과 벽에 그림자가 생긴 모습"><figcaption>광원 B · 쿠션 위의 어두운 경계가 이동했다.</figcaption></figure></div>
<p class="lead">사진의 밝은 선은 <strong>물체가 끝나는 곳</strong>일 수도, <strong>빛이 가려지는 곳</strong>일 수도 있다. #43에서 정답을 아는 작은 세계를 만들어 원인을 따로 바꾼다. #42에서는 그렇게 얻은 단서가 실사에서도 통하는 학습 신호인지 묻는다.</p>
<div class="split"><div class="pane good"><h4>#43 · 관측을 고르는 교사</h4><p>같은 장면의 여러 사진을 비교할 수 있다. 한 사진에 생긴 그림자를 벽의 단차로 오인했다면 다른 조명의 깊이 기울기가 도움을 줄 수 있다.</p></div>
<div class="pane"><h4>#42 · 한 장만 보는 LightView-Depth</h4><p>합성 장면의 조명 변화와 실사 시차는 학습 때 쓴다. 새 실사 사진 한 장에서 고정 배율을 이기고, Hypersim 대조의 깊이 오차를 상대적으로 3% 줄이는지 확인한다.</p></div></div>
</section><section><div class="sec-head"><span class="sec-num">02</span><h2>전체 파이프라인</h2></div>
<div class="diagram"><svg viewBox="0 0 920 232" role="img" aria-label="렌더러가 동일한 표면의 두 RGB를 만들고, 같은 깊이 모델의 따로 얻은 예측을 경계 교사가 비교한다. 별도 학생은 학습 후 한 장만 쓴다.">
<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="var(--accent)"/></marker></defs>
<g text-anchor="middle" font-size="17"><rect class="flowbox" x="8" y="30" width="166" height="76" rx="6"/><text x="91" y="60">렌더러</text><text class="sub" x="91" y="83">카메라·표면 고정</text>
<rect class="flowbox" x="215" y="13" width="160" height="48" rx="5"/><text x="295" y="43">조명 A 사진</text><rect class="flowbox" x="215" y="79" width="160" height="48" rx="5"/><text x="295" y="109">조명 B 사진</text>
<rect class="flowbox" x="418" y="30" width="185" height="76" rx="6"/><text x="510" y="60">Hypersim 대조</text><text class="sub" x="510" y="83">같은 가중치 · 각각 추론</text>
<rect class="flowbox" x="658" y="30" width="241" height="76" rx="6"/><text x="778" y="60">경계 교사 #43</text><text class="sub" x="778" y="83">위치 비교 → 형상 보정</text>
<rect class="flowbox" x="409" y="157" width="202" height="60" rx="6"/><text x="510" y="183">LightView-Depth #42</text><text class="sub" x="510" y="202">이번 경계 신호 전달은 미검증</text>
<text class="sub" x="182" y="190">렌더 정답 + 검증된 보조 목표</text><text class="sub" x="769" y="190">운용 목표 · RGB 한 장 → 깊이</text></g>
<g class="flowline" marker-end="url(#arr)"><path d="M174 65L208 38"/><path d="M174 72L208 102"/><path d="M375 38L412 61"/><path d="M375 102L412 78"/><path d="M603 68H650"/><path d="M778 107V135H510V151" stroke-dasharray="4 4"/><path d="M299 186H402"/><path d="M612 186H657"/></g></svg></div>
<p class="compact">실선은 데이터 흐름, 점선은 이번 경계 관측을 학생에게 전달하는 후속 실험이다. 서로 다른 조명의 두 RGB를 모델에 동영상처럼 한꺼번에 넣지 않는다.</p>
<div class="kv"><div><b>48</b><span>카메라 구도가 다른 합성 장면</span></div><div><b>2 × 3</b><span>무광·유광 × 광원 세 방향</span></div><div><b>288</b><span>1024 × 768 실제 재렌더 RGB</span></div><div><b>14</b><span>같은 셔터로 찍은 실사 장면</span></div></div>
<p class="sources"><a href="#model">모델과 학습으로 →</a><a href="#results">실험 결과로 →</a></p></section></article>

<article class="sheet" id="model" role="tabpanel" aria-labelledby="t-model"><section><div class="sec-head"><span class="sec-num">03</span><h2>모델은 사진에서 무엇을 내는가</h2></div>
<p><strong>Depth Pro 2-S (baseline)</strong>은 Apple 공개 가중치다. <strong>Hypersim 대조</strong>는 같은 모델에 합성 깊이만 추가로 가르친 가중치다. <strong>LightView-Depth</strong>는 여기에서 출발해 합성 조명 변화와 실사 시차에서 단안 배율을 배우는 우리 모델이다.</p>
<div class="spec"><div class="spec-head"><h3>실사 단안 깊이 성능</h3></div><div class="spec-body">
<p>AbsRel은 실제 거리로 나눈 절대 오차다. 작을수록 좋다.</p>
<div style="overflow-x:auto"><table><thead><tr><th scope="col">실사 자료</th><th scope="col">Depth Pro 2-S (baseline)</th><th scope="col">Hypersim 대조</th><th scope="col">고정 0.819배</th><th scope="col">LightView-Depth (이전 시차)</th><th scope="col">LightView-Depth (이번 학습)</th></tr></thead><tbody>{{LIGHTVIEW_ROWS}}</tbody></table></div>
<p class="sources"><a href="../../review/occlusion42_43_20261124/LIGHTVIEW42_PROTOCOL.md">배율 조건과 PF 수식</a><a href="../../review/occlusion42_43_20261124/LIGHTVIEW42_RESULT.md">학습과 실사 평가</a></p></div></div>
<div class="spec"><div class="spec-head"><span class="badge b-c">코드</span><h3>Depth Pro 2-S · 단안 경로</h3></div><div class="spec-body">
<div class="diagram"><svg viewBox="0 0 820 100" role="img" aria-label="RGB에서 16픽셀 패치, 12개 인코더 블록, 4개 특징 지도, 디코더와 역깊이로 진행"><g text-anchor="middle" font-size="15">
<rect class="flowbox" x="2" y="18" width="135" height="64" rx="5"/><text x="69" y="45">RGB → 패치</text><text class="sub" x="69" y="66">16 × 16 픽셀</text>
<rect class="flowbox" x="167" y="18" width="168" height="64" rx="5"/><text x="251" y="45">ViT-S · DINOv3</text><text class="sub" x="251" y="66">인코더 12블록</text>
<rect class="flowbox" x="367" y="18" width="150" height="64" rx="5"/><text x="442" y="45">4단계 특징</text><text class="sub" x="442" y="66">3·6·9·12번째 출력</text>
<rect class="flowbox" x="550" y="18" width="133" height="64" rx="5"/><text x="616" y="45">디코더</text><text class="sub" x="616" y="66">특징을 격자로 복원</text><text x="761" y="54">역깊이 q</text></g><g class="flowline"><path d="M138 50h24m174 0h25m157 0h26m141 0h33"/></g></svg></div>
<p>인코더는 사진을 작은 조각으로 나누고, 한 조각이 벽·의자·그림자 중 어디에 속하는지 주변을 보며 특징을 만든다. 디코더는 크기가 다른 특징을 합쳐 픽셀마다 정규화 역깊이 <i>q</i>를 낸다. 인코더의 RoPE는 패치 위치를 표현한다. 이번 그림자 움직임 판정은 두 영상의 경계 위치를 직접 비교한다.</p>
<div class="mat"><math aria-label="광축 깊이 Z는 초점 f를 영상 너비 W와 역깊이 q의 곱으로 나눈 값"><mi>Z</mi><mo>=</mo><mfrac><msub><mi>f</mi><mi>px</mi></msub><mrow><mi>W</mi><mi>q</mi></mrow></mfrac></math></div>
<p><i>W</i>는 영상 너비, <i>f</i>는 픽셀 단위 초점이다. 출력 <i>Z</i>는 카메라 광축을 따라 잰 미터 깊이다. 이번 #43 비교는 렌더러의 실제 초점과 Middlebury의 보정 초점 3740/3픽셀을 쓴다. 기존 #42 단안 운용 시험에서는 RGB에서 초점도 예측했다.</p></div></div></section>
<section><div class="sec-head"><span class="sec-num">04</span><h2>무엇을 실제로 학습했는가</h2></div>
<div class="tscroll"><table><thead><tr><th>단계</th><th>학습 신호</th><th>변한 것</th></tr></thead><tbody>
<tr><th>Hypersim 대조</th><td>Hypersim 공식 학습 앵커 59,543개. 미터 역깊이, 정규화 형상, 여러 해상도의 경계, 시점 간 일관성</td><td>Apple 공개 S에서 출발한다. 두 패스 중 합성 검증으로 92,000회 가중치를 선택했다.</td></tr>
<tr><th>#42 인코더 비교</th><td>같은 합성 정답으로 지도 계속학습, 고정 게인, 조건별 게인 비교</td><td>마지막 4블록 7,096,320개 파라미터를 각각 1,160회 AdamW로 갱신. 디코더·초점 추정기는 동결</td></tr>
<tr><th>#43 이번 시험</th><td>다른 조명에서 얻은 깊이 기울기를 어느 곳에 얼마나 쓸지 합성 32장면에서 추정</td><td>깊이 모델은 동결. 보정 강도 λ 하나를 방법별로 적합. 실사 정답은 마지막 채점용</td></tr></tbody></table></div>
<div class="split"><div class="pane"><h4>칼만 개념은 어디에 들어가나</h4><p>관측값에 신뢰도를 곱해 예측을 수정한다. 고전적 스칼라 KF에서 독립·무편향 오차 분산이 P, R이면 K=P/(P+R)이다. 이번 형상 보정은 예상되는 오차 방향과 실제 보정 방향의 내적으로 λ를 유도한다.</p></div>
<div class="pane"><h4>합성에서 실사로 넘어가는 시험</h4><p>합성의 재질·광원·정답으로 규칙을 정한 뒤 고정한다. 실사의 정답으로 배율이나 강도를 다시 맞추면 새로운 사진 한 장의 성능을 재는 문제가 바뀐다.</p></div></div>
<details class="fold"><summary>깊이 보정 · 파라미터 갱신 · 결합은 어떻게 다른가</summary><div class="fold-body"><p>깊이 보정은 출력 Z를 바꾼다. 파라미터 갱신은 인코더 가중치 θ 자체를 바꾼다. 결합은 두 효과를 같은 초기값과 데이터에서 대조한다. 과거의 고정 교사 목표에 게인을 곱해 역전파하는 구성은 가중 지도학습으로도 쓸 수 있다. 신경망 전체에 대한 리카티 최적성은 미증명이다. 환경 스타일·반사 다양화와 깊이 경계 보존도 따로 검증할 요소다.</p></div></details>
<p class="sources"><a href="../../review/occlusion42_43_20261124/LIGHT_ENCODER_RESULT.md">인코더 학습 기록</a><a href="../../review/occlusion42_43_20261124/PARAMETER_GAIN_THEORY.md">파라미터 게인 유도</a><a href="https://apple-aiml-research.github.io/ml-depth-pro-2/">Depth Pro 2 공식 자료</a></p></section></article>
'''

TEMPLATE += '''<article class="sheet" id="theory" role="tabpanel" aria-labelledby="t-theory"><section><div class="sec-head"><span class="sec-num">05</span><h2>그림자를 얼마나 움직여야 구별할 수 있을까</h2></div>
<p class="lead">사진 두 장에서 같은 선을 찾고, 선을 가로지르는 방향의 위치를 비교한다. 카메라가 고정돼도 선을 읽는 위치에는 오차가 있다.</p>
<div class="spec"><div class="spec-head"><span class="badge b-d">유도</span><h3>경계 위치의 분리 조건</h3></div><div class="spec-body">
<p>한 사진에서 위치를 읽는 오차의 절댓값을 ε 이하라 하자. 서로 다른 선으로 잘못 갈아타지 않고, 두 조명에서 같은 경계를 볼 수 있다고 가정한다. 고정된 선은 처음과 나중의 위치 오차가 반대로 쌓여도 2ε까지만 움직여 보인다. 실제로 δ 이상 옮겨 간 선은 두 오차가 이동을 감추더라도 δ−2ε 이상 움직여 보인다.</p>
<div class="mat">고정 경계　|x₂ − x₁| ≤ 2ε<br>이동 그림자　|x₂ − x₁| ≥ δ − 2ε<br>두 범위가 분리될 조건　<strong>δ &gt; 4ε</strong></div>
<p>ε=1픽셀, δ=6픽셀이면 고정 경계의 관측 이동은 최대 2픽셀이고 그림자는 최소 4픽셀이다. 그 사이의 3픽셀을 문턱으로 쓸 수 있다. δ=4픽셀에서는 두 범위가 2픽셀에서 만난다. 실제 영상의 위치 오차가 항상 1픽셀 이하라고 측정한 것은 아니다.</p></div></div>
<div class="spec"><div class="spec-head"><h3>광원을 옮겨 반례를 직접 보기</h3><span class="badge b-d">평면 단면</span></div><div class="spec-body">
<div class="controls"><label>단면 조건 <select id="case"><option value="free">떨어진 그림자</option><option value="touch">접촉 그림자</option><option value="overlap">첫 조명에서 단차와 겹침</option></select></label>
<label>광원 이동 <input id="lamp" type="range" min="-24" max="24" value="12" step="1" aria-label="광원의 가로 이동량"><output id="lampval">12</output></label></div>
<div class="diagram"><svg id="ray" viewBox="0 0 760 174" role="img" aria-labelledby="raytitle">
<title id="raytitle">높이가 다른 차폐물과 광원이 만드는 그림자의 이동</title>
<path d="M35 140H725" stroke="var(--rule-strong)" stroke-width="2"/><path id="line1" stroke="var(--ink-3)" stroke-width="2" stroke-dasharray="5 3" fill="none"/><path id="line2" stroke="var(--warm)" stroke-width="3" fill="none"/>
<circle id="lamp1" cy="24" r="5" fill="var(--ink-3)"/><circle id="lamp2" cy="24" r="6" fill="var(--warm)"/>
<path id="object" stroke="var(--ink)" stroke-width="5"/><path id="ledge" stroke="var(--accent)" stroke-width="4" d="M460 126V149"/>
<circle id="s1" cy="140" r="6" fill="var(--ink-3)"/><circle id="s2" cy="140" r="5" fill="var(--warm)"/>
<text x="40" y="27" font-size="13">광원 높이 H = 60</text><text x="40" y="164" font-size="13">바닥</text><text x="470" y="158" font-size="13">고정 깊이 경계</text></svg></div>
<p id="readout" class="note" aria-live="polite">광원을 옮기면 그림자는 반대 방향으로 움직인다.</p>
<p class="compact">수평 바닥과 점광원 모형이다. 좌표는 같은 길이 단위이며 화면은 가로·세로 축척이 다르다. 회색은 첫 광선, 주황은 둘째 광선, 파랑은 고정 단차다. h는 차폐물 끝의 높이다.</p>
<div class="mat"><math aria-label="그림자 위치의 변화는 마이너스 h 나누기 H 빼기 h 곱하기 광원의 수평 변화"><mi>Δxₛ</mi><mo>=</mo><mo>−</mo><mfrac><mi>h</mi><mrow><mi>H</mi><mo>−</mo><mi>h</mi></mrow></mfrac><mi>Δxᴸ</mi></math></div>
</div></div>
<div class="split"><div class="pane"><h4>같은 위치에 남는 반례</h4><p>차폐물이 바닥과 맞닿으면 h=0이다. 광원을 옮겨도 이 접촉점은 움직이지 않는다. 바닥에 그린 줄도 고정된다. 실제 깊이 단차도 빛에 따라 대비가 약해지면 사진에서 사라질 수 있다.</p></div><div class="pane bad"><h4>두 경계가 한 픽셀에 겹치는 반례</h4><p>첫 조명에서 그림자와 소파 모서리가 x=0에 겹치고, 둘째 조명에서 그림자만 x=8로 간다고 하자. 첫 사진에 기록된 밝기 점프는 두 효과의 합이다. 같은 합을 만드는 조합이 여럿이고 두 효과가 상쇄될 수도 있다.</p></div></div>
<figure><img loading="lazy" src="{{OVER}}" alt="합성 거실의 정답 깊이 단차와 광선 차단 경계를 겹쳐 그린 영상"><figcaption>실제 재렌더의 겹침. 파랑은 참 깊이 단차, 주황은 렌더의 광선 차단 경계, 분홍은 두 경계의 1픽셀 띠가 겹친 곳이다. 표시 격자는 가로 512픽셀이다.</figcaption></figure>
</section><section><div class="sec-head"><span class="sec-num">06</span><h2>경계를 골랐을 때 형상이 좋아질 조건</h2></div>
<p>평소 비교 기준은 미터 깊이 두 장의 산술 평균 <i>m=(A+B)/2</i>이다. 조명 A에서만 그림자로 보이는 선이 있다면 그 자리에서는 B의 깊이 기울기를 빌린다. 서로 이웃한 점들의 지시를 모아 한 장의 깊이 지도로 복원한다.</p>
<div class="mat">b = log m　　(DᵀD + ρI)v = DᵀH　　Z = m · exp(λv)</div>
<p><i>D</i>는 이웃끼리 뺄셈하는 연산이다. <i>H</i>는 빌릴 기울기와 기준 기울기의 차, <i>v</i>는 지도를 얼마나 고칠지, <i>λ</i>는 그 세기다. <i>ρ=1/16</i>은 주변에 근거가 없을 때 큰 수정을 억제한다. 로그 보정의 영상 전체 평균을 빼서 기준 깊이 지도의 기하평균을 보존한다.</p>
<div class="spec"><div class="spec-head"><span class="badge b-d">유도</span><h3>근거 있는 기울기만 이득을 준다</h3></div><div class="spec-body">
<p>16픽셀 떨어진 두 점의 로그 깊이 차를 본다. <i>r</i>은 참 형상에서 현재 형상을 뺀 잔차, <i>u</i>는 보정이 제안하는 두 점의 변화다. 지도 전체를 두 배 확대해도 로그 깊이의 차는 같다.</p>
<div class="mat">G(λ) = E[r² − (r − λu)²] = 2λ E[ru] − λ² E[u²]</div>
<p><i>a=E[ru]</i>와 <i>c=E[u²]&gt;0</i>라 두면, 양의 강도에서 오차가 줄어드는 정확한 조건은 <b>a&gt;0이고 λ&lt;2a/c</b>이다. 합성 적합 장면에서 <i>λ=clip(a/c,0,1)</i>로 정한 뒤 실사에서는 고정한다. <i>c=0</i>이면 보정이 형상에 주는 변화도 0이다.</p>
<details class="fold"><summary>왜 지도를 하나로 복원할 수 있을까</summary><div class="fold-body"><p>목적함수는 ‖Dv−H‖²+ρ‖v‖²이다. v로 미분해 0을 놓으면 위 선형식이 나온다. 0이 아닌 x에 대해 xᵀ(DᵀD+ρI)x=‖Dx‖²+ρ‖x‖²&gt;0이므로 유일한 해가 있다. 고정 경계에서 H=0으로 두는 것은 그 위치의 최종 기울기를 강제로 같게 하는 제약과는 다르다.</p></div></details>
</div></div><p class="sources"><a href="../../review/occlusion42_43_20261124/EDGE_MOTION_PROTOCOL.md">전체 가정·증명·검증 절차</a></p></section></article>

<article class="sheet" id="results" role="tabpanel" aria-labelledby="t-results"><section><div class="sec-head"><span class="sec-num">07</span><h2>같은 장면으로 무엇을 비교했나</h2></div>
<div class="split"><div class="pane"><h4>합성 · 원인을 바꿀 수 있는 시험</h4><p>Hypersim의 미터 거리, 법선, 재질 반사율, 노드 ID를 읽었다. 여러 시점의 삼각형으로 광원까지의 가림을 계산하고, 같은 카메라에서 무광과 유광 코팅을 세 방향으로 비췄다. 적합 32장면과 검증 16장면은 장면 접두어까지 분리했다.</p></div>
<div class="pane"><h4>실사 · 실제 사진에 옮겨 가는 시험</h4><p>Middlebury의 동일 카메라·동일 셔터 사진 14쌍을 11계열로 균등 집계했다. 정답 시차로 미터 깊이를 계산했다. 어떤 그림자가 어느 광원에 의해 생겼는지 알려주는 실사 정답은 제공되지 않는다. 이 실사 장면은 기존 연구에서도 사용했다.</p></div></div>
<figure><img loading="lazy" src="{{RENDER}}" alt="같은 거실의 무광과 유광 코팅을 세 광원으로 다시 렌더링한 여섯 영상과 가림·반사 마스크"><figcaption>맨 아래 파랑은 광원으로 향하는 광선의 차단, 주황은 큰 거울반사 성분이다. 이 렌더는 복원한 기하에서의 직접광을 계산한다. 보이지 않았던 차폐물과 간접광은 한계다.</figcaption></figure>
<div class="spec"><div class="spec-head"><span class="badge b-c">실측</span><h3>픽셀 경계를 찾는 규칙</h3></div><div class="spec-body"><p>가로 512 격자에서 두 사진의 로그 밝기를 부드럽게 한 뒤 가로·세로 단면을 살핀다. 같은 부호의 밝기 점프가 서로 가장 가까운 짝이고 1픽셀 안에 남으면 고정 후보, 4~20픽셀 옮겨 가며 원위치에서 사라지면 이동 후보다. 고정 후보 근처와 짝이 불명확한 곳은 유보한다. 정답 깊이와 그림자 라벨은 검출기에 넣지 않았다.</p></div></div>
<div class="pair"><figure><img loading="lazy" src="{{EDGE}}" alt="합성 거실의 두 조명 원본과 고정·이동 RGB 경계"><figcaption>Hypersim 거실 · 위는 조명 두 장, 아래는 고정 후보(파랑)와 이동 후보(주황).</figcaption></figure>
<figure><img loading="lazy" src="{{BOOKS}}" alt="실제 Middlebury Books의 두 조명 원본과 고정·이동 RGB 경계"><figcaption>Middlebury Books · 책등처럼 고정된 선이 많다. 그 선 모두가 깊이 단차인 것은 아니다.</figcaption></figure></div></section>
<section><div class="sec-head"><span class="sec-num">08</span><h2>형상 오차가 반반 평균보다 줄었는가</h2></div>
<p class="lead">합성에서 정한 이동 경계 보정 강도는 <strong>{{LAMBDA}}</strong>였다. 합성과 실사의 표면 오차 평균은 조금 줄었지만, 두 자료 모두 계열 재표집 구간에 0이 들어갔다.</p>
<div class="tscroll"><table><thead><tr><th>자료</th><th>반반 MSE</th><th>이동 경계 MSE</th><th>상대 개선</th><th>95% 재표집 구간</th><th>개선 계열</th></tr></thead><tbody>{{ROWS}}</tbody></table></div>
<p class="compact">양수는 오차 감소. 합성은 같은 렌더 노드에 속한 두 점, 실사는 정답 깊이비가 1.05 미만인 두 점이다. 가로·세로로 16픽셀 떨어진 점에서 (예측 로그 깊이 차 − 참 로그 깊이 차)²을 원해상도로 채점했다. 동일 렌더 노드 또는 비슷한 깊이는 동일 평면의 근사다.</p>
<details class="fold"><summary>점쌍과 대조를 바꾸면</summary><div class="fold-body"><div class="tscroll"><table><thead><tr><th>점쌍</th><th>합성 이동</th><th>실사 이동</th><th>합성 한쪽 경계</th><th>실사 한쪽 경계</th></tr></thead><tbody>{{AUX}}</tbody></table></div><p>한쪽 경계 대조는 같은 선이 어디로 갔는지 찾지 않고 한 사진에만 나타난 자리를 고른다. 공간 위치를 옆으로 옮긴 대조는 합성에서 강도 0이 선택되었다. {{PLACEBO}} 합성의 표면과 단차 집합은 일부 겹칠 수 있다.</p></div></details>
<div class="split"><div class="pane"><h4>위치를 얼마나 잘 찾았나</h4><p>합성 이동 대응 끝점 33,202건 중 11,396건({{MOTIONP}})에서 두 위치의 실제 광선 차단 경계와 원위치 소실을 확인했다. 이 수치는 같은 물리 차폐물을 추적했다는 뜻까지 담지는 않는다.</p>
<p>실사 이동 후보는 317개였고 53개가 참 깊이 단차 근처였다. 고정 후보 중 참 깊이 단차 근처는 20,387/156,653개였다. 반대 방향으로 보면 참 단차 위치 중 {{EDGERECALL}}에서 2픽셀 이내에 고정 후보가 있었다.</p></div>
<div class="pane"><h4>렌더러 정답을 써 본 진단</h4><p>합성에서만 실제 가림 경계가 한 사진에만 있는 곳을 고르고, 진짜 깊이 단차를 피하도록 해 봤다. 보류 합성 장면의 표면 오차 개선은 {{ORACLE}}였다. 실사에서는 이 정답 마스크를 구할 수 없다. 해당 측정은 이 보정 방식의 진단이다.</p></div></div>
</section><section><div class="sec-head"><span class="sec-num">09</span><h2>앞선 실험과 현재 판정</h2></div>
<div class="tscroll"><table><thead><tr><th>질문</th><th>기존 → 비교</th><th>해석</th></tr></thead><tbody>
<tr><th>#43 밝기만 보고 미터 깊이를 고르면</th><td>Middlebury 반반 → 조건별, 사진별 추정 초점에서 +1.17%. 같은 실제 초점에서는 −0.44%</td><td>그림자 후보 부분은 실제 초점에서도 +0.54%. 전체 배율과 국소 형상을 분리해야 한다.</td></tr>
<tr><th>#42 단안 인코더 학습</th><td>같은 예산 지도 → 조건별 게인. DIODE 0.335610 → 0.336166, iBims 0.269547 → 0.269653</td><td>상대 AbsRel 개선은 −0.166%, −0.039%. 이번 1시드 적응은 3% 목표에 미달.</td></tr>
<tr><th>강한 합성·실사 참조</th><td>배율 1의 DIODE에서 #42 학생 0.292746, DA V2 Hypersim 대형 0.228539, NYU 실사 지도 0.350363</td><td>학습 자료·모델 크기·예산이 다른 참조. iBims에서는 NYU 실사 지도 0.187852, 학생 0.189882.</td></tr></tbody></table></div>
<p class="compact">미터 오차 AbsRel은 평균 |예측 깊이−참 깊이|/참 깊이이다. 예측 2.2m와 정답 2m의 AbsRel은 0.1이다. 형상 오차 MSE는 두 점 사이의 로그 깊이 차가 얼마나 틀렸는지 제곱해서 평균한다. 두 지표의 개선률은 따로 읽는다.</p><div class="verdict"><h3>정리하면</h3><ul><li><b>증명한 범위</b>는 독립적으로 식별한 경계의 이동 분리와, 보정 방향이 정답 잔차에 충분히 맞을 때의 형상 개선 조건이다.</li><li><b>실제 병목</b>은 사라진 그림자·반사·겹친 깊이 단차 사이에서 같은 선을 다시 찾는 일이다. 현재 규칙은 실사에서 이동 후보가 드물고 안정적인 형상 개선이 확인되지 않았다.</li><li><b>다음 연구</b>는 각 물리 경계가 여러 광원에서 언제 나타나고 사라지는지 추적하고, 같은 검출 면적과 강도의 대조에서 교사의 이득을 확인하는 것이다. 다음 단안 학생 학습은 별도 시험이다.</li></ul></div>
<p class="sources"><a href="../../review/occlusion42_43_20261124/EDGE_MOTION_RESULT.md">이번 결과·원자료</a><a href="../../review/occlusion42_43_20261124/edge_motion43.py">경계 검출·채점 코드</a><a href="../../review/occlusion42_43_20261124/MATERIAL_LIGHT_RESULT.md">재질·광원 실험</a><a href="../../review/occlusion42_43_20261124/LIGHT_ENCODER_RESULT.md">단안 학생 실험</a></p>
<p class="sources">재렌더 원자료와 이미지 · Mike Roberts 등, Apple Hypersim, CC BY-SA 3.0. 이 노트의 재렌더·오버레이는 수정본이다. <a href="https://github.com/apple-aiml-research/ml-hypersim">Hypersim</a><a href="https://creativecommons.org/licenses/by-sa/3.0/">이미지 라이선스</a><a href="https://vision.middlebury.edu/stereo/data/scenes2005/">Middlebury 2005</a><a href="https://vision.middlebury.edu/stereo/data/scenes2006/">Middlebury 2006</a></p></section></article>
</main><footer><div class="wrap" style="padding-bottom:0">바이브연구 · #42 합성 깊이 학습 / #43 조명 통제 실험</div></footer>
<script>
const buttons=Array.from(document.querySelectorAll('[role="tab"]')), panels=Array.from(document.querySelectorAll('[role="tabpanel"]'));
function show(){const hash=location.hash.slice(1), id=panels.some(p=>p.id===hash)?hash:'overview';
buttons.forEach(b=>{const on=b.getAttribute('aria-controls')===id;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});
panels.forEach(p=>p.hidden=p.id!==id);}
function select(i){const b=buttons[i];location.hash=b.getAttribute('aria-controls');show();b.focus({preventScroll:true});window.scrollTo(0,0);}
buttons.forEach((b,i)=>{b.addEventListener('click',()=>select(i));b.addEventListener('keydown',e=>{let j=i;if(e.key==='ArrowRight')j=(i+1)%4;else if(e.key==='ArrowLeft')j=(i+3)%4;else if(e.key==='Home')j=0;else if(e.key==='End')j=3;else return;e.preventDefault();select(j);});});
window.addEventListener('hashchange',show);show();
function ray(){const get=id=>document.getElementById(id),set=(id,k,v)=>get(id).setAttribute(k,v);
const mode=get('case').value,dx=Number(get('lamp').value),h=mode==='touch'?0:20,H=60,xb=0,xLight=mode==='overlap'?-40:-8;
const px=x=>400+3*x,py=y=>140-y*116/H; const x1=(H*xb-h*xLight)/(H-h),x2=(H*xb-h*(xLight+dx))/(H-h);
set('line1','d','M'+px(xLight)+' 24 L'+px(x1)+' 140');set('line2','d','M'+px(xLight+dx)+' 24 L'+px(x2)+' 140');
set('lamp1','cx',px(xLight));set('lamp2','cx',px(xLight+dx));set('s1','cx',px(x1));set('s2','cx',px(x2));set('object','d','M400 140 V'+py(h));
get('lampval').value=dx;get('readout').textContent='차폐물 높이 h = '+h+' · 광원 이동 '+dx+' · 바닥 그림자 이동 '+(x2-x1).toFixed(1)+' 길이 단위. '+(mode==='touch'?'접촉점이 고정된 선처럼 보인다.':mode==='overlap'?'첫 그림자는 파란 단차와 같은 위치다.':'그림자는 광원 이동과 반대로 간다.');}
document.getElementById('lamp').addEventListener('input',ray);document.getElementById('case').addEventListener('change',ray);ray();
</script></body></html>'''


if __name__=="__main__":
    build()
