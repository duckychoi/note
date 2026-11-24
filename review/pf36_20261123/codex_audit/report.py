"""Recover original gate decisions and compare frozen models on independent paired tests."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
METRICS = ("energy","cover_x","cover_z","brier_bounce2","brier_xpast")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def interval(blocks, comparisons):
    y = np.asarray(blocks)
    radius = t.ppf(1-.05/(2*comparisons),len(y)-1)*np.std(y,ddof=1)/np.sqrt(len(y))
    return [float(y.mean()-radius),float(y.mean()+radius)]


def original():
    g5, g4, diag, calib = (read(SOURCE/name) for name in
        ("gate5.json","gate4a.json","gate5_diag.json","gate5_calib.json"))
    ret = g4["table"][g4["best_retrieval"]]
    ref = g4["reference"]["pf_N8192"]
    out = dict(retrieval=ret,reference=ref,matched_pf=g4["reference"]["pf_N1024"],
               roles={}, records=[], calibration={}, diagnostics=diag["new_wins"])
    for role in ("primary","secondary"):
        runs = [r for r in g5["runs"] if r["role"]==role]
        scores = np.array([r["energy_per_seed"] for r in runs])
        per = scores.mean(1)
        np.testing.assert_allclose(per,[r["energy"] for r in runs])
        passed = ((per<=1.05*ref["energy"]) & (per<=.95*ret["energy"]) &
                  (scores<np.asarray(ret["energy_per_seed"])).all(1))
        np.testing.assert_array_equal(passed,[r["passed"] for r in runs])
        np.testing.assert_allclose([per.mean(),per.std(ddof=1)],g5["aggregate"][role]["energy"])
        m,n = runs[0]["M"],runs[0]["N"]
        out["roles"][role] = dict(score=float(per.mean()),sd=float(per.std(ddof=1)),
            passed=int(passed.sum()),calls=m+192*n,ref_calls=192*8192,
            retrieval_drop=float(1-per.mean()/ret["energy"]), matched_retrieval_pf=g4["table"][f"rpf_L{m}_N{n}"]["energy"],
            **{k:float(np.mean([r[k] for r in runs])) for k in METRICS[1:]})
        for r,passed_check in zip(runs,passed):
            out["records"].append(dict(role=role,seed=r["train_seed"],energy=r["energy"],passed=bool(passed_check)))
    for k in ("cover_x","cover_z"):
        out["calibration"][k]=[float(np.mean([r["val"]["0.1"][k] for r in calib["rows"]])),
                               float(np.mean([r["after"][k] for r in calib["rows"]]))]
    with np.load(HERE/"primary_replay.npz") as replay:
        differences = [abs(replay[f"{r['train_seed']}_{i}_energy"].mean()-e)
            for r in g5["runs"] if r["role"]=="primary" for i,e in enumerate(r["energy_per_seed"])]
        out["max_energy_replay_error"] = float(max(differences))
    history = read(HERE/"patience.json")
    assert [v["patience"] for v in history]==[5,15]
    assert history[0]["curve"]==history[1]["curve"][:len(history[0]["curve"])]
    out["paired_training"] = []
    for v in history:
        with np.load(HERE/f"patience_{v['patience']}.npz") as raw:
            e = float(np.mean([raw[f"{i}_energy"] for i in range(4)]))
        np.testing.assert_allclose(e, v["energy"])
        out["paired_training"].append({k:v[k] for k in ("training_seed","epochs","patience","best_val_nll","energy","temperature")})
    old_run = read(SOURCE/"gate4b.json")["table"]["flowpf_broad_M98304_N512"]
    out["original_failed_run"]=dict(energy=old_run["energy"], **old_run["train"])
    return out


def fresh():
    methods = ("flow_8100","flow_8200","flow_8300","retrieval","original","tuned")
    scores = {m:{k:[] for k in METRICS} for m in methods}
    inner = {k:[] for k in ("cover_x","cover_z")}
    for seed in range(56370,56378):
        with np.load(HERE/f"fresh_{seed}.npz") as f, np.load(HERE/f"inner_{seed}.npz") as inside:
            np.testing.assert_array_equal(f["case_mask"],inside["case_mask"])
            for m in methods:
                for k in METRICS:
                    values = f[f"{m}_{k}"]
                    assert len(values)==24 and np.isfinite(values).all()
                    scores[m][k].append(float(np.mean(values)))
            for k in inner:
                inner[k].append(float(np.mean([inside[f"{m}_{k}"] for m in methods[:3]])))
    scores["flow_mean"] = {k:np.mean([scores[m][k] for m in methods[:3]],axis=0).tolist() for k in METRICS}
    means = {m:{k:float(np.mean(b)) for k,b in d.items()} for m,d in scores.items()}
    flow = np.array(scores["flow_mean"]["energy"])
    comparisons = {}
    for m,multiple in (("retrieval",.95),("original",1.05),("tuned",1.05)):
        other = np.array(scores[m]["energy"])
        margin = multiple*other-flow
        comparisons[m] = dict(margin=float(margin.mean()),ci=interval(margin,3),
            wins=int((flow<other).sum()),rel_change=float(flow.mean()/other.mean()-1))
    effect = {}
    for k,values in inner.items():
        outside = np.array(scores["flow_mean"][k])
        diff = outside - values
        effect[k]=dict(inside=float(np.mean(values)),outside=float(outside.mean()),delta=float(diff.mean()),ci=interval(diff,2))
    return dict(means=means, blocks=scores, comparisons=comparisons, paired_coverage=effect,
                temperature=read(HERE/"pf_selection.json"))


def figure(means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Malgun Gothic", "axes.unicode_minus":False, "font.size":10})
    labels = ["검색", "기존 PF", "조정 PF", "Flow-PF"]
    methods = ["retrieval", "original", "tuned", "flow_mean"]
    colors = ["#6C7D8C", "#6C7D8C", "#6C7D8C", "#176751"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1,2,figsize=(9.4,3.7),layout="constrained")
    energy = [means[m]["energy"] for m in methods]
    bars = axes[0].bar(x,energy,color=colors,width=.62)
    axes[0].bar_label(bars,fmt="%.4f",padding=3,fontsize=9)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,max(energy)*1.24),title="미래 궤적 예측 · 낮을수록 좋음",ylabel="에너지 점수")
    for k,offset,marker,c in (("cover_x",-.11,"o","#245881"),("cover_z",.11,"s","#A85824")):
        axes[1].scatter(x+offset,[100*means[m][k] for m in methods],s=52,marker=marker,color=c,label=k[-1]+" 좌표")
    axes[1].axhline(80,color="#333333",ls="--",lw=1,label="목표 80%")
    axes[1].set(xticks=x,xticklabels=labels,ylim=(0,101),title="마지막 위치의 예측구간",ylabel="실제 위치를 포함한 비율 (%)")
    axes[1].legend(frameon=False,loc="lower center",ncol=3,fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(HERE/"fresh_comparison.png",dpi=160)
    plt.close(fig)


def note(old,new):
    pct = lambda n: f"{100*n:.1f}%"
    p,s=old["roles"].values()
    groups=new["comparisons"]
    tmp=new["temperature"]
    short,longer=old["paired_training"]
    result=[
        "# #36 · 학습한 출발 입자와 물리 시뮬레이션의 독립 검수",
        "기존 여섯 조합은 문서에 정한 5차 관문을 통과했다. 원 시험의 주 분배 세 모델은 저장 가중치에서 재추론했다. 새 사례에서는 조정한 검색과 입자가 많은 PF를 각각 비교했다. 에너지 점수는 미래 위치 예측 분포를 채점하며 낮을수록 좋다.",
        "## 기존 기록과 회계",
        "| 분배 | 에너지 평균 ± 학습 3회 표준편차 | 같은 예산 검색 대비 감소 | x / z 포함률 | 통과 |\n|---|---:|---:|---:|---:|"+
        "".join(f"\n| {label} | {v['score']:.5f} ± {v['sd']:.5f} | {pct(v['retrieval_drop'])} | {pct(v['cover_x'])} / {pct(v['cover_z'])} | {v['passed']}/3 |" for label,v in (("주",p),("보조",s))),
        f"같은 배정 예산의 기본 PF는 {old['matched_pf']['energy']:.5f}, 재조정한 검색은 {old['retrieval']['energy']:.5f}, 검색으로 출발점을 제안하는 PF는 주 분배 {p['matched_retrieval_pf']:.5f}, 보조 분배 {s['matched_retrieval_pf']:.5f}를 기록했다. 큰 PF는 {old['reference']['energy']:.5f}였다. 원 시험 묶음을 재생한 주 모델의 에너지 최대 오차는 {old['max_energy_replay_error']:.3g}였다. 보조 모델은 가중치가 보관되지 않아 네 시험 묶음별 로그로 검산했다.",
        f"주 모델 하나에 배정한 궤적 수는 147,456 + 192 × 256 = {p['calls']:,}, 보조 모델 하나는 98,304 + 192 × 512 = {s['calls']:,}다. 큰 PF는 192 × 8,192 = {p['ref_calls']:,}다. 한 모델의 배정 궤적 수만 큰 PF보다 적어지는 최소 적용 건수는 주 분배 {int(np.ceil(147456/(8192-256)))}건, 보조 분배 {int(np.ceil(98304/(8192-512)))}건이다. 이 비율은 고정한 한 모델을 192사례에 적용할 때의 궤적 예산이다. 실제 원 스크립트는 공용 준비에서 넓은·좁은 사전 라이브러리를 각각 196,608개 생성했다. 학습에 사용한 궤적만 모델에 배분했다. 검증, 최적화 시간, 비교 방법들의 재실행을 합친 총 작업량이나 실측 속도는 별도다.",
        f"학습 설정 변경 후보는 분리 검증에서 6개 중 {old['diagnostics']}개를 이겨 기각됐다. 기존 시험 4묶음은 앞선 관문과 같다. 여러 학습 시드가 하나의 넓은 학습 라이브러리와 시험 사례를 공유하므로 24번의 독립 성공으로 셀 수 없다.",
        f"4b에서 실패했던 학습 시드({short['training_seed']})를 그대로 쓰고, 같은 학습 자료와 검증 규칙으로 다시 비교했다. 인내값 5 → 15에서 실제 학습은 {short['epochs']} → {longer['epochs']}에폭, 최저 검증 손실은 {short['best_val_nll']:.3f} → {longer['best_val_nll']:.3f}, 기존 192사례의 에너지 점수는 {short['energy']:.5f} → {longer['energy']:.5f}였다. 검증이 고른 PF 온도도 {short['temperature']}에서 {longer['temperature']}로 바뀌었다. 원 로그의 실패 실행은 {old['original_failed_run']['epochs']}에폭, 에너지 {old['original_failed_run']['energy']:.5f}였다. 두 재학습의 공통 구간에서 매 에폭 훈련·검증 손실이 일치했다. 이 대조는 해당 학습 시드에 한정한다. [학습 곡선](patience.json)",
        "원래 관문 3 전체를 단독 실행할 때의 재현 오류도 확인했다. pf36_g3.py의 library()는 현재 다섯 값을 반환하지만 main()은 좁은 라이브러리를 두 값으로 풀어 받는다. 작은 입력에서도 ValueError가 재현됐다. 해당 줄에서 첫 번째 위치 배열만 선택하면 된다. 이번 감사의 기준 PF는 같은 파일의 run_pf_full()을 직접 호출했다. [반환부](../pf36_g3.py#L93) · [호출부](../pf36_g3.py#L297)",
        "## 새 시험과 공정한 기준 PF",
        f"원래 환경 생성기에서 새 난수 8묶음 × 24사례를 생성했다. 주 분배 세 모형의 가중치와 검색 설정은 고정했다. 기존 PF-8192는 PF-1024로 고른 온도 {tmp['original']}를 물려받았다. 검증 48사례로 입자 8,192개에 맞춰 다시 골랐을 때 온도는 {tmp['chosen']}였다. 온도는 관측 오차를 얼마나 강하게 반영할지 정한다. 실제 새 시험 전에 고정했다.",
        "| 새 192사례에서의 방법 | 에너지 | 마지막 x / z의 80% 예측구간 포함률 |\n|---|---:|---:|"+
        "".join(f"\n| {label} | {new['means'][m]['energy']:.5f} | {pct(new['means'][m]['cover_x'])} / {pct(new['means'][m]['cover_z'])} |" for m,label in (("retrieval","재조정한 검색"),("original","기존 PF-8192"),("tuned","입자 수에 맞춰 조정한 PF-8192"),("flow_mean","Flow-PF 주 분배, 세 모델 평균"))),
        "![미사용 192사례에서의 에너지 점수와 80% 예측구간 포함률](fresh_comparison.png)",
        "| 새 시험에서 요구한 성능 | 통과선까지 여유 (양수면 점 추정 통과) | 세 비교 동시 95% 구간 | 판정 |\n|---|---:|---:|---|"+
        "".join(f"\n| {label} | {v['margin']:+.5f} | [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}] | "+("통과" if v['ci'][0]>0 else ("통과선 미달 확인" if v['ci'][1]<0 else "미확정"))+" |" for key,label in (("retrieval","검색보다 5% 이상 낮음"),("original","기존 큰 PF 대비 5% 이내"),("tuned","조정 큰 PF 대비 5% 이내")) for v in [groups[key]]),
        f"세 모델 평균이 검색보다 낮은 시험 묶음은 {groups['retrieval']['wins']}/8개다. 평균 점수 변화는 검색 대비 {100*groups['retrieval']['rel_change']:+.1f}%, 기존 큰 PF 대비 {100*groups['original']['rel_change']:+.1f}%, 조정 큰 PF 대비 {100*groups['tuned']['rel_change']:+.1f}%다. 기준에 비율을 곱한 뒤 같은 묶음에서 Flow 점수를 뺀 차이로 구간을 구했다. 8묶음 t 구간은 근사치이며 세 고정 가중치를 다시 학습할 때의 변동까지 보장하지 않는다.",
        "## 과신의 원인과 이론",
        f"기존 주 모델의 x 포함률은 검증 {pct(old['calibration']['cover_x'][0])}에서 시험 {pct(old['calibration']['cover_x'][1])}로, z는 {pct(old['calibration']['cover_z'][0])}에서 {pct(old['calibration']['cover_z'][1])}로 바뀌었다. 검증에서도 x는 80%보다 낮다. 이 구간은 모형이 공의 마지막 위치에 대해 내놓는 예측구간이다.",
        "| 별도 짝 시험 | 안쪽 물성 → 바깥 물성의 포함률 | 바깥−안쪽 차이와 두 좌표 동시 95% 구간 |\n|---|---:|---:|"+
        "".join(f"\n| {k[-1]} | {pct(v['inside'])} → {pct(v['outside'])} | {100*v['delta']:+.1f}%p [{100*v['ci'][0]:+.1f}, {100*v['ci'][1]:+.1f}] |" for k,v in new['paired_coverage'].items()),
        "짝 시험은 동일한 초기 상태, 관측 잡음, 누락 및 바운스 난수열을 사용하고 마찰·댐핑의 분포만 바꿨다. 두 좌표의 포함률 차이 구간은 0을 포함하므로 이 표본에서는 감소도 증가도 확정되지 않았다. 안쪽 물성에서도 x의 과신이 남았다. 두 물성을 함께 바꿔 각 물성의 효과는 나누지 않았다. 학습·추론은 넓은 범위의 이산 물성 144쌍을 사용하고, 시험값은 그 범위 안에서 연속으로 생성한다. 좁은 검증 범위를 벗어나지만 넓은 학습 범위 안의 격자 사이에서 평가한 것이다. 좁은 범위에서만 시뮬레이션해 학습한 기존 두 모델은 실패했다.",
        "앞부분의 관측과 누락 마스크를 보고 flow가 시작 상태와 물성 후보를 낸다. 초기 가중치에는 사전밀도를 실제 제안 혼합밀도로 나눈 비율이 들어가고, 물리 엔진으로 생성한 관측 우도를 곱한다. 초기 가중식과 RealNVP 밀도의 변수변환은 맞다. 이후 입자 위치를 흔들고 물성 격자를 이동시키는 단계에는 정확한 사후분포로 돌아가는 보정이 없다. 5차 관문은 에너지 기준이고, 앞선 연구 목표의 포함률 범위와 두 번째 환경은 별도다.",
        "## 연구 판단과 증거",
        "동일 궤적 예산의 검색을 상대로 관측을 이용해 출발 입자를 고르는 방법을 계속 검토할 가치가 있다. 같은 조건부 가우시안·혼합밀도 제안, 총실행 시간, 두 번째 환경, 최신 직접 대조가 남는다. FPPF는 조건부 생성 제안과 중요도 보정을 사용하고 LD-EnFF는 잠재 공간에서 순차 갱신을 수행한다. 초기 제안이라는 구조의 선행성 및 SOTA 우위는 이 비교로 입증되지 않았다. [FPPF](https://arxiv.org/abs/2607.01012) · [LD-EnFF](https://arxiv.org/abs/2610.04034)",
        "[기존 계획](../PREREG.md) · [원래 결과](../RESULT.md) · [독립 계획](protocol.md) · [물성 짝 시험 계획](coverage_protocol.md) · [새 기준 PF 검증 점수](pf_selection.json) · [원 시험 재실행](replay.json) · [전체 집계](audit.json)",
        "MuJoCo·torch 환경에서 replay.py, fresh.py, coverage_pair.py, patience.py로 실행한다. fresh_*.npz와 inner_*.npz에는 시험 정답·관측 및 사례별 점수가, primary_replay.npz에는 기존 주 가중치의 재평가가 있다. 보고서는 report.py로 재생성한다. report.py --check는 원자료와 저장 보고서의 일치를 검사한다.",
    ]
    return "\n\n".join(result)+"\n"


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    a,b=original(),fresh()
    data=dict(original=a,fresh=b)
    if args.check:
        assert read(HERE/"audit.json")==data
        assert (HERE/"README.md").read_text(encoding="utf8")==note(a,b)
    else:
        (HERE/"audit.json").write_text(json.dumps(data,indent=2),encoding="utf8")
        (HERE/"README.md").write_text(note(a,b),encoding="utf8")
        figure(b["means"])
