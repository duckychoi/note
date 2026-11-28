"""Equal physical-scene weights and paired scene/seed resampling; generate tables from scores."""
import json
import numpy as np

from common import ROOT,SEEDS,TEST,save

CAND="lines_order"
NAMES=dict(constant="상수 거리",legacy="기존 쌍 효용 모델",plain_depth="새 교차·추가 선 없음",plain_order="새 교차·추가 선 없음 + 순서",
           lines_depth="끊기는 선·깊이 학습",lines_order="끊기는 선 + 직접 순서 학습",grid_shrink="기존 출력의 전역 이웃 진폭 축소",
           exact_old="기존 출력의 채점 경계 평균 진폭 일치",pair_exact_sign="쌍별 후보 크기 + 기존 부호",
           small_old="기존 출력, 둘 중 작은 평균 진폭",small_new="후보 출력, 둘 중 작은 평균 진폭")


def load(name):
    return json.loads((ROOT/(name+".json")).read_text(encoding="utf8"))


def main():
    rows=load("scores")
    keyed={(r["subset"],r["budget"],r["seed"],r["scene"],r["arm"]):r for r in rows}
    assert len(rows)==len(keyed)==2*2*len(SEEDS)*len(TEST)*len(NAMES)
    cases=load("amplitude_checks")
    out={}
    tables=["# 장면 단위 집계","", "순서 정답률은 %로 표기했다. 전체 깊이와 단차 오차, 예측 단차 진폭은 mm다. 동점은 0.5점을 준다. 각 물리 장면에 같은 비중을 준 뒤 학습 시드 평균을 구했다.",""]
    for subset in ("strict","wide"):
        for budget in ("zero","few"):
            key=subset+"_"+budget
            result=dict(means={},contrasts={},counts={})
            out[key]=result
            for metric in ("whole","jump","order","strict","tie","amplitude"):
                scenes=[s for s in TEST if keyed[subset,budget,SEEDS[0],s,CAND][metric] is not None]
                fam=sorted({s[:-1] for s in scenes})
                result["counts"][metric]=dict(views=len(scenes),families=len(fam),names=fam)
                matrix={}
                for a in NAMES:
                    if keyed[subset,budget,SEEDS[0],scenes[0],a][metric] is None:
                        continue
                    matrix[a]=np.array([[np.mean([keyed[subset,budget,i,s,a][metric] for s in scenes if s[:-1]==k]) for k in fam] for i in SEEDS])
                    result["means"].setdefault(a,{})[metric]=float(matrix[a].mean())
                if metric not in ("whole","jump","order"):
                    continue
                rng=np.random.default_rng(423512)
                g=rng.integers(len(fam),size=(20_000,len(fam)))
                si=rng.integers(len(SEEDS),size=(20_000,len(SEEDS)))
                targets=[(CAND,a) for a in matrix if a!=CAND and a not in ("small_new","small_old")]
                targets += [("small_new","small_old"),("plain_order","plain_depth"),("lines_depth","plain_depth"),
                            ("interaction","factorial")]
                for a,b in targets:
                    if a=="interaction":
                        d=(matrix[CAND]-matrix["lines_depth"])-(matrix["plain_order"]-matrix["plain_depth"])
                    else:
                        d=matrix[a]-matrix[b]
                    boot=d.mean(0)[g].mean(1)
                    cross=d[si[:,:,None],g[:,None,:]].mean((1,2))
                    result["contrasts"].setdefault(a+"-"+b,{})[metric]=dict(delta=float(d.mean()),
                        lower=float(np.quantile(boot,.05)),upper=float(np.quantile(boot,.95)),
                        joint_lower=float(np.quantile(cross,.05)),joint_upper=float(np.quantile(cross,.95)),
                        per_seed=d.mean(1).tolist(),per_family=dict(zip(fam,d.mean(0).tolist())))
            sub=[v for v in cases if v["eligible"] and v["subset"]==subset and v["budget"]==budget]
            result["amplitude"]=dict(total=len(sub),expanded=sum(v["exact_ratio"]>1 for v in sub if v["exact_ratio"] is not None),
                impossible=sum(v["impossible"] for v in sub),ties=sum(v["old_ties"] for v in sub))
            comparisons=[result["contrasts"][CAND+"-"+r] for r in ("constant","legacy","exact_old")]
            result["positive_components"]=sum(int(c[m]["lower"]>0 if m=="order" else c[m]["upper"]<0)
                                              for c in comparisons for m in ("order","jump","whole"))
            result["joint_pass"]=result["positive_components"]==9 and result["counts"]["order"]["families"]>=5
            tables+=["## "+key,"","| 모델 | 순서 % | 단차 오차 | 전체 오차 | 단차 진폭 |","|---|---:|---:|---:|---:|"]
            for a in NAMES:
                v=result["means"][a]
                whole=f'{v["whole"]:.1f}' if "whole" in v else "—"
                tables.append(f'| {NAMES[a]} | {100*v["order"]:.2f} | {v["jump"]:.1f} | {whole} | {v["amplitude"]:.1f} |')
            tables+=["",f'전체 {result["counts"]["whole"]["families"]}장면 계열, 경계 {result["counts"]["order"]["families"]}계열. '
                       f'확장 허용 정확 진폭 지도에서 기존 모델 확대 {result["amplitude"]["expanded"]}/{len(sub)}.','']
    save("summary.json",out)
    ledgers=[load(f"train_{seed}") for seed in SEEDS]
    tables+=["## 별도 합성 반전 시험","","앞뒤 상태별 평균 및 두 상태를 모두 맞힌 비율이다. 8개 독립 합성 장면/시드에서 같은 좌표 쌍을 두 상태에 대입했다.","",
             "| 모델 | 상태별 정답률 % | 두 상태 모두 정답 % | 예측 순서 반전 % |","|---|---:|---:|---:|"]
    for a in ("legacy","plain_depth","plain_order","lines_depth",CAND):
        scores=np.mean([[v["audit"][a][k] for k in ("order","both","flipped")] for v in ledgers],axis=0)*100
        tables.append("| "+NAMES[a]+" | "+" | ".join(f"{v:.2f}" for v in scores)+" |")
    (ROOT/"TABLES.md").write_text("\n".join(tables)+"\n",encoding="utf8")
    print("Wrote TABLES.md and summary.json")


if __name__=="__main__":
    main()
