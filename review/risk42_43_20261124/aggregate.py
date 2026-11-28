"""Equal scene-family and seed means with paired superiority limits."""
import json
import numpy as np

from acquire import HERE,TEST
from fit import SEEDS


def summarize(rows,filename,comparisons,primary):
    lut={(r["budget"],r["arm"],r["seed"],r["scene"]):r for r in rows}
    groups={}
    for name in TEST:
        groups.setdefault("Sword" if name in ("Sword1","Sword2") else name,[]).append(name)
    arms=sorted({r["arm"] for r in rows})
    output={}
    for budget in ("zero","few"):
        out=dict(means={},differences={})
        output[budget]=out
        for metric in ("whole","jump","front","back","sign"):
            keep=[[s for s in names if lut[budget,"utility",SEEDS[0],s][metric] is not None] for names in groups.values()]
            keep=[names for names in keep if names]
            if not keep:
                continue
            g=len(keep)
            rng=np.random.default_rng(423117)
            gi=rng.integers(g,size=(20_000,g))
            si=rng.integers(len(SEEDS),size=(20_000,len(SEEDS)))
            def matrix(arm):
                return np.array([[np.mean([lut[budget,arm,seed,s][metric] for s in ns]) for ns in keep] for seed in SEEDS])
            matrices={a:matrix(a) for a in arms}
            for arm,m in matrices.items():
                out["means"].setdefault(arm,{})[metric]=float(m.mean())
            if metric not in ("whole","jump"):
                continue
            for cand,baselines in comparisons:
                for ref in baselines:
                    diff=matrices[cand]-matrices[ref]
                    by_scene=diff.mean(0)
                    boots=by_scene[gi].mean(1)
                    both=diff[si[:,:,None],gi[:,None,:]].mean((1,2))
                    record=dict(mean=float(diff.mean()),upper95=float(np.quantile(boots,.95)),
                        lower95=float(np.quantile(boots,.05)),joint_upper95=float(np.quantile(both,.95)),
                        scene_wins=int((by_scene<0).sum()),scene_groups=g,
                        seed_differences=diff.mean(1).tolist())
                    out["differences"].setdefault(cand+"-"+ref,{})[metric]=record
        if primary:
            gates=[out["differences"]["utility-"+a][k] for a in ("constant","replay_raw","replay") for k in ("whole","jump")]
            out["primary_components_pass"]=sum(r["mean"]<0 and r["upper95"]<0 for r in gates)
            out["pass"]=bool(out["primary_components_pass"]==6 and gates[1]["scene_groups"]>=5)
        print(budget)
        for a,v in out["means"].items():
            print(a,*(round(v[k],1) for k in ("whole","jump","front","back")),round(v["sign"]*100,1))
        for name,d in out["differences"].items():
            print(name,[(round(v["mean"],1),round(v["upper95"],1),round(v["joint_upper95"],1)) for v in d.values()])
    (HERE/(filename+".json")).write_text(json.dumps(output,indent=1,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    primary=json.loads((HERE/"scene_scores.json").read_text())
    comparisons=(("utility",("constant","replay_raw","replay","prior","correct","shuffle","true_utility")),
                 ("student_stereo",("raw_stereo","fixed_stereo","replay_stereo","utility")))
    summarize(primary,"summary",comparisons,True)
