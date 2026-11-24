"""Recover all paired policy contrasts from the archived forecasts and outcomes."""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from swing import BAND,G,L,R,LEVELS
from evaluate import choices,bootstrap

DOMAINS=("source","shift")
METHODS=("legacy","prior_photo","prior_probe","flow_probe","prior_swapped","flow_swapped")


def summarize():
    results={"methods":{},"contrasts":{},"identification":{}}
    v={}
    with np.load(HERE/"forecasts.npz") as f:
        for d in DOMAINS:
            y=abs(f["peak_"+d]-f["goal_"+d])<=BAND
            denominator=len(y)
            results["identification"][d]={
                "oracle":float(y.any(1).mean()),
                "strength_MAE_s2":float(abs(f["k_"+d]-G*np.sin(np.deg2rad(LEVELS[d][f["level_"+d]]))
                                            /(L+.4*R*R/L)).mean())}
            for name in METHODS:
                keys=sorted(k for k in f if k.startswith(name+"_") and k.endswith("_"+d))
                assert len(keys)==(1 if name.startswith("prior_") else 3)
                p=np.stack([f[k] for k in keys])
                assert np.isfinite(p).all()
                a=choices(p)
                picked=y[np.arange(len(y)),a]
                v[name,d]=picked.mean(0)
                results["methods"].setdefault(name,{})[d]={
                    "success":float(picked.mean()),"by_seed":picked.mean(1).tolist(),
                    "chosen_peak_MAE_deg":float(np.rad2deg(abs(f["peak_"+d][np.arange(len(y)),a]
                                                               -f["goal_"+d][:,0])).mean())}
        contrasts={}
        for first,second in (("flow_probe","legacy"),("prior_probe","prior_photo"),("flow_probe","prior_probe")):
            for d in DOMAINS:
                contrasts[f"{first}-minus-{second}_{d}"]=v[first,d]-v[second,d]
        ci=bootstrap(np.column_stack(list(contrasts.values())),comparisons=len(contrasts))
        for (k,c),bounds in zip(contrasts.items(),ci):
            results["contrasts"][k]=[float(c.mean()),*bounds.tolist()]
        results["paired_starts"]=denominator
        # Illustrative near-60-degree target with one successful action per domain, different winners.
        labels=[abs(f["peak_"+d]-f["goal_"+d])<=BAND for d in DOMAINS]
        ids=np.flatnonzero((labels[0].sum(1)==1)&(labels[1].sum(1)==1)&~(labels[0]&labels[1]).any(1))
        i=int(ids[np.abs(np.rad2deg(f["goal_source"][ids,0])-60).argmin()])
        ex={"index":i,"goal_deg":float(np.rad2deg(f["goal_source"][i,0]))}
        for d,y in zip(DOMAINS,labels):
            ex[d]={"practice_deg":float(np.rad2deg(f["start_probe_angles_"+d][i,1])),
                   "strength":float(f["k_"+d][i]),"correct_action":int(y[i].argmax()+1),
                   "prior_action":int(choices(f["prior_probe_"+d][i])+1),
                   "full_actions":[int(choices(f[f"flow_probe_{s}_{d}"][i])+1) for s in (4821,4822,4823)]}
        results["example"]=ex
    return results


if __name__=="__main__":
    result=summarize()
    (HERE/"results.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    for k,row in result["methods"].items():
        print(k,*[round(row[d]["success"],4) for d in DOMAINS])
    print(result["contrasts"],result["example"])
