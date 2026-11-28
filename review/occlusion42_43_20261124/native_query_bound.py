"""Measure finite-step interval lower bounds on physical 1->5 TAP query pairs."""
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/"research-sources/encoder42_eval"
OUT=ROOT/"native-surface-direction"

def gain(q,h):
    return abs(q-1)-abs(q*np.exp(h)-1)

def lower(lo,hi,h):
    qs=(lo,hi,np.clip(np.ones_like(lo),lo,hi),np.clip(np.exp(-h),lo,hi))
    return np.min([gain(q,h) for q in qs],axis=0)

def balanced(h):
    pos=np.maximum(0,h);neg=np.maximum(0,-h)
    a,b=pos.sum(),neg.sum()
    if min(a,b)==0:
        return 0*h
    return pos*min(1,b/a)-neg*min(1,a/b)

def main():
    rows=[]
    with np.load(OUT/"tap-probe/result.npz") as obs:
        for path in sorted((OUT/"middlebury-native").glob("*.npz")):
            scene=path.stem
            with np.load(path) as d:
                x,y=obs[scene+"/query_xy"].astype(int).T
                z0=d["base"][y,x].astype(float)*float(d["k"])
                student=d["h"][y,x].astype(float)
            dmin=float((ROOT/f"repeat-capture/physical/{scene}/dmin.txt").read_text())/3
            A=(3740/3)*.16
            prefix=scene+"/L1-L1/view5/"
            rho=obs[prefix+"gt_inverse_depth_per_m"]
            valid=obs[prefix+"gt_eligible"]&(rho>=1/30)&(rho<=4)
            q=z0*rho
            ds=[]
            seen=np.ones(len(x),bool)
            for light in (1,2):
                field=f"{scene}/L1-L{light}/view5/"
                delta=x-obs[field+"tap_xy"][:,0]+dmin
                seen &= obs[field+"tap_visible"]&np.isfinite(delta)
                ds.append(delta)
            qlo=z0*(np.minimum(*ds)-1)/A
            qhi=z0*(np.maximum(*ds)+1)/A
            seen &= (qlo>0)&(qhi>=qlo)
            # Unseen queries get zero teacher move. Do not use GT to normalize its gauge.
            h=np.zeros_like(q)
            near=seen&(qlo>1);far=seen&(qhi<1)
            h[near]=-np.log(qlo[near]);h[far]=-np.log(qhi[far])
            h=balanced(np.clip(h,-.0875,.0875))
            lo=np.where(seen,qlo,z0/30);hi=np.where(seen,qhi,z0/.25)
            contained=(q>=lo)&(q<=hi)
            row=dict(scene=scene,group=re.sub(r"\d+$","",scene),points=int(valid.sum()),
                     joint_visible=int((seen&valid).sum()),covered=float(contained[seen&valid].mean()))
            for name,correction in (("teacher",h),("deployed",student)):
                lb=lower(lo,hi,correction)
                actual=gain(q,correction)
                if np.any(lb[valid&contained]>actual[valid&contained]+2e-7):
                    raise ValueError("Finite-step inequality failed on covered support")
                row[name]=dict(conditional_bound=float(lb[valid].mean()),actual=float(actual[valid].mean()),
                               positive_pixels=int(((lb>1e-10)&valid).sum()))
            rows.append(row)
    def avg(fn):
        groups=defaultdict(list)
        for r in rows:
            groups[r["group"]].append(fn(r))
        return float(np.mean([np.mean(xs) for xs in groups.values()]))
    summary={k:{m:avg(lambda r:r[k][m]) for m in ("conditional_bound","actual")} for k in ("teacher","deployed")}
    summary["covered_on_visible"]=avg(lambda r:r["covered"])
    target=OUT/"tap-probe/finite-step.json"
    target.write_text(json.dumps(dict(summary=summary,rows=rows),indent=1),encoding="utf8")
    print(summary)

if __name__=="__main__":
    main()
