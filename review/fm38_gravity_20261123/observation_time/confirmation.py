import argparse
import json
import numpy as np
from simulate import HERE,SNAP,CONFIRM_SEEDS
from analyse import load_results,intervals


def lock():
    counts=sum(load_results(f"strong_{s}")[0]["prior"].sum((0,2)) for s in (58411,58412))
    choice={"idx":int(counts.argmax()),"counts":counts.tolist()}
    (HERE/"strong_selection.json").write_text(json.dumps(choice,indent=2),encoding="utf8")


def confirm():
    selected=json.loads((HERE/"strong_selection.json").read_text())["idx"]
    source=json.loads((HERE/"selection.json").read_text())["prior"]["all"]
    old=int(np.where(SNAP==.55)[0][0])
    mats=[]
    hits=[]
    possible=[]
    for seed in CONFIRM_SEEDS:
        b,y=load_results(f"strong_{seed}")
        h=b["prior"].mean(0)
        mats.append(np.column_stack([h[selected]-h[j] for j in (old,source)]))
        hits.append(h)
        possible.extend(y)
    hi=np.concatenate(hits,axis=-1)
    r={"chosen":selected,"n":hi.shape[1],"reachable":int(np.sum(possible)),
       "rates":{str(j):float(hi[j].mean()) for j in (selected,old,source)},
       "conditional":float(hi[selected,possible].mean()),"comparisons":{}}
    for key,j,d,(lo,up) in zip(("original","source_selected"),(old,source),np.concatenate(mats).mean(0),intervals(mats)):
        r["comparisons"][key]=dict(delta=float(d),ci=[float(lo),float(up)],improvement=bool(d>=.03 and lo>0),
            faster_noninferior=bool(SNAP[j]-SNAP[selected]>=.1-1e-10 and lo>-.02))
    (HERE/"confirmation.json").write_text(json.dumps(r,indent=2),encoding="utf8")
    print(r)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("lock","confirm"))
    if p.parse_args().stage=="lock":
        lock()
    else:
        confirm()
