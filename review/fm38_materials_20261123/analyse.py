"""Counterfactual action, policy and temporal-score rankings at paired test starts."""
import json
import argparse
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np

from worlds import ROOT, TASKS, labels

HERE=Path(__file__).resolve().parent
NAMES=("flow","mixture","retrieval")
DOMAINS=("source","shift")
SAFE=np.array([.1,.3,.5,.7])
PAIRS=list(combinations(range(4),2))


def policy(p,y):
    # Shape is seed, state, action. State utilities are paired across both domains.
    choice=p.argmax(-1)
    chosen=np.take_along_axis(np.broadcast_to(y,p.shape),choice[...,None],axis=-1)[...,0]
    enter=p.max(-1)[...,None]>SAFE
    u=np.where(enter,chosen[...,None],SAFE)
    return u,enter,choice


def order(scores,y):
    margins=np.stack([scores[:,:,i]-scores[:,:,j] for i,j in PAIRS],-1)
    truth=np.stack([y[:,i]-y[:,j] for i,j in PAIRS],-1)
    eligible=truth!=0
    sign=margins*truth
    return ((sign>0)+.5*(sign==0))[:,eligible].mean()


def interval(x,n):
    z=NormalDist().inv_cdf(.975)
    p=x/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-delta,center+delta]


def bootstrap(diffs,comparisons=24):
    # Rows are independent start states, columns are paired differences.
    rng=np.random.default_rng(3829)
    n,width=diffs.shape
    values=np.empty((64000,width))
    for a in range(0,len(values),500):
        counts=rng.multinomial(n,np.full(n,1/n),size=500)
        values[a:a+500]=counts@diffs/n
    alpha=.05/comparisons
    return np.quantile(values,[alpha/2,1-alpha/2],axis=0).T


def evaluate(task,from_archive=False):
    def truth(split):
        with np.load(ROOT/f"{task}_{split}.npz") as data:
            return labels(task,data)[:,-3:].all(1).astype(int).reshape(-1,4)
    archive=HERE/f"{task}_outcomes.npz"
    if from_archive:
        with np.load(archive) as d:
            y={s:d[s] for s in DOMAINS}
            v=d["validation_action_success_probability"]
    else:
        y={s:truth(s) for s in DOMAINS}
        v=truth("val").mean(0)
    fixed=int(v.argmax())
    f=np.load(HERE/f"{task}_forecasts.npz")
    out={"models":{},"physical":{},"controls":{},"pairs":{}}
    tracks={}
    for name in (*NAMES,"raw"):
        keys=sorted(k for k in f if k.startswith(name+"_"))
        assert len(keys)==(3 if name in ("flow","mixture") else 1)
        probs=np.stack([f[k] for k in keys])
        assert np.isfinite(probs).all() and probs.min()>=0 and probs.max()<=1
        stats={}
        for domain in DOMAINS:
            stats[domain]={}
            for k,score in enumerate(("joint","independent")):
                p=probs[:,:,:,k]
                u,enter,choice=policy(p,y[domain])
                tracks[name,score,domain]=u.mean((0,2))
                stats[domain][score]={"value":float(u.mean()),"each_seed":u.mean((1,2)).tolist(),
                    "each_safe":u.mean((0,1)).tolist(),"attempt_rate_per_safe":enter.mean((0,1)).tolist(),
                    "forced_action_value":float(np.take_along_axis(np.broadcast_to(y[domain],p.shape),
                                                                    choice[...,None],axis=-1).mean()),
                    "action_frequency_before_abstention":np.bincount(choice.ravel(),minlength=4).tolist(),
                    "pair_accuracy":float(order(p,y[domain])),"brier":float(((p-y[domain])**2).mean()),
                    "conditional_action_scores":p.mean((0,1)).tolist()}
        out["models"][name]=stats
    for domain in DOMAINS:
        outcomes=y[domain]
        out["controls"][domain]={"oracle":float(np.maximum(outcomes.max(1)[:,None],SAFE).mean()),"always_safe":.4,
                "fixed_action":fixed,"fixed_source_val_prob":float(v.max()),
                "fixed_policy_value":float(np.where(v.max()>SAFE,outcomes[:,fixed,None],SAFE).mean()),
                "action_success_rate":outcomes.mean(0).tolist()}
    margins=[np.stack([y[domain][:,i]-y[domain][:,j] for i,j in PAIRS],1) for domain in DOMAINS]
    swapped=margins[0]*margins[1]<0
    n=len(y["source"])
    state_id,pair_id=np.argwhere(swapped)[0]
    i,j=PAIRS[pair_id]
    out["physical"]={"n":n,"at_least_one_strict_reversal":int(swapped.any(1).sum()),
        "reversal_state_ci95":interval(swapped.any(1).sum(),n),"strict_pairs":int(swapped.sum()),
        "reversals_by_pair":{f"{i+1}/{j+1}":int(swapped[:,k].sum()) for k,(i,j) in enumerate(PAIRS)},
        "example":{"state":int(state_id),"actions":[i+1,j+1],
                   "source":y["source"][state_id].tolist(),"shift":y["shift"][state_id].tolist()}}
    differences={}
    for name in NAMES:
        differences[name+"_joint_minus_product"]=[tracks[name,"joint",s]-tracks[name,"independent",s] for s in DOMAINS]
    for a,b in combinations(NAMES,2):
        differences[a+"_minus_"+b]=[tracks[a,"joint",s]-tracks[b,"joint",s] for s in DOMAINS]
    cols=np.column_stack([x for pair in differences.values() for x in pair])
    intervals=bootstrap(cols).reshape(-1,2,2)
    for (k,diffs),(s0,s1) in zip(differences.items(),intervals):
        out["pairs"][k]={s:{"difference":float(d.mean()),"simultaneous_ci":ci.tolist()}
                         for s,d,ci in zip(DOMAINS,diffs,(s0,s1))}
        out["pairs"][k]["confirmed_reversal"]=bool((s0[0]>0 and s1[1]<0) or (s1[0]>0 and s0[1]<0))
    # Preserve all physically labeled test branches beside the score arrays for independent checks.
    if not from_archive:
        np.savez_compressed(archive,**y,validation_action_success_probability=v)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--from-archive",action="store_true")
    options=parser.parse_args()
    result={task:evaluate(task,options.from_archive) for task in TASKS}
    (HERE/"results.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    for task,out in result.items():
        print(task,"physical",out["physical"])
        for name in (*NAMES,"raw"):
            print(name,[[round(out["models"][name][s][k]["value"],4) for s in DOMAINS] for k in ("joint","independent")])
        for name,val in out["pairs"].items():
            print(name,*(np.round([val[s]["difference"],*val[s]["simultaneous_ci"]],4) for s in DOMAINS),
                  val["confirmed_reversal"])
