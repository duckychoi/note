"""Separate scene sensitivity, model adequacy, and selection on fixed teacher outputs."""
import json
import sys

import numpy as np

from pf42_multibase import K, OUT, choice, corrections, infer, load, mean, observations, subjects, subset, valid
from pf42_multibase_report import contrast


THRESHOLDS = (.005, .010, .020, .035)


def views(d):
    ok = np.isfinite(d)
    r = np.maximum(d/K[None, :],1/30)
    r = np.log(r)
    hi = np.max(np.where(ok,r,-np.inf),axis=1)
    lo = np.min(np.where(ok,r,np.inf),axis=1)
    return ok.sum(1), np.where(ok.sum(1)>=2,hi-lo,0.)


def actions(d):
    count, width = views(d)
    return {"중앙값":np.zeros(len(d),dtype=bool),
            "항상 교사":count >= 2,
            **{f"{t:.3f} 이상":(count>=2)&(width>=t) for t in THRESHOLDS},
            **{f"{t:.3f} 미만":(count>=2)&(width<t) for t in THRESHOLDS}}


def tune_gate():
    cal = load("cal")
    rows = {}
    for g in np.unique(cal["group"]):
        keep = cal["group"] == g
        fit,score = subset(cal,~keep),subset(cal,keep)
        selected,_ = choice(fit)
        d,fallback,baseline = observations(score,corrections(fit))
        teacher = infer(d,fallback,*selected["joint"])
        rows[str(g)] = {k:mean(subjects(score,np.where(a,teacher,baseline))) for k,a in actions(d).items()}
    best = min(next(iter(rows.values())),key=lambda key:np.mean([v[key] for v in rows.values()]))
    return best,rows


def run():
    with np.load(OUT/"held_predictions.npz",allow_pickle=False) as f:
        full = {key:f[key] for key in f.files}
    d = subset(full,valid(full))
    z = d["truth"]
    med, teacher = (abs(d[n]/z-1) for n in ("continuous_all","joint"))
    oracle = np.minimum(med,teacher)
    family = d["group"]
    names = np.unique(family)
    group_scores = {str(g):dict(median=float(med[family==g].mean()),
                                teacher=float(teacher[family==g].mean()),
                                oracle=float(oracle[family==g].mean()),
                                points=int((family==g).sum())) for g in names}
    loo = {str(g):100*(1-np.mean([row["teacher"] for k,row in group_scores.items() if k!=g])/
                           np.mean([row["median"] for k,row in group_scores.items() if k!=g])) for g in names}
    group_median = {k:v["median"] for k,v in group_scores.items()}
    group_oracle = {k:v["oracle"] for k,v in group_scores.items()}
    delta,ci = contrast(group_oracle,group_median)

    count, width = views(d["disparity"])
    e = d["disparity"]-K[None,:]/z[:,None]
    # An optimal interval of fixed width can start on an observed residual.
    difference = e[:,None,:]-e[:,:,None]
    best = ((difference>=0)&(difference<=4.8)).sum(axis=2).max(axis=1)
    considered = count >= 4
    more_than_one = considered&(count-best>=2)
    miss = {str(g):int((more_than_one&(family==g)).sum()) for g in names}
    bins = [(0.,.005),(.005,.02),(.02,np.inf)]
    table = []
    for lower,upper in bins:
        use = (count>=2)&(width>=lower)&(width<upper)
        scores = [{"group":str(g),"n":int((use&(family==g)).sum()),
                   "same_input_median":float(med[use&(family==g)].mean()),
                   "teacher":float(teacher[use&(family==g)].mean())}
                  for g in names if (use&(family==g)).any()]
        table.append(dict(range=[lower,upper if np.isfinite(upper) else None],points=int(use.sum()),
                          median=float(np.mean([g["same_input_median"] for g in scores])),
                          teacher=float(np.mean([g["teacher"] for g in scores])),
                          group_scores=scores))

    rule, folds = tune_gate()
    choice_mask = actions(d["disparity"])[rule]
    gate = subjects(d,np.where(choice_mask,d["joint"],d["continuous_all"]))
    gain,interval = contrast(gate,group_median)
    output = dict(groups=group_scores,leave_one_group_out=loo,oracle_both=dict(relative=delta,ci=ci.tolist()),
                  multiple_residual_violations=dict(points=int(considered.sum()),violations=int(more_than_one.sum()),
                                                    counts_by_family=miss),span_table=table,
                  gate=dict(rule=rule,points=int(choice_mask.sum()),mean=mean(gate),relative=gain,
                            interval=interval.tolist(),calibration=folds))
    (OUT/"next-diagnostic.json").write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    print(json.dumps(output,indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    run()
