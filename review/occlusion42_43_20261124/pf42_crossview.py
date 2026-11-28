"""Shared-alias and independent-visibility teachers with identical camera evidence."""
import json
import warnings

import numpy as np
from scipy.special import logsumexp

from pf42_crossview_data import CAL, DEPTH, HELD, OUT, cache


TEMPS = (.05,.10,.20)
RATES = (.05,.15,.30)
METHODS = ("median","visibility","balanced","shared")
SEED = 421341


def load(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def entries(partition,names):
    paths = sorted(p for p in cache(partition).glob("*.npz")
                   if not p.name.endswith((".truth.npz", ".partial.npz")))
    for name in names:
        matches = [p for p in paths if p.stem.rsplit("-",1)[0] == name]
        if not matches:
            raise ValueError(f"No cached data for {partition}/{name}")
        yield from matches


def log_mean(l):
    return logsumexp(l,axis=-1,keepdims=True)-np.log(l.shape[-1])


def mix_evidence(log,eta):
    return np.logaddexp(np.log1p(-eta)+log,np.log(eta)+log_mean(log))


def soft_curve(c,visibility,temperature):
    score = c.astype(np.float32)
    valid = np.isfinite(score)
    raw = np.where(valid,score/temperature,-np.inf)
    n = valid.sum(axis=-1,keepdims=True)
    log_norm = logsumexp(raw,axis=-1,keepdims=True)-np.log(np.maximum(n,1))
    log_norm = np.where(n>0,log_norm,0.)
    # Observations outside the source projection and textureless patches are neutral.
    ratio = np.where(valid,np.exp(raw-log_norm),1.)
    v = np.where(valid,visibility,0.).astype(np.float32)
    return np.log(1-v+v*ratio),valid.sum(1)


def risk_decode(lp,allowed,fallback):
    active = allowed.any(1)
    lp = np.where(allowed,lp,-np.inf)
    lp[~active,0] = 0.
    p = np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
    # Under AbsRel, choose depth by its inverse-depth weighted posterior mass.
    w = p/DEPTH[None]
    w /= w.sum(1,keepdims=True)
    s = w.cumsum(1)
    index = (s<.5).sum(1).clip(0,len(DEPTH)-1)
    i = np.arange(len(index))
    lo = np.maximum(0,s[i,index]-w[i,index])
    fraction = (.5-lo)/np.maximum(w[i,index],1e-15)
    logz = np.log(DEPTH[index])+(fraction-.5)*np.log(DEPTH[1]/DEPTH[0])
    estimate = np.exp(logz).clip(DEPTH[0],DEPTH[-1])
    return np.where(active,estimate,fallback),p


def decide(data,kind,temp,rate,posterior=False):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="All-NaN slice encountered")
        median = np.nanmedian(data["peaks"],axis=1)
    if kind=="median":
        return median
    evidence, support = soft_curve(data["cost"],data["visibility"],temp)
    if kind=="visibility":
        total = mix_evidence(evidence,rate).sum(axis=1)
    else:
        groups = data["groups"]
        total = np.zeros_like(evidence[:,0])
        for g in np.unique(groups):
            cur = evidence[:,groups==g]
            group_sum = cur.sum(1)
            if kind=="shared":
                total += mix_evidence(group_sum,rate)
            else:
                total += mix_evidence(cur,rate).mean(1)
    pred,p = risk_decode(total,support>=2,median)
    pred = np.where(np.isfinite(median),pred,np.nan)
    return (pred,p) if posterior else pred


def masks(data,truth):
    guess = data["peaks"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="All-NaN slice encountered")
        q1,q3 = np.nanquantile(np.log(guess),[.25,.75],axis=1)
    disagree = (np.isfinite(guess).sum(1)>=3)&(q3-q1>=.08)
    occluded = (truth["occlusion"]==1).any(1)
    mask = np.isfinite(truth["truth"])&(truth["truth"]>=DEPTH[0])&(truth["truth"]<=DEPTH[-1])
    supported = np.isfinite(guess).any(1)
    return dict(all=mask&supported,hard=mask&supported&(occluded|disagree),
                lidar_occluded=mask&supported&occluded,disagreement=mask&supported&disagree,
                only_lidar=mask&supported&occluded&~disagree,
                only_disagreement=mask&supported&~occluded&disagree,both=mask&supported&occluded&disagree,
                no_lidar_state=mask&supported&(truth["occlusion"]<0).all(axis=1),
                valid_reference=mask)


def calibration():
    search = {k:{(t,r):{} for t in TEMPS for r in RATES} for k in METHODS if k!="median"}
    for path in entries("cal",CAL):
        data, truth = load(path),load(path.with_suffix(".truth.npz"))
        name = str(data["scene"])
        hard = masks(data,truth)["hard"]
        if not hard.any():
            continue
        for kind,options in search.items():
            for (t,r),by_scene in options.items():
                pred = decide(data,kind,t,r)
                error = float(abs(pred[hard]/truth["truth"][hard]-1).mean())
                by_scene.setdefault(name,[]).append(error)
    choices, trace = {},{}
    for kind,options in search.items():
        trace[kind] = {f"{t},{r}":{g:float(np.mean(values)) for g,values in scores.items()}
                       for (t,r),scores in options.items()}
        feasible = [key for key,value in options.items() if len(value)==len(CAL)]
        if not feasible:
            raise ValueError("Calibration has no hard pixels in a registered scene")
        choices[kind] = min(feasible,key=lambda key:np.mean([np.mean(v) for v in options[key].values()]))
    return choices,trace


def comparison(a,b):
    x,y = np.array([[field[k] for k in sorted(a)] for field in (a,b)])
    idx = np.random.default_rng(SEED).integers(len(x),size=(20000,len(x)))
    rel = 100*(1-x.mean()/y.mean())
    ci = np.quantile(100*(1-x[idx].mean(1)/y[idx].mean(1)),[.025,.975])
    return dict(relative=float(rel),ci=ci.tolist())


def run():
    choices,trace = calibration()
    rows = []
    predictions = OUT/"predictions"
    predictions.mkdir(exist_ok=True)
    for path in entries("held",HELD):
        data = load(path)
        options = dict(choices,median=(.1,.15))
        pred = {a:decide(data,a,*options[a]) for a in METHODS}
        # This ablation differs only in whether the direction group shares the wrong hypothesis.
        pred["same_params_visibility"] = decide(data,"visibility",*choices["shared"])
        np.savez_compressed(predictions/path.name,**pred)
        truth = load(path.with_suffix(".truth.npz"))
        mask = masks(data,truth)
        point_count = {k:int(v.sum()) for k,v in mask.items()}
        sizes = truth["truth"]
        scores = {part:{a:float(abs(p[ok]/sizes[ok]-1).mean()) for a,p in pred.items()}
                  for part,ok in mask.items() if ok.any() and part!="valid_reference"}
        rows.append(dict(scene=str(data["scene"]),ref=str(data["ref"]),count=point_count,scores=scores,
                         cameras=len(data["groups"]),direction_groups=len(set(data["groups"].tolist())),
                         min_baseline=float(data["offsets"].min()),max_baseline=float(data["offsets"].max()),
                         direction=float(np.degrees(np.arcsin(min(1,data["directions"].item()))))))
    reduced = {}
    point_totals = {}
    for k in rows[0]["count"]:
        point_totals[k] = sum(r["count"][k] for r in rows)
        if k=="valid_reference":
            continue
        reduced[k] = {}
        for scene in HELD:
            vals = [r["scores"][k] for r in rows if r["scene"]==scene and k in r["scores"]]
            if vals:
                reduced[k][scene] = {a:float(np.mean([r[a] for r in vals])) for a in vals[0]}
    means = {k:{a:float(np.mean([row[a] for row in vals.values()])) for a in next(iter(vals.values()))}
             for k,vals in reduced.items() if vals}
    pairs = {k:{b:comparison({g:r["shared"] for g,r in scene.items()},{g:r[b] for g,r in scene.items()})
                 for b in ("median","visibility","balanced","same_params_visibility")}
             for k,scene in reduced.items() if scene}
    gate = all(means["hard"]["shared"]<means["hard"][a] for a in ("median","visibility"))
    result = dict(settings=choices,calibration=trace,rows=rows,counts=point_totals,
                  means=means,scenes=reduced,comparisons=pairs,student_gate=gate)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print("scores",means,"student gate",gate,flush=True)


if __name__=="__main__":
    run()
