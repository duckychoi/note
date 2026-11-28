"""Fit swap-invariant AbsRel gains only from render training scenes, then score synthetic and physical pairs."""
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from light_encoder_score import middlebury_pairs
from light_encoder_study import CKPT, Model, sha
from material_light_assets import OUT, save_json
from material_light_depth import depth


NAMES = ("quiet","highlight","shadow","both")
MODES = ("half","full_candidate","common","conditional","best_endpoint")


def image_features(rgb):
    x = cv2.blur(rgb.astype(np.float32)/255,(5,5))
    y = x@np.array([.2126,.7152,.0722],np.float32)
    peak = x.max(2)
    sat = (peak-x.min(2))/(peak+.015)
    chrom = x/(.03+x.sum(2,keepdims=True))
    return y,peak,sat,chrom


def candidate(a,b):
    ya,va,sa,ca = a
    yb,vb,sb,cb = b
    bbright = yb>ya
    hi,lo = np.maximum(ya,yb),np.minimum(ya,yb)
    delta = hi-lo
    vh = np.where(bbright,vb,va)
    sh = np.where(bbright,sb,sa)
    sl = np.where(bbright,sa,sb)
    change = delta>.035
    high = change & (vh>.46) & (((vh>.78)&(sh<.20)) | ((sl-sh)>.085))
    shadow = change & (lo<.62) & ((hi+.02)>1.35*(lo+.02)) & (np.max(abs(ca-cb),axis=2)<.08)
    bins = high.astype(np.int8)+2*shadow.astype(np.int8)
    # Highlight selects the dimmer view. Shadow and mixed select the brighter view.
    choose_b = np.where(bins==1,~bbright,bbright)
    side = np.where(bins==0,0,np.where(choose_b,1,-1)).astype(np.int8)
    return bins,side


def scenes(split):
    return [r for r in json.loads((OUT/"scenes.json").read_text()) if r["split"]==split]


def render_data(row):
    folder = OUT/"renders"/row["scene"]
    with np.load(folder/"passes.npz") as f:
        p = {k:f[k] for k in f.files}
    valid = p["valid"] & (p["z"]>=.25)&(p["z"]<=30.)
    with np.load(folder/"predictions.npz") as f:
        z = {k:f[k] for k in f.files}
    features = {k:image_features(np.asarray(Image.open(folder/(k+".png")))) for k in z}
    return p,valid,z,features


def pairs():
    for material in ("matte","coat"):
        for i,j in itertools.combinations(range(3),2):
            yield material,i,j,f"{material}-{i}",f"{material}-{j}"


def weighted_median(s,w):
    if not len(s):
        return 0.
    idx = np.argsort(s)
    cum = np.cumsum(w[idx],dtype=np.float64)
    return float(np.clip(s[idx[np.searchsorted(cum,cum[-1]/2)]],-1,1))


def fit():
    values = {k:[] for k in range(1,4)}
    weights = {k:[] for k in range(1,4)}
    rows = scenes("train")
    for row in rows:
        p,mask,z,features = render_data(row)
        r = hashlib.sha256(row["scene"].encode()).digest()
        ix = (slice(r[0]%4,None,4),slice(r[1]%4,None,4))
        gt = p["z"][ix].astype(float)
        valid = mask[ix]
        for _,_,_,a,b in pairs():
            bins,side = candidate(features[a],features[b])
            za,zb = z[a][ix].astype(float),z[b][ix].astype(float)
            m = .5*(za+zb)
            u = .5*(zb-za)*side[ix]
            for k in values:
                use = valid & (bins[ix]==k) & (u!=0)
                s = (gt[use]-m[use])/u[use]
                w = abs(u[use])/gt[use]/valid.sum()
                values[k].append(s)
                weights[k].append(w)
    v = {k:np.concatenate(group) for k,group in values.items()}
    w = {k:np.concatenate(group) for k,group in weights.items()}
    fitted = [0.]+[weighted_median(v[k],w[k]) for k in v]
    common = weighted_median(np.concatenate(list(v.values())),np.concatenate(list(w.values())))
    save_json(OUT/"gain-lock.json",dict(coefficients=fitted,common=common,bins=NAMES,
              calibration_points={NAMES[k]:len(v[k]) for k in v},fit_scenes=[r["scene"] for r in rows],
              manifest=sha(OUT/"scenes.json"),model=sha(CKPT),algorithm=sha(Path(__file__))))
    print("fitted",fitted,common,flush=True)


def raw_mean(x,mask):
    return float(x[mask].mean(dtype=np.float64))


def eval_pair(za,zb,features_a,features_b,gt,mask,lock,entity=None,events=None):
    bins,side = candidate(features_a,features_b)
    other_bins,other_side = candidate(features_b,features_a)
    if not np.array_equal(bins,other_bins) or not np.array_equal(side,-other_side):
        raise ValueError("Swapping pictures changed the chosen physical observation")
    za,zb = za.astype(float),zb.astype(float)
    m = .5*(za+zb)
    u = .5*(zb-za)*side
    g = np.asarray(lock["coefficients"])[bins]
    adjusted = m+g*u
    safe = np.where(mask,gt,1.)
    def loss(z):
        return abs(z/safe-1)
    half = loss(m)
    proposal = loss(adjusted)
    losses = dict(half=half,conditional=proposal,full_candidate=loss(m+u),common=loss(m+lock["common"]*u),
                  best_endpoint=np.minimum(loss(za),loss(zb)))
    scores = {k:raw_mean(v,mask) for k,v in losses.items()}
    region = {}
    for k,name in enumerate(NAMES):
        use = mask&(bins==k)
        if use.any():
            region[name] = detailed(m,u,g,safe,use)
    physical = {}
    if events:
        shadow_a,shadow_b,glint_a,glint_b = events
        diff_shadow = shadow_a^shadow_b
        diff_glint = glint_a^glint_b
        for tag,one,absent_b,expected in (
                ("ray_shadow",diff_shadow&~diff_glint,~shadow_b,(bins==2)|(bins==3)),
                ("specular",diff_glint&~diff_shadow,~glint_b,(bins==1)|(bins==3))):
            use = mask&one
            if use.any():
                phys_u = .5*(zb-za)*np.where(absent_b,1,-1)
                physical[tag] = detailed(m,phys_u,np.ones_like(m),safe,use)
                physical[tag]["half_step"] = raw_mean(loss(m+.5*phys_u),use)
                physical[tag]["rgb_gain_loss"] = raw_mean(proposal,use)
                physical[tag]["proxy_recall"] = raw_mean(expected,use)
                physical[tag]["proxy_fraction_valid"] = raw_mean(expected,mask)
                physical[tag]["proxy_correct_mass"] = raw_mean(expected&one,mask)
                physical[tag]["proxy_side_correct"] = raw_mean(side==np.where(absent_b,1,-1),use&expected) if (use&expected).any() else None
    # Use positive metric profiles and mask only in scoring, after fusion.
    shape = {}
    for name,pred in (("half",m),("conditional",adjusted)):
        residual = np.log(pred)-np.log(safe)
        total,n = 0.,0
        for axis in (0,1):
            first,second = [slice(None)]*2,[slice(None)]*2
            first[axis],second[axis] = slice(None,-16,4),slice(16,None,4)
            first[1-axis]=second[1-axis]=slice(None,None,4)
            a,b = tuple(first),tuple(second)
            valid = mask[a]&mask[b]
            if entity is not None:
                valid &= entity[a]==entity[b]
            else:
                valid &= abs(np.log(safe[a]/safe[b]))<np.log(1.05)
            total += ((residual[a]-residual[b])[valid]**2).sum()
            n += int(valid.sum())
        if n:
            shape[name] = float(total/n)
    return dict(scores=scores,regions=region,physical=physical,shape=shape,n=int(mask.sum()))


def detailed(m,u,g,gt,use):
    mid,mv,target = m[use],u[use],gt[use]
    lam = g[use]
    loss0 = abs(mid-target)/target
    step = lam*mv
    direction = np.sign(target-mid)
    move = direction*step/target
    overshoot = 2*np.maximum(move-loss0,0)
    tie_cost = np.where(direction==0,abs(step)/target,0.)
    gain = (loss0-abs(mid+step-target)/target).mean()
    if not np.isclose(gain,(move-overshoot-tie_cost).mean(),atol=1e-12):
        raise ValueError("Finite-step metric reduction disagrees with the identity")
    left,right = abs(mid+mv-target),abs(mid-mv-target)
    return dict(n=len(mv),mass=float(use.mean()),half=float(loss0.mean()),conditional=float((abs(mid+step-target)/target).mean()),
                full_candidate=float((left/target).mean()),opposite=float((right/target).mean()),
                candidate_wins=float(((left<right)+.5*(left==right)).mean()),gain=float(gain),
                directional=float(move.mean()),overshoot=float(overshoot.mean()),tie_cost=float(tie_cost.mean()))


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    def avg(records,field):
        return {k:float(np.mean([r[field][k] for r in records])) for k in records[0][field]}
    grp = {key:avg(rs,"scores") for key,rs in groups.items()}
    base = np.array([r["half"] for r in grp.values()])
    cand = np.array([r["conditional"] for r in grp.values()])
    ids = np.random.default_rng(432611).integers(0,len(grp),(10000,len(grp)))
    ci = np.quantile(100*(1-cand[ids].mean(1)/base[ids].mean(1)),[.025,.975])
    means = {k:float(np.mean([r[k] for r in grp.values()])) for k in grp[next(iter(grp))]}
    shapes = {k:float(np.mean([np.mean([r["shape"][k] for r in rs]) for rs in groups.values()]))
              for k in ("half","conditional")}
    # Conditional subsets use the same population measure as the primary score,
    # conditioning after weighting each group and image equally.
    def subset(field):
        subset_names = {k for r in rows for k in r[field]}
        result = {}
        for name in sorted(subset_names):
            samples = [(r[field][name],r[field][name]["n"]/r["n"]/len(rs)/len(groups))
                       for rs in groups.values() for r in rs if name in r[field]]
            den = sum(w for rec,w in samples)
            metrics = {}
            for k in samples[0][0]:
                if k in ("n","mass","proxy_side_correct","proxy_fraction_valid","proxy_correct_mass"):
                    continue
                metrics[k] = sum(w*rec[k] for rec,w in samples)/den
            if field=="physical":
                weight_detected = sum(w*rec["proxy_recall"] for rec,w in samples)
                if weight_detected:
                    metrics["proxy_side_correct_on_detected"] = sum(
                        w*rec["proxy_recall"]*(rec["proxy_side_correct"] or 0.) for rec,w in samples)/weight_detected
            result[name] = dict(fraction=den,**metrics)
        return result
    return dict(groups=grp,means=means,shape=shapes,gain=100*(1-float(cand.mean()/base.mean())),
                ci=ci.tolist(),wins=int((cand<base).sum()),regions=subset("regions"),physical=subset("physical"))


def run():
    cv2.setNumThreads(1)
    fit()
    lock = json.loads((OUT/"gain-lock.json").read_text())
    val,interventions = [],[]
    for row in scenes("val"):
        p,mask,z,feat = render_data(row)
        for mat,i,j,a,b in pairs():
            blank = np.zeros_like(mask)
            events = (p[f"shadow{i}"],p[f"shadow{j}"],p[f"glint{i}"] if mat=="coat" else blank,
                      p[f"glint{j}"] if mat=="coat" else blank)
            rec = eval_pair(z[a],z[b],feat[a],feat[b],p["z"],mask,lock,p["entity"],events)
            val.append(dict(scene=row["scene"],group=row["family"],material=mat,lights=[i,j],**rec))
        single = {k:float(abs(v[mask].astype(float)/p["z"][mask]-1).mean()) for k,v in z.items()}
        changes = [single[f"coat-{i}"]-single[f"matte-{i}"] for i in range(3)]
        interventions.append(dict(scene=row["scene"],single=single,coat_effect=changes,
                interaction_range=float(max(changes)-min(changes))))
    save_json(OUT/"validation.json",dict(**aggregate(val),rows=val,factorial=interventions,
             by_material={m:aggregate([r for r in val if r["material"]==m]) for m in ("matte","coat")}))
    model = Model()
    real = []
    for scene,group,a,b,z,mask in middlebury_pairs():
        da,db = depth(model,a),depth(model,b)
        rec = eval_pair(da,db,image_features(a),image_features(b),z.cpu().numpy(),mask.cpu().numpy(),lock)
        real.append(dict(scene=scene,group=group,**rec))
    save_json(OUT/"middlebury.json",dict(**aggregate(real),rows=real,gain_lock=sha(OUT/"gain-lock.json")))
    print("real",aggregate(real)["means"],flush=True)


if __name__=="__main__":
    run()
