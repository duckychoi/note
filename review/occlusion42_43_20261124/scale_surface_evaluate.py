"""Score frozen monocular readouts; all target depth accesses are confined to scoring."""
from collections import defaultdict
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image

from depth_full_data import NyuDepth
from depth_full_eval import load_sample, sample_names
from encoder_expand_data import selection_for
from repeat_encoder import DESIGN, family
from scale_surface_rgb import ARMS, OUT, SRC, Reader, infer_all, nearest, read_heads, read_syn, resize_rgb, sha, write


def frames(dataset):
    if dataset=="hypersim_val":
        for scene,available in sorted(selection_for("validation_24").items()):
            for idx in np.linspace(0,len(available)-1,min(4,len(available)),dtype=int):
                cam,f=available[idx]
                rgb,z,_=read_syn(scene,cam,f)
                yield f"{scene}/{cam}/{f}",scene,rgb,z,np.isfinite(z)&(z>=.25)&(z<=30)
    elif dataset=="ibims":
        with ZipFile(OUT.parent/"iBims-1.zip") as archive:
            for name in sample_names(archive,False):
                rgb,z,valid,_=load_sample(archive,name,1)
                key=name.split("/")[-1]
                yield key,key.rsplit("_",1)[0],rgb,z,valid
    elif dataset=="middlebury":
        for src in sorted((OUT.parent/"repeat-capture/physical").iterdir()):
            group=family(src.name)
            if group in set(map(family,DESIGN)):
                continue
            rgb=np.asarray(Image.open(src/"Illum1/Exp1/view1.png").convert("RGB"))
            disp=np.asarray(Image.open(src/"disp1.png"),float)
            z=3740*.16/(disp+float((src/"dmin.txt").read_text()))
            yield src.name,group,rgb,z,(disp>0)&(z>=.25)&(z<=30)
    elif dataset=="nyu":
        ds=NyuDepth(OUT.parent/"nyu-depthv2-wds","test",1)
        for i in range(len(ds)):
            (rgb,z,valid,_),_,_=ds[i]
            yield str(i),str(i),rgb,z,valid
    else:
        raise ValueError(dataset)


def metrics(depths,truth,mask,labels):
    shape=depths["base"].shape
    z=nearest(truth.astype(np.float32),shape).astype(float)
    m=nearest(mask.astype(np.uint8),shape).astype(bool)
    m &= np.isfinite(z)&(z>=.25)&(z<=30)
    if not m.any():
        return None
    z=z[m]
    _,region,counts=np.unique(labels[m],return_inverse=True,return_counts=True)
    base=depths["base"][m]
    t=np.log(z/base)
    bias=t.mean()
    shape_res=t-bias
    losses={}
    for name,field in depths.items():
        pred=field[m]
        if not np.isfinite(pred).all() or pred.min()<=0 or np.abs(pred-base).max()>.250001:
            raise ValueError("Invalid prediction or unbounded correction")
        c=np.log(pred/base)
        delta=c.mean()
        h=c-delta
        scale_gain=2*bias*delta-delta**2
        shape_gain=np.mean(2*shape_res*h-h*h)
        residual=t-c
        log_mse=np.mean(residual**2)
        region_means=np.bincount(region,weights=residual)/counts
        between=np.average((region_means-residual.mean())**2,weights=counts)
        within=np.mean((residual-region_means[region])**2)
        if not np.isclose(np.mean(t*t)-log_mse,scale_gain+shape_gain,rtol=1e-8,atol=2e-13):
            raise ValueError("Orthogonal decomposition failed on scoring mask")
        if not np.isclose(log_mse,residual.mean()**2+between+within,atol=2e-13):
            raise ValueError("Region decomposition failed")
        error=(z-base)/z
        move=np.where(error>=0,1.,-1.)*(pred-base)/z
        overshoot=2*np.maximum(move-np.abs(error),0)
        if not np.isclose(np.mean(move-overshoot),np.mean(np.abs(error)-np.abs(pred/z-1)),atol=1e-12):
            raise ValueError("Signed AbsRel change differs from loss difference")
        losses[name]=dict(absrel=float(np.mean(np.abs(pred/z-1))),log_mse=float(log_mse),
            scale_mse=float((bias-delta)**2),surface_mse=float(np.mean((shape_res-h)**2)),
            between_regions_mse=float(between),within_regions_mse=float(within),
            oracle_log_scale=float(bias),applied_log_scale=float(delta),
            same_scale_direction=float(bias*delta>0),
            signed_move=float(np.mean(move)),overshoot_cost=float(np.mean(overshoot)),
            scale_gain=float(scale_gain),surface_gain=float(shape_gain))
    return dict(pixels=len(z),scores=losses)


def betting_lower(gains,lower=-1.):
    """Lower confidence bound in native units; only a known lower support bound is needed."""
    if gains.min()<lower-1e-10:
        raise ValueError("Observed gain below known support")
    bets=np.array([.1,.25,.5,.75,1.])[:,None]
    def log_e(m):
        factors=1+bets*(gains-m)/(m-lower)
        with np.errstate(divide="ignore"):
            logs=np.sum(np.log(np.maximum(factors,0.)),axis=1)
        return np.logaddexp.reduce(logs)-np.log(len(logs))
    lo=np.nextafter(lower,0.)
    # At or above the sample mean each product is <= 1 by AM-GM.
    hi=max(float(gains.mean()),lo)
    if log_e(lo)<np.log(20):
        return lower
    for _ in range(70):
        mid=(lo+hi)/2
        if log_e(mid)>=np.log(20):
            lo=mid
        else:
            hi=mid
    return float(lo)


def summarize(rows,dataset):
    groups=defaultdict(list)
    for r in rows.values():
        groups[r["group"]].append(r["scores"])
    names=list(groups)
    means={}
    for arm in ARMS:
        means[arm]={k:np.array([np.mean([r[arm][k] for r in groups[g]]) for g in names])
                    for k in next(iter(rows.values()))["scores"][arm]}
    report={}
    baseline=means["base"]["absrel"]
    for arm in ARMS:
        report[arm]={k:float(x.mean()) for k,x in means[arm].items()}
        diff=baseline-means[arm]["absrel"]
        report[arm].update(relative_percent=float(100*diff.mean()/baseline.mean()),
                           absolute_gain=float(diff.mean()),improved_groups=int(np.sum(diff>0)))
        if dataset!="nyu" and arm=="stereo_joint":
            idx=np.random.default_rng(424343).integers(0,len(groups),size=(6000,len(groups)))
            # Descriptive interval; one paired group sample is reused for both arms.
            report[arm]["relative_bootstrap95_percent"]=np.quantile(
                100*diff[idx].mean(axis=1)/baseline[idx].mean(axis=1),[.025,.975]).tolist()
            report[arm]["lower_absrel95"]=betting_lower(diff)
            # Official predicted FoV >= 1 degree on the shorter side and q >= 1e-5.
            max_base=.5/np.tan(np.deg2rad(.5))/1e-5
            lower=-1-.03*max_base/.25
            report[arm]["lower_3pct_margin95"]=betting_lower(diff-.03*baseline,lower)
    dense=means["dense_joint"]["absrel"]
    incremental=dense-means["stereo_joint"]["absrel"]
    contrast=dict(absolute_gain=float(incremental.mean()),
                  relative_percent=float(100*incremental.mean()/dense.mean()),
                  improved_groups=int((incremental>0).sum()))
    return dict(frames=len(rows),groups=len(groups),group_ids=names,stereo_vs_dense=contrast,
                valid_pixels=sum(r["pixels"] for r in rows.values()),arms=report)


def main():
    model=Reader()
    heads=read_heads()
    frozen=sha(OUT/"weights.json")
    summary={}
    for dataset in ("hypersim_val","ibims","middlebury","nyu"):
        rows={}
        excluded=[]
        for key,group,rgb,z,mask in frames(dataset):
            img=resize_rgb(rgb)
            rep=model.extract(img)
            depths=infer_all(rep,heads)
            scores=metrics(depths,z,mask,rep[1])
            if scores is None:
                excluded.append(key)
                continue
            rows[key]=dict(group=group,regions=len(rep[-1]),**scores)
        summary[dataset]=summarize(rows,dataset)
        summary[dataset]["no_valid_target"]=excluded
        write(OUT/(dataset+"-rows.json"),dict(heads_sha256=frozen,rows=rows))
        write(OUT/"metrics.json",dict(heads_sha256=frozen,native_check_max=model.official_check,datasets=summary))
        print(dataset,len(rows),summary[dataset]["arms"]["stereo_joint"]["relative_percent"],flush=True)
    assert frozen==sha(OUT/"weights.json")


if __name__=="__main__":
    main()
