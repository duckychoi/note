"""Transfer a teacher scale only when an RGB-anchored interval certifies the direction."""
import argparse
import copy
import json
import math
import time

import cv2
import numpy as np
import torch

import pf42_visibility_encoder as study
from pf42_crossview import entries, load
from pf42_crossview_data import CAL, DEPTH, FIT, HELD
from pf42_monocular import DATASETS, absrel, fold_keys, pixel_sample, summarize
import pf42_monocular as previous
from pf42_visibility import NEW, PARTITION, consensus_map, predictions, scene_context


RUN = study.OUT / "scale-confidence"
ARMS = ("direct", "always_scale", "selected_scale")
CALIBRATION = (("cal", CAL), ("held", HELD))


def write(path, data):
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf8")
    temp.replace(path)


def weighted_median(x, weight):
    order = np.argsort(x)
    acc = weight[order].cumsum()
    return float(x[order[np.searchsorted(acc, .5*acc[-1])]])


def optimum(base, target):
    return weighted_median(target/base,base/target)


def use_scale(kd,kt,lo,hi):
    if kd<lo and kt>kd:
        return min(kt,lo)
    if kd>hi and kt<kd:
        return max(kt,hi)
    return kd


def supported(obs):
    return (np.isfinite(obs["peaks"]).sum(1)>=3) & (np.isfinite(obs["cost"]).sum(1)>=2).any(1)


@torch.no_grad()
def rgb_depth(model, rgb, xy):
    x,y=xy.T
    return model.depth(model.encode(rgb), None)[y,x].exp().cpu().numpy().astype(np.float64)


def proposal(base,direct,probability,interval):
    kd=optimum(base,direct)
    ratio=DEPTH[None]/base[:,None]
    kt=weighted_median(ratio.ravel(),(probability/ratio).ravel())
    k=use_scale(kd,kt,*interval)
    return dict(direct=kd,teacher=kt,selected=k,accepted=k!=kd,
                always_factor=kt/kd,selected_factor=k/kd)


def scene_scales(model):
    values={}
    for part,names in CALIBRATION:
        for path in entries(part,names):
            obs,lab=load(path),load(path.with_suffix(".truth.npz"))["truth"]
            keep=supported(obs)&np.isfinite(lab)&(lab>=DEPTH[0])&(lab<=DEPTH[-1])
            if not keep.any():
                continue
            rgb,xy=study.rescale(obs["rgb"],obs["tally"][keep])
            factor=optimum(rgb_depth(model,rgb,xy),lab[keep])
            values.setdefault(str(obs["scene"]),[]).append(dict(image=path.stem,scale=factor,points=int(keep.sum())))
    if set(values)!={s for _,names in CALIBRATION for s in names}:
        raise ValueError("Missing a calibration scene")
    maxima={s:max(abs(math.log(r["scale"])) for r in rows) for s,rows in values.items()}
    rank=math.ceil((len(maxima)+1)*.8)
    radius=sorted(maxima.values())[rank-1]
    return (math.exp(-radius),math.exp(radius)),values,maxima


def held_scales(model, interval):
    report=[]
    for name in NEW:
        paths,cams,cameras,anchors=scene_context(name,PARTITION)
        maps={}
        for path in paths:
            obs,truth=load(path),load(path.with_suffix(".truth.npz"))["truth"]
            keep=supported(obs)
            valid=np.isfinite(truth[keep])&(truth[keep]>=DEPTH[0])&(truth[keep]<=DEPTH[-1])
            if not valid.any():
                continue
            sources=[cameras[str(s)] for s in obs["sources"]]
            for c in sources:
                if c not in maps:
                    maps[c]=consensus_map(c,cams,anchors)
            use={k:obs[k][keep] for k in ("tally","cost","peaks")}
            use["groups"]=obs["groups"]
            prob=predictions(use,cameras[str(obs["ref"])],sources,maps,posterior=True)
            rgb,xy=study.rescale(obs["rgb"],use["tally"])
            base=rgb_depth(model,rgb,xy)
            rec=proposal(base,np.nanmedian(use["peaks"],axis=1),prob,interval)
            gt=truth[keep][valid]
            best=optimum(base[valid],gt)
            errors={k:float(abs(rec[k]*base[valid]/gt-1).mean()) for k in ("direct","teacher","selected")}
            report.append(dict(scene=name,frame=path.stem,covered=interval[0]<=best<=interval[1],
                               optimum=best,scores=errors,**rec))
    return report


def oracle():
    scores={}
    for ds in DATASETS:
        path=previous.RUN/f"diagnostic-{ds}.json"
        rows=json.loads(path.read_text(encoding="utf8"))["rows"]
        for r in rows:
            v=r["scores"]
            r["scores"]={"direct":v["direct"],"oracle":min(v["direct"],v["shared_scale_direct_shape"])}
        scores[ds]=summarize(rows,[("oracle","direct")])
    return scores


def prepare(model, real):
    interval, frames, maxima=scene_scales(model)
    fit=[]
    for path,row in zip(entries("fit",FIT),real,strict=True):
        base=rgb_depth(model,row["rgb"],row["xy"])
        fit.append(dict(frame=path.stem,scene=str(row["group"]),points=len(base),
                        **proposal(base,np.exp(row["direct"]),row["probability"],interval)))
    print("RGB scale interval",interval,"fit selected",sum(r["accepted"] for r in fit),"/",len(fit),flush=True)
    # Calibration and label shifts are locked before inspecting the held scale outcomes.
    write(RUN/"calibration.json",dict(interval=interval,calibration=frames,maxima=maxima,fit=fit))
    held=held_scales(model,interval)
    write(RUN/"diagnostics.json",dict(held=held,oracle=oracle()))
    print("held scale covered",sum(r["covered"] for r in held),"selected",sum(r["accepted"] for r in held),"/",len(held),flush=True)


@torch.no_grad()
def assess(model):
    for seed in study.SEEDS:
        payload=torch.load(RUN/f"seed-{seed}.pt",weights_only=True,map_location="cpu")
        if payload["step"]!=study.WARM+study.JOINT:
            raise ValueError(f"Incomplete checkpoint for {seed}")
        for arm in ARMS:
            key=f"{seed}-{arm}"
            model.arms[key]=copy.deepcopy(model.encoder.blocks[model.start:]).requires_grad_(False)
            model.arms[key].load_state_dict(payload["weights"][arm])
    for ds in DATASETS:
        path=RUN/f"{ds}.json"
        if path.is_file():
            continue
        rows=[]
        keys=fold_keys(ds)
        for key,group,rgb,z,valid in study.frames(ds):
            if key not in keys:
                continue
            yy,xx=pixel_sample(z,valid)
            if not len(xx):
                continue
            rgb,xy=study.rescale(rgb,np.column_stack((xx,yy)))
            x,y=xy.T
            state=model.encode(rgb)
            truth=np.log(z[yy,xx].astype(np.float64))
            seeds=[{a:absrel(model.depth(state,f"{s}-{a}")[y,x].cpu().numpy(),truth) for a in ARMS} for s in study.SEEDS]
            score={a:float(np.mean([r[a] for r in seeds])) for a in ARMS}
            rows.append(dict(key=key,group=group,fold=keys[key],points=len(x),scores=score,seeds=seeds))
        reference=json.loads((previous.RUN/f"diagnostic-{ds}.json").read_text(encoding="utf8"))["rows"]
        expected={(r["key"],r["fold"],r["group"],r["points"]) for r in reference}
        if expected!={(r["key"],r["fold"],r["group"],r["points"]) for r in rows}:
            raise ValueError(f"External evaluation changed for {ds}")
        summary=summarize(rows,[("selected_scale","direct"),("always_scale","direct")])
        write(path,dict(summary=summary,rows=rows))
        print(ds,{f:(s["means"],s["relative"]) for f,s in summary.items()},flush=True)


def run(phase):
    cv2.setNumThreads(1)
    RUN.mkdir(exist_ok=True)
    real=study.real_data()
    with study.gpu_slot():
        while not study.capacity():
            time.sleep(5)
        model=study.Trainer()
        if phase=="run":
            if not (RUN/"diagnostics.json").is_file():
                prepare(model,real)
            saved=json.loads((RUN/"calibration.json").read_text(encoding="utf8"))["fit"]
            for r,file,pro in zip(real,entries("fit",FIT),saved,strict=True):
                if file.stem!=pro["frame"] or len(r["xy"])!=pro["points"]:
                    raise ValueError("Scale labels refer to different training images")
                r["arm_targets"]={"always_scale":r["direct"]+math.log(pro["always_factor"]),
                                  "selected_scale":r["direct"]+math.log(pro["selected_factor"])}
            study.SYN,pool,study.SYN_ALTERNATIVES,study.SYN_REPLACEMENTS=study.syn_rows()
            study.RUN,study.ARMS=RUN,ARMS
            model.arms={a:copy.deepcopy(model.encoder.blocks[model.start:]).requires_grad_(True) for a in ARMS}
            study.train(model,real,pool)
        assess(model)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",nargs="?",choices=("run","evaluate"),default="run")
    run(parser.parse_args().phase)
