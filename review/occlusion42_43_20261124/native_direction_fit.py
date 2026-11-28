"""Fit a native metric scale, calibrate disparity intervals, then train frozen RGB students."""
from collections import defaultdict
import json
import math

import numpy as np
import torch
from torch import nn

from native_direction_data import OUT,SEED,cache_path,sha,write

ARMS=("dense","stereo","joint")
LIMIT=.35

def load(folder,row):
    with np.load(cache_path(folder,row)) as src:
        return {k:src[k] for k in src.files}

def valid(z):
    return np.isfinite(z)&(z>=.25)&(z<=30)

def weighted_median(v,w):
    order=np.argsort(v)
    weights=w[order].cumsum(dtype=np.float64)
    return float(v[order[np.searchsorted(weights,.5*weights[-1])]])

def scale_and_radius(folder,rows):
    fit=defaultdict(list)
    calibration=defaultdict(list)
    for row in rows:
        split,scene=row[:2]
        if split=="val":
            continue
        data=load(folder,row)
        z,b=data["z"],data["base"]
        ok=valid(z)
        if split=="fit" and ok.any():
            fit[scene].append((b[ok]/z[ok]).astype(np.float64))
        if split=="cal":
            a,one,two=(data[n] for n in ("A","d1","d2"))
            ok &= np.isfinite(one)&np.isfinite(two)
            if ok.any():
                gt=a[ok]/z[ok]
                lo,hi=np.minimum(one,two)[ok],np.maximum(one,two)[ok]
                calibration[scene].append(np.maximum(0,np.maximum(lo-gt,gt-hi)))
    ratios,weights=[],[]
    for images in fit.values():
        for f in images:
            ratios.append(1/f)
            weights.append(f/(len(images)*len(f)))
    k=weighted_median(np.concatenate(ratios),np.concatenate(weights))
    qs=sorted(float(np.quantile(np.concatenate(rs),.9)) for rs in calibration.values())
    index=min(len(qs),math.ceil(.9*(len(qs)+1)))
    radius=qs[index-1]
    return dict(k=k,radius_px=radius,fit_scenes=len(fit),fit_frames=sum(map(len,fit.values())),
                cal_scenes=len(qs),cal_frames=sum(map(len,calibration.values())),order=index,
                native_calibration_shape=list(data["shape"].astype(int).tolist()))

def bounds(data,k,radius):
    a,one,two=(data[n] for n in ("A","d1","d2"))
    d0=a/(k*data["base"])
    dl,du=np.minimum(one,two)-radius,np.maximum(one,two)+radius
    ok=(dl>0)&np.isfinite(dl)&np.isfinite(du)&(d0>0)
    low=np.log(np.maximum(d0,1e-12)/np.maximum(du,1e-12))
    high=np.log(np.maximum(d0,1e-12)/np.maximum(dl,1e-12))
    signal=np.where(low>0,low,np.minimum(high,0))
    return ok,low,high,signal

def region_target(lab,mask,values,n):
    amount=np.bincount(lab[mask],minlength=n)
    y=np.bincount(lab[mask],weights=np.clip(values[mask],-LIMIT,LIMIT),minlength=n)/np.maximum(1,amount)
    return y,amount

def train_data(folder,rows,k,radius):
    features,targets,dense_weights,observed_weights=[],[],[],[]
    stats=defaultdict(int)
    eligible=defaultdict(int)
    samples=[]
    for row in rows:
        if row[0]!="fit":
            continue
        data=load(folder,row)
        z,base,lab,x=(data[n] for n in ("z","base","lab","X"))
        mask=valid(z)
        if not mask.any():
            continue
        truth=np.log(np.where(mask,z,1)/(k*base))
        ok,low,high,signal=bounds(data,k,radius)
        choices=ok & (signal!=0)
        scored=choices&mask
        stats["grid"]+=lab.size
        stats["depth_interval"]+=int(ok.sum())
        stats["direction_eligible"]+=int(choices.sum())
        stats["direction_with_gt"]+=int(scored.sum())
        stats["right"]+=int((signal[scored]*truth[scored]>0).sum())
        stats["interval_gt"]+=int((ok&mask).sum())
        stats["contained"]+=int((ok&mask&(truth>=low)&(truth<=high)).sum())
        yd,wd=region_target(lab,mask,truth,len(x))
        ys,ws=region_target(lab,choices,signal,len(x))
        samples.append((row[1],x,yd,ys,wd/float(wd.sum()),ws/lab.size))
        eligible[row[1]]+=1
    for scene,x,yd,ys,wd,ws in samples:
        features.append(x);targets.append(np.stack([yd,ys],axis=1))
        dense_weights.append(wd/eligible[scene]);observed_weights.append(ws/eligible[scene])
    return tuple(np.concatenate(v).astype(np.float32) for v in (features,targets,dense_weights,observed_weights)),dict(stats)

def network(n):
    return nn.Sequential(nn.Linear(n,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

def train_heads(arrays):
    X,Y,D,S=(torch.from_numpy(v) for v in arrays)
    mu=(X*D[:,None]).sum(0)/D.sum()
    sd=(((X-mu)**2*D[:,None]).sum(0)/D.sum()).sqrt().clamp_min(.001)
    X=((X-mu)/sd).clamp(-8,8)
    if S.sum()==0:
        raise ValueError("No directional supervisor")
    D=D/D.mean();S=S/S.mean()
    states={}
    for arm in ARMS:
        torch.manual_seed(SEED)
        model=network(X.shape[1])
        optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        generator=torch.Generator().manual_seed(SEED)
        for epoch in range(60):
            for ids in torch.randperm(len(X),generator=generator).split(1024):
                pred=LIMIT*torch.tanh(model(X[ids]).flatten())
                loss_d=D[ids]*(pred-Y[ids,0])**2
                loss_s=S[ids]*(pred-Y[ids,1])**2
                loss={"dense":loss_d,"stereo":loss_s,"joint":.5*(loss_d+loss_s)}[arm].mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
        states[arm]=model.state_dict()
    return dict(mu=mu,sd=sd,states=states)

def unpack(heads):
    models={}
    for name,state in heads["states"].items():
        model=network(len(heads["mu"]))
        model.load_state_dict(state)
        models[name]=model.eval()
    return models

@torch.inference_mode()
def direction(x,area,heads,models):
    x=((torch.from_numpy(x)-heads["mu"])/heads["sd"]).clamp(-8,8)
    result={}
    for name,model in models.items():
        h=(LIMIT*torch.tanh(model(x).flatten())).numpy().astype(float)
        pos,neg=np.maximum(h,0),np.maximum(-h,0)
        p,n=area@pos,area@neg
        if min(p,n)==0:
            h=np.zeros_like(h)
        else:
            h=pos*min(1,n/p)-neg*min(1,p/n)
        if abs(area@h)>1e-9*area.sum():
            raise ValueError("Surface output changed geometric mean scale")
        result[name]=h
    return result

def main():
    torch.set_num_threads(2)
    inv=json.loads((OUT/"inventory.json").read_text())
    folder=OUT/inv["folder"]
    rows=inv["rows"]
    cal=scale_and_radius(folder,rows)
    write(OUT/"fixed-scale.json",dict(**cal,signature=inv["signature"]))
    arrays,stats=train_data(folder,rows,cal["k"],cal["radius_px"])
    print("Training",len(arrays[0]),"regions",cal,stats,flush=True)
    heads=train_heads(arrays)
    models=unpack(heads)
    multipliers=np.array([0.,.25,.5,1.])
    scores=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row[0]!="val":
            continue
        d=load(folder,row)
        mask=valid(d["z"])
        if not mask.any():
            continue
        ratio=cal["k"]*d["base"][mask]/d["z"][mask]
        for arm,h in direction(d["X"],d["area"],heads,models).items():
            c=h[d["lab"][mask]]
            scores[arm][row[1]].append([float(np.abs(ratio*np.exp(s*c)-1).mean()) for s in multipliers])
    dev={k:np.mean([np.mean(v,axis=0) for v in scenes.values()],axis=0).tolist()
         for k,scenes in scores.items()}
    choice={k:float(multipliers[np.argmin(v)]) for k,v in dev.items()}
    payload=dict(**heads,k=cal["k"],radius=cal["radius_px"],choice=choice,ckpt=inv["ckpt"])
    temp=OUT/"student.partial";torch.save(payload,temp);temp.replace(OUT/"student.pt")
    write(OUT/"selection.json",dict(**cal,teacher=stats,dev=dev,choice=choice,seed=SEED,
          fit_regions=len(arrays[0]),fit_code=sha(__import__("pathlib").Path(__file__)),
          dev_scenes=len(scores["joint"]),student_sha256=sha(OUT/"student.pt"),ckpt=inv["ckpt"]))
    print("Frozen",choice,dev,flush=True)

if __name__=="__main__":
    main()
