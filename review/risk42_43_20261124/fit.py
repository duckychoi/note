"""Fit the correction, learn its risk reduction on other sources and distill both."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from acquire import CACHE, HERE

sys.path.insert(0,str(HERE.parent/"gain42_studio43_20261124"))
from scenes import COUNTS, DepthNet, disparities, pair, tensor_input

SEEDS=(5271,5279,5281,5297,5303)
ARMS=("replay","prior","correct","utility","shuffle","true_utility")
CKPT=CACHE/"checkpoints"


def observe(left,right,focal,baseline,offset,disp_range):
    gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in (left,right)]
    if disp_range==48:
        ds=disparities(*gray)
    else:
        matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=disp_range,blockSize=3,
                 P1=72,P2=288,uniquenessRatio=8,speckleWindowSize=0)
        ds=(matcher.compute(*gray).astype(np.float32)/16,
            np.fliplr(matcher.compute(np.fliplr(gray[1]).copy(),np.fliplr(gray[0]).copy())).astype(np.float32)/16)
    y,x=np.indices(gray[0].shape)
    views=[]
    for i,sign in enumerate((-1,1)):
        d=ds[i]
        other_x=np.rint(x+sign*d).astype(int)
        inside=(other_x>=0)&(other_x<x.shape[1])
        reverse=ds[1-i][y,other_x.clip(0,x.shape[1]-1)]
        z=focal*baseline/np.maximum(d+offset,.01)
        ok=inside&(d>=0)&(reverse>=0)&(d+offset>.25)&(np.abs(reverse-d)<=1)&(z>=.15)&(z<=12)
        grayf=gray[i].astype(np.float32)/255
        grad=np.hypot(cv2.Sobel(grayf,-1,1,0),cv2.Sobel(grayf,-1,0,1))/4
        local=cv2.dilate(d,np.ones((3,3),np.uint8))-cv2.erode(d,np.ones((3,3),np.uint8))
        views.append((z.astype(np.float32),ok,grad,local))
    return views


def bank(seed,split,n):
    parents=np.concatenate([b*8+np.repeat(np.arange(8),COUNTS) for b in range(4)]) if split==5 else np.arange(n)
    rows=[]
    for i,parent in enumerate(parents):
        variant=int(np.sum(parents[:i]==parent))
        frames,f,b=pair(seed,split,int(parent),variant)
        obs=observe(frames[0][0],frames[1][0],f,b,0,48)
        for (im,z,_),s in zip(frames,obs):
            rows.append((tensor_input(im,f),z,*s))
    out={"x":torch.stack([r[0] for r in rows]).cuda()}
    for i,k in enumerate(("truth","stereo","ok","grad","spread"),1):
        out[k]=np.stack([r[i] for r in rows])
    return out


def predict(net,x):
    with torch.no_grad():
        return torch.cat([net(b)[:,0].exp() for b in x.split(8)]).cpu().numpy()


def med(values,weights):
    order=np.argsort(values)
    cumulative=np.cumsum(weights[order],dtype=np.float64)
    return float(values[order[np.searchsorted(cumulative,cumulative[-1]/2)]])


def bins(m,s,g,spread):
    rel=np.log(np.maximum(s,.01)/m)
    return ((np.digitize(rel,[-.55,-.15,0,.15,.55])*2+(g>.09))*2+(spread>1.5)).astype(np.int16)


def estimate_gain(z,m,s):
    step=s-m
    ok=np.abs(step)>1e-6
    return float(np.clip(med((z[ok]-m[ok])/step[ok],np.abs(step[ok])),0,1)) if ok.any() else 0.


def calib_gain(net,data):
    ok=data["ok"]
    m=predict(net,data["x"])
    group=bins(m,data["stereo"],data["grad"],data["spread"])[ok]
    z,m,s=data["truth"][ok],m[ok],data["stereo"][ok]
    single=estimate_gain(z,m,s)
    gain=np.full(24,single,np.float32)
    for i in range(24):
        mask=group==i
        if mask.sum()>=128:
            gain[i]=estimate_gain(z[mask],m[mask],s[mask])
    return single,gain


def correct(m,data,gain):
    group=bins(m,data["stereo"],data["grad"],data["spread"])
    revised=m+np.where(data["ok"],gain[group],0)*(data["stereo"]-m)
    return revised,group


def calib_value(net,data,gain):
    m=predict(net,data["x"])
    target,group=correct(m,data,gain)
    ok=data["ok"]
    improvement=np.abs(m-data["truth"])-np.abs(target-data["truth"])
    avg=improvement[ok].mean()
    count=np.bincount(group[ok],minlength=24)
    sums=np.bincount(group[ok],weights=improvement[ok],minlength=24)
    table=((sums+128*avg)/(count+128)).astype(np.float32)
    positive=table[group[ok]]
    cap=float(np.quantile(positive[positive>0],.9)) if (positive>0).any() else 1.
    return table,cap


def audit(net,data,constant,gain,table):
    m=predict(net,data["x"])
    revised,group=correct(m,data,gain)
    ok=data["ok"]
    truth=data["truth"][ok]
    diff=np.abs(m[ok]-truth)-np.abs(revised[ok]-truth)
    predicted=table[group[ok]]
    order=np.argsort(predicted,kind="stable")
    parts=np.array_split(order,4)
    return dict(coverage=float(ok.mean()),n=int(ok.sum()),positive_fraction=float((predicted>0).mean()),
       mae_mm={name:float(np.mean(np.abs(value-truth))*1000) for name,value in (
            ("prior",m[ok]),("stereo",data["stereo"][ok]),("fixed",m[ok]+constant*(data["stereo"][ok]-m[ok])),
            ("corrected",revised[ok]))},
       quartiles=[dict(n=len(idx),pred_mm=float(predicted[idx].mean()*1000),true_mm=float(diff[idx].mean()*1000))
                  for idx in parts],positive_precision=float((diff[predicted>0]>0).mean()) if (predicted>0).any() else None)


def terms(error):
    return error.abs().mean()+.5*(error.diff(dim=1).abs().mean()+error.diff(dim=2).abs().mean())


def train(net,data,steps,seed,arm,aux=None,weights=None):
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001)
    z=torch.from_numpy(np.log(data["truth"])).cuda()
    rng=np.random.default_rng(seed)
    indices=[]
    for t in range(steps):
        n=len(z) if steps==80 else (t//48+1)*len(z)//4
        idx=rng.integers(n,size=4)
        indices.extend(idx)
        out=net(data["x"][idx])[:,0]
        loss=terms(out-z[idx])
        if aux is not None:
            w=weights[idx]
            diff=(out-aux[idx]).abs()
            loss=loss+.35*((w*diff).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(1e-6)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    return hashlib.sha256(np.int32(indices).tobytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    CKPT.mkdir(exist_ok=True,parents=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        teacher=DepthNet(False).cuda()
        warm=bank(seed,0,48)
        train(teacher,warm,80,seed,"warm")
        torch.save(teacher.state_dict(),CKPT/f"{seed}_prior_net.pt")
        del warm
        teacher.eval()
        gain_data=bank(seed,1,32)
        constant,gain=calib_gain(teacher,gain_data)
        del gain_data
        val_data=bank(seed,2,32)
        value,cap=calib_value(teacher,val_data,gain)
        del val_data
        scale=bank(seed,3,16)
        # Fixed lattice for scale-only controls, distinct from the calibration and audit sets.
        scale_z=scale["truth"][:,::4,::4].ravel()
        rec=dict(seed=seed,fixed_gain=constant,gain=gain.tolist(),utility_m=value.tolist(),cap=cap,
                 synth_constant=float(np.median(scale_z)),scales={},batch_hashes={})
        test=bank(seed,4,16)
        rec["audit"]=audit(teacher,test,constant,gain,value)
        del test
        full=bank(seed,5,128)
        m=predict(teacher,full["x"])
        T,G=correct(m,full,gain)
        mask=full["ok"]
        w=np.where(mask,np.clip(value[G]/cap,0,1),0).astype(np.float32)
        sh=w.copy()
        rng=np.random.default_rng(seed+91)
        for j,valid in enumerate(mask):
            sh[j,valid]=rng.permutation(w[j,valid])
        def gpu(x):
            return torch.from_numpy(np.asarray(x,np.float32)).cuda()
        target,prior,wt,permuted,support=map(gpu,(np.log(T),np.log(m),w,sh,mask))
        gt=gpu(np.log(full["truth"]))
        for arm in ARMS:
            net=copy.deepcopy(teacher).train()
            aux={"replay":None,"prior":prior,"correct":target,"utility":target,"shuffle":target,"true_utility":gt}[arm]
            weight={"prior":support,"correct":support,"utility":wt,"shuffle":permuted,"true_utility":wt}.get(arm)
            rec["batch_hashes"][arm]=train(net,full,192,seed+81,arm,aux,weight)
            torch.save(net.state_dict(),CKPT/f"{seed}_{arm}.pt")
            y=predict(net.eval(),scale["x"])[:,::4,::4].ravel()
            rec["scales"][arm]=med(scale_z/y,y)
        # Strong stereo controls: their own scale on the same synthetic calibration coordinates.
        for k in ("replay","utility"):
            net=DepthNet(False).cuda()
            net.load_state_dict(torch.load(CKPT/f"{seed}_{k}.pt",weights_only=True))
            y=predict(net.eval(),scale["x"])
            for name,estimate in stereo_arms(y,scale,gain,constant,k).items():
                pred=estimate[:,::4,::4].ravel()
                rec["scales"][name]=med(scale_z/pred,pred)
        (HERE/f"seed{seed}.json").write_text(json.dumps(rec,indent=1),encoding="utf8")
        print("completed",seed,rec["audit"]["mae_mm"],flush=True)


def stereo_arms(m,data,gain,fixed,arm):
    y,_=correct(m,data,gain)
    if arm=="utility":
        return {"student_stereo":y}
    s=np.where(data["ok"],data["stereo"],m)
    return {"raw_stereo":s,"fixed_stereo":m+fixed*(s-m),"replay_stereo":y}


if __name__=="__main__":
    main()
