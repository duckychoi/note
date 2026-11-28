import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import CACHE, DEV, HERE
from studio import FAMILIES, base_depth, categories, real_frame, render

SEEDS=(431,433,439)
ARMS=("zero_random","zero_synthetic","few_random","few_guided")


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())


class DepthNet(nn.Module):
    def __init__(self,prior):
        super().__init__()
        self.prior=prior
        self.local=block(5,12)
        self.middle=block(12,20)
        self.context=block(20,28)
        self.merge=block(48,20)
        self.result=nn.Sequential(block(32,12),nn.Conv2d(12,1,1))
        nn.init.zeros_(self.result[-1].weight)
        nn.init.zeros_(self.result[-1].bias)

    def forward(self,x):
        # Full-frame depth context, then restore edges at the input resolution.
        a=self.local(x)
        b=self.middle(F.avg_pool2d(a,2))
        c=self.context(F.avg_pool2d(b,4))
        def up(y,other):
            return F.interpolate(y,size=other.shape[-2:],mode="bilinear",align_corners=False)
        d=self.merge(torch.cat((b,up(c,b)),1))
        raw=self.result(torch.cat((a,up(d,a)),1))
        if self.prior:
            corr=.45*torch.tanh(raw)
            return x[:,3:4]+corr-corr.flatten(2).median(-1).values[:,:,None,None]
        return math.log(2.)+1.9*torch.tanh(raw)


def tensor_input(im,fx,base=None):
    rgb=(im.astype(np.float32)/127.5-1).transpose(2,0,1)
    h,w=im.shape[:2]
    prior=np.zeros((h,w),np.float32) if base is None else np.log(base)
    focal=np.full((h,w),np.log(fx/w),np.float32)
    return torch.from_numpy(np.concatenate((rgb,prior[None],focal[None]))).float()


def pack(seed,index,phase,family,prior):
    key=f"r{seed}_{phase}_{index}_{family}"
    folder=CACHE/"renders"/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+".npz")
    if not path.exists():
        image,depth,fx=render(seed,index,phase,int(family))
        np.savez(path,image=image,depth=depth,fx=fx)
    with np.load(path) as d:
        image,depth,fx=d["image"],d["depth"],float(d["fx"])
    base=base_depth(key,image,fx) if prior else None
    return tensor_input(image,fx,base),torch.from_numpy(np.log(depth)[None]),categories(image)


def make_bank(seed,phase,families,prior):
    data=[pack(seed,i,phase,int(f),prior) for i,f in enumerate(families)]
    return torch.stack([r[0] for r in data]),torch.stack([r[1] for r in data]),np.stack([r[2] for r in data])


def train_steps(net,bank,seed,steps):
    x,z,_=bank
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001)
    net.train()
    for _ in range(steps):
        idx=rng.integers(0,len(x),4)
        xb,yb=x[idx].cuda(),z[idx].cuda()
        pred=net(xb)
        diff=pred-yb
        loss=diff.abs().mean()+.5*((diff[:,:,:,1:]-diff[:,:,:,:-1]).abs().mean()+
                                   (diff[:,:,1:]-diff[:,:,:-1]).abs().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    return float(loss.detach().cpu())


@torch.no_grad()
def infer(net,x):
    return net(x[None].cuda())[0,0].cpu().numpy()


def mixture(err,labels):
    average=float(np.mean(err))
    score=np.array([(err[labels==i].sum()+4*average)/(np.sum(labels==i)+4) for i in range(4)])
    weight=(score+.01)**2
    return (.125+.5*weight/weight.sum()).tolist(), score.tolist()


def synthetic_profile(net,bank):
    x,z,labels=bank
    diffs,regions=[],[]
    for xi,yi,region in zip(x,z,labels):
        delta=yi[0].numpy()-infer(net,xi)
        diffs.append(np.abs(delta-np.median(delta))[::4,::4].ravel())
        regions.append(region[::4,::4].ravel())
    return mixture(np.concatenate(diffs),np.concatenate(regions))


def measured_profile(net,rows,prior):
    scale,errors,labels=[],[],[]
    successes=[]
    for row in rows:
        im,gt,fx=real_frame(row)
        pred=infer(net,tensor_input(im,fx,base_depth(row["key"],im,fx) if prior else None))
        h,w=gt.shape
        iy,ix=np.indices((4,8))
        y=np.floor((iy+.5)*h/4).astype(int)
        x=np.floor((ix+.5)*w/8).astype(int)
        # Only these 32 entries can be used. No neighboring values or replacements.
        z=gt[y,x]
        valid=(z>=.2)&(z<=10)
        fit=(ix+iy)%2==0
        value=np.log(np.maximum(z,.01))-pred[y,x]
        scale.extend(value[fit&valid])
        errors.extend(value[~fit&valid])
        labels.extend(categories(im)[y,x][~fit&valid])
        successes.append([int((fit&valid).sum()),int((~fit&valid).sum())])
    gain=float(np.median(scale)) if scale else 0.
    if errors:
        probs,scores=mixture(np.abs(np.array(errors)-gain),np.array(labels))
    else:
        probs,scores=[.25]*4,[0.]*4
    return gain,probs,scores,successes


def run():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
    inputs=json.loads((HERE/"inputs.json").read_text())["rows"]
    dev=[r for r in inputs if r["scene"] in DEV]
    assert len(dev)==2
    out=CACHE/"checkpoints"
    out.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for prior in (False,True):
        family="adapter" if prior else "scratch"
        for seed in SEEDS:
            torch.manual_seed(seed)
            warm=DepthNet(prior).cuda()
            pre=make_bank(seed,0,np.tile(np.arange(4),16),prior)
            warm_loss=train_steps(warm,pre,seed,80)
            del pre
            scout=make_bank(seed,1,np.tile(np.arange(4),8),prior)
            synth_probs,synth_scores=synthetic_profile(warm.eval(),scout)
            del scout
            gain,real_probs,real_scores,valid=measured_profile(warm,dev,prior)
            uniform=np.random.default_rng(seed+2000).random(128)
            # zero_random and few_random have identical synthetic supervision.
            # Charge the same budget; only their deployment scale differs.
            records={}
            for arm in ARMS:
                few=arm.startswith("few")
                probs=real_probs if arm=="few_guided" else synth_probs if arm=="zero_synthetic" else [.25]*4
                tags=np.searchsorted(np.cumsum(probs),uniform).clip(0,3)
                if arm=="few_random":
                    ckpt,loss=records["zero_random"]
                else:
                    bank=make_bank(seed,2,tags,prior)
                    model=copy.deepcopy(warm)
                    loss=train_steps(model,bank,seed+73,160)
                    ckpt=f"{family}_{seed}_{arm}.pt"
                    torch.save(model.state_dict(),out/ckpt)
                    del model,bank
                records[arm]=(ckpt,loss)
                ledger.append(dict(model=family,seed=seed,arm=arm,checkpoint=ckpt,
                            log_gain=gain if few else 0,probs=probs,counts=np.bincount(tags,minlength=4).tolist(),
                            phase2_families=tags.tolist(),scout_scores=real_scores if few else synth_scores,
                            dev_frames=2 if few else 0,dev_queries=64 if few else 0,
                            successful_split=valid if few else [],renders=[64,32,128],steps=[80,160],
                            batch=4,warm_loss=warm_loss,last_loss=loss))
                print(family,seed,arm,ledger[-1]["counts"],flush=True)
            (HERE/"training.json").write_text(json.dumps(ledger,indent=1),encoding="utf8")


if __name__=="__main__":
    run()
