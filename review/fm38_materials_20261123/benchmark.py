"""Train on original-material demonstrations. Predict from one RGB frame per decision."""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from public_model import HERE, action_tensor, load, vision
from worlds import BAND, ROOT, TASKS, labels, render

SEEDS=(3811,3812,3813)
MODELS=("flow","mixture","retrieval")
STEPS=500
SAMPLES=32
SAFE=(.1,.3,.5,.7)


class Data:
    def __init__(self,task,split):
        with np.load(ROOT/f"{task}_{split}.npz") as data:
            to_gpu=lambda x:torch.as_tensor(x,device="cuda")
            self.z=to_gpu(np.load(ROOT/f"{task}_{split}_z.npy"))
            self.a=to_gpu(data["action"])
            self.y=to_gpu(labels(task,data))
            self.touch=to_gpu(data["contact"]).float()
            self.start=data["path"][:,0].copy()
            half=(BAND[task][1]-BAND[task][0])/2
            self.offset=to_gpu((data["path"][:,:,0]-data["path"][:,:,-1])/half).float()


class Probe(nn.Module):
    def __init__(self,train):
        super().__init__()
        self.register_buffer("center",train.z.mean((0,1)))
        self.register_buffer("scale",train.z.std((0,1)).clamp_min(.02))
        self.net=nn.Sequential(nn.Linear(192,192),nn.SiLU(),nn.Linear(192,96),nn.SiLU(),nn.Linear(96,3))
        self.use_class=True

    def forward(self,z):
        return self.net((z-self.center)/self.scale)

    def hit(self,z,task):
        out=self(z)
        if self.use_class:
            return out[...,2]>0
        hit=out[...,0].abs()<=1
        return hit if task=="push" else hit & (out[...,1]>0)


def metrics(pred,true):
    recall=pred[true].float().mean().item()
    specificity=(~pred[~true]).float().mean().item()
    return dict(recall=recall,specificity=specificity,balanced=(recall+specificity)/2,
                accuracy=(pred==true).float().mean().item())


def fit_probe(task,train,val):
    torch.manual_seed(3810)
    model=Probe(train).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    target=torch.stack([train.offset,train.touch,train.y.float()],-1).reshape(-1,3)
    z=train.z.flatten(0,1)
    best=-1
    for t in range(1,1001):
        ix=torch.randint(len(z),(256,),device="cuda")
        out=model(z[ix])
        loss=F.smooth_l1_loss(out[:,0],target[ix,0])+F.binary_cross_entropy_with_logits(out[:,1:],target[ix,1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if t%200==0:
            with torch.no_grad():
                for scheme in (True,False):
                    model.use_class=scheme
                    v=metrics(model.hit(val.z[:,-3:],task),val.y[:,-3:])["balanced"]
                    if v>best:
                        best=v
                        saved=({k:x.clone() for k,x in model.state_dict().items()},scheme,t)
    model.load_state_dict(saved[0])
    model.use_class=saved[1]
    return model.eval(),{"direct_classifier":saved[1],"updates":saved[2],"val_balanced":best}


class Mixture(nn.Module):
    """Four diagonal Gaussian trajectory components with one component index per rollout."""
    def __init__(self,train):
        super().__init__()
        self.register_buffer("center",train.z[:,0].mean(0))
        self.register_buffer("scale",train.z[:,0].std(0).clamp_min(.02))
        self.register_buffer("delta_scale",(train.z[:,1:]-train.z[:,:1]).std((0,1)).clamp_min(.02))
        self.net=nn.Sequential(nn.Linear(196,384),nn.SiLU(),nn.Linear(384,384),nn.SiLU(),
                               nn.Linear(384,4*(1+2*5*192)))

    def forward(self,z,a):
        inputs=torch.cat(((z-self.center)/self.scale,F.one_hot(a,4).float()),-1)
        raw=self.net(inputs).reshape(-1,4,1+2*5*192)
        mu,ls=raw[:,:,1:].reshape(-1,4,2,5,192).unbind(2)
        return raw[:,:,0],mu,ls.clamp(-3.,2.)

    def loss(self,z,a,y):
        logits,mu,ls=self(z,a)
        delta=(y-z[:,None])/self.delta_scale
        energy=(.5*((delta[:,None]-mu)*(-ls).exp()).square()+ls).sum((-2,-1))
        return -torch.logsumexp(logits.log_softmax(1)-energy,1).mean()/(5*192)

    def sample(self,z,a,n):
        logits,mu,ls=self(z,a)
        idx=torch.multinomial(logits.softmax(1),n,replacement=True)
        batch=torch.arange(len(z),device=z.device)[:,None]
        pred=mu[batch,idx]+ls[batch,idx].exp()*torch.randn((len(z),n,5,192),device=z.device)
        return z[:,None,None]+self.delta_scale*pred


class Retrieval:
    def __init__(self,train):
        self.scale=train.z[:,0].std(0).clamp_min(.02)
        self.start=train.z[:,0]/self.scale
        self.path=train.z[:,1:]
        self.action=train.a

    def sample(self,z,a,n):
        dist=torch.cdist(z/self.scale,self.start)
        dist.masked_fill_(a[:,None]!=self.action[None],float("inf"))
        return self.path[dist.topk(n,largest=False).indices]


def dyn_loss(task,name,model,data,ix):
    z=data.z[ix]
    a=data.a[ix]
    if name=="mixture":
        return model.loss(z[:,0],a,z[:,1:])
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return model.flow_loss(z[:,:1,None],model.action_encoder(action_tensor(task,a)),z[:,1:,None])


def fit(task,name,train,val,seed):
    torch.manual_seed(seed)
    model=load(task) if name=="flow" else Mixture(train).cuda()
    pars=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(pars,lr=7e-5 if name=="flow" else .0007,weight_decay=.0005,fused=True)
    best=float("inf")
    ckpt=ROOT/f"{task}_{name}_{seed}.pt"
    history=[]
    batch_rng=torch.Generator(device="cuda").manual_seed(seed)
    for t in range(1,STEPS+1):
        model.train()
        ix=torch.randint(len(train.a),(48,),device="cuda",generator=batch_rng)
        loss=dyn_loss(task,name,model,train,ix)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pars,1.)
        opt.step()
        if t%125==0:
            model.eval()
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(3819)
                ix=torch.arange(len(val.a),device="cuda")
                v=torch.stack([dyn_loss(task,name,model,val,ii) for ii in ix.split(64)]).mean().item()
            history.append([t,v])
            if v<best:
                best=v
                state={k:x for k,x in model.state_dict().items() if name!="flow" or k.startswith(("predictor.","action_encoder."))}
                torch.save(state,ckpt)
    if not np.isfinite(best):
        raise ValueError(f"non-finite loss in {task}/{name}/{seed}")
    state=torch.load(ckpt,weights_only=True)
    if name=="flow":
        for k in ("predictor","action_encoder"):
            getattr(model,k).load_state_dict({s[len(k)+1:]:x for s,x in state.items() if s.startswith(k+".")})
    else:
        model.load_state_dict(state)
    return model.eval(),history


@torch.inference_mode()
def sample_paths(task,name,model,z,a):
    if name in ("flow","raw"):
        hist=z.repeat_interleave(SAMPLES,0)[:,None,None]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            act=model.action_encoder(action_tensor(task,a)).repeat_interleave(SAMPLES,0)
            return model.predict(hist,act,num_steps=8).reshape(len(z),SAMPLES,5,192).float()
    return model.sample(z,a,SAMPLES)


@torch.inference_mode()
def scores(task,name,model,probe,z,a):
    samples=sample_paths(task,name,model,z,a)
    hits=probe.hit(samples[:,:,-3:],task)
    joint=hits.all(-1).float().mean(1)
    independent=hits.float().mean(1).prod(-1)
    return torch.stack((joint,independent),-1)


def predictions(task,name,model,probe,data):
    torch.manual_seed(3820)
    return torch.cat([scores(task,name,model,probe,data.z[i:i+16,0],data.a[i:i+16])
                      for i in range(0,len(data.a),16)]).cpu().numpy().reshape(-1,4,2)


@torch.inference_mode()
def latency(task,name,model,visual,probe,data):
    # Single-state steady-state decision from a camera raster, including preprocessing, encoding,
    # prediction, decoding and copying the selected commands to the CPU.
    raster=[render(task,x) for x in data.start[::4][:28]]
    safe=torch.tensor(SAFE,device="cuda")
    act=torch.arange(4,device="cuda")
    times=[]
    for image in raster:
        torch.cuda.synchronize()
        t=time.perf_counter()
        z=vision(visual,image[None]).expand(4,-1)
        p=scores(task,name,model,probe,z,act).T
        peak,choice=p.max(1)
        torch.where(peak[:,None]>safe,choice[:,None],-1).cpu()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-t)*1000)
    return times[4:]


def run(task):
    train,val=Data(task,"train"),Data(task,"val")
    test,shift=Data(task,"source"),Data(task,"shift")
    assert np.array_equal(test.start,shift.start) and torch.equal(test.a,shift.a)
    assert np.array_equal(test.start[::4].repeat(4,axis=0),test.start)
    assert torch.all(test.a.reshape(-1,4)==torch.arange(4,device="cuda")).item()
    probe,fit_info=fit_probe(task,train,val)
    with torch.no_grad():
        for split,data in (("val",val),("source",test),("shift",shift)):
            fit_info[split]=metrics(probe.hit(data.z[:,-3:],task),data.y[:,-3:])
    torch.save({"state":probe.state_dict(),"use_class":probe.use_class},ROOT/f"{task}_probe.pt")
    visual=load(task)
    raw={}
    details={"readout":fit_info,"fit":{},"latency_ms":{},"seed":list(SEEDS)}
    for name in ("retrieval","raw","mixture","flow"):
        for seed in (SEEDS if name in ("mixture","flow") else SEEDS[:1]):
            print("start",task,name,seed,flush=True)
            if name in ("flow","mixture"):
                model,history=fit(task,name,train,val,seed)
                details["fit"][f"{name}_{seed}"]=history
            else:
                model=visual if name=="raw" else Retrieval(train)
            # Identical observed input in the two material worlds. Reuse exactly the same
            # Monte Carlo forecasts as well as actions to isolate the counterfactual outcome.
            raw[f"{name}_{seed}"]=predictions(task,name,model,probe,test)
            if seed==SEEDS[0]:
                details["latency_ms"][name]=latency(task,name,model,visual,probe,test)
            print("finished",task,name,seed,flush=True)
            np.savez_compressed(HERE/f"{task}_forecasts.npz",**raw)
            (HERE/f"{task}_run.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
            del model


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=TASKS)
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    for task in ([args.task] if args.task else TASKS):
        run(task)
