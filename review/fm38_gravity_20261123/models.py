"""Learn in the low-incline range, compare paired observations on both ranges."""
import json
import hashlib
import sys
import shutil

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from swing import BAND,CACHE,HERE,raster
sys.path.insert(0,str(HERE.parent/"fm38_materials_20261123"))
from public_model import load,vision,action_tensor

SEEDS=(4821,4822,4823)
MODES=("photo","probe")
DOMAINS=("source","shift")
SAMPLES=32


def ensure_reacher_checkpoint():
    folder=CACHE.parent/"checkpoints"/"reacher"
    revision="f26a1a5f3d1e0aeeb68377a7e7463088f244c79c"
    for filename in ("config.json","weights_epoch_20.pt"):
        if not (folder/filename).is_file():
            from huggingface_hub import hf_hub_download
            cached=hf_hub_download("HuoYanchen/Flow-JEPA",f"flow-jepa-reacher/{filename}",revision=revision)
            folder.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(cached,folder/filename)


class Data:
    def __init__(self,split,visual,bank):
        with np.load(CACHE/f"{split}.npz") as f:
            self.a=torch.as_tensor(f["action"],device="cuda")
            self.goal=torch.as_tensor(f["goal"],device="cuda",dtype=torch.float32)
            self.angles=torch.as_tensor(f["angles"],device="cuda",dtype=torch.float32)
            # Reusing a frozen representation of identical rasters has no label dependency.
            keys=[]
            new={}
            for q in f["angles"].ravel():
                im=raster(q)
                key=hashlib.blake2b(im,digest_size=16).digest()
                keys.append(key)
                if key not in bank and key not in new:
                    new[key]=im
            novel=list(new)
            for i in range(0,len(novel),64):
                kk=novel[i:i+64]
                zs=vision(visual,np.stack([new[k] for k in kk])).cpu().numpy()
                bank.update(zip(kk,zs))
            z=np.stack([bank[k] for k in keys]).reshape(*f["angles"].shape,192)
            self.z=torch.as_tensor(z,device="cuda")


class AngleReader(nn.Module):
    def __init__(self,tr=None,*,stats=None):
        super().__init__()
        center,scale=stats if stats is not None else (tr.z.mean((0,1)),tr.z.std((0,1)).clamp_min(.015))
        self.register_buffer("center",center)
        self.register_buffer("scale",scale)
        self.net=nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,z):
        return self.net((z-self.center)/self.scale).squeeze(-1)


def reader(tr,val):
    torch.manual_seed(4820)
    model=AngleReader(tr).cuda()
    x=tr.z.flatten(0,1)
    y=tr.angles.flatten()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    best=np.inf
    for t in range(1,1001):
        ix=torch.randint(len(y),(256,),device="cuda")
        err=F.smooth_l1_loss(model(x[ix]),y[ix],beta=.05)
        opt.zero_grad();err.backward();opt.step()
        if t%125==0:
            with torch.no_grad():
                mae=(model(val.z)-val.angles).abs().mean().item()
                if mae<best:
                    saved={k:v.clone() for k,v in model.state_dict().items()}
                    best=mae
    model.load_state_dict(saved)
    return model.eval(),float(np.rad2deg(best))


def observations(z,with_probe):
    start=z[:,0]
    practice=torch.where(with_probe[:,None],z[:,1],start)
    return torch.stack([start,practice,start],1)


class Mixture(nn.Module):
    def __init__(self,tr):
        super().__init__()
        self.register_buffer("center",tr.z[:,:2].mean((0,1)))
        self.register_buffer("scale",tr.z[:,:2].std((0,1)).clamp_min(.02))
        self.register_buffer("dscale",(tr.z[:,2:]-tr.z[:,:1]).std((0,1)).clamp_min(.02))
        self.net=nn.Sequential(nn.Linear(388,384),nn.SiLU(),nn.Linear(384,384),nn.SiLU(),
                               nn.Linear(384,4*(1+2*5*192)))

    def forward(self,h,a):
        inp=torch.cat([((h[:,:2]-self.center)/self.scale).flatten(1),F.one_hot(a,4).float()],1)
        output=self.net(inp).reshape(-1,4,1+2*5*192)
        mu,ls=output[:,:,1:].reshape(-1,4,2,5,192).unbind(2)
        return output[:,:,0],mu,ls.clamp(-3,2)

    def loss(self,h,a,target):
        w,mu,ls=self(h,a)
        delta=(target-h[:,:1])/self.dscale
        energy=(.5*((delta[:,None]-mu)*(-ls).exp()).square()+ls).sum((-2,-1))
        return -torch.logsumexp(w.log_softmax(1)-energy,1).mean()/(5*192)

    def sample(self,h,a):
        w,mu,ls=self(h,a)
        comp=torch.multinomial(w.softmax(1),SAMPLES,replacement=True)
        idx=torch.arange(len(a),device="cuda")[:,None]
        draw=mu[idx,comp]+ls[idx,comp].exp()*torch.randn((len(a),SAMPLES,5,192),device="cuda")
        return h[:,None,:1]+draw*self.dscale


class Retrieval:
    def __init__(self,tr):
        self.tr=tr
        self.scale=tr.z[:,:2].std((0,1)).clamp_min(.02)

    def sample(self,h,a,mode):
        candidate=self.tr.z[:,:2] if mode=="probe" else self.tr.z[:,[0,0]]
        dst=torch.cdist((h[:,:2]/self.scale).flatten(1),(candidate/self.scale).flatten(1))
        dst.masked_fill_(a[:,None]!=self.tr.a[None],float("inf"))
        return self.tr.z[dst.topk(SAMPLES,largest=False).indices,2:]


def loss(name,net,data,ix):
    # The same demonstration and target appear with and without the practice frame.
    z=data.z[ix].repeat(2,1,1)
    mask=torch.arange(len(z),device="cuda")>=len(ix)
    h=observations(z,mask)
    a=data.a[ix].repeat(2)
    if name=="mixture":
        return net.loss(h,a,z[:,2:])
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return net.flow_loss(h[:,:,None],net.action_encoder(action_tensor("reacher",a)),z[:,2:,None])


def train(name,tr,val,seed):
    torch.manual_seed(seed)
    net=load("reacher") if name=="flow" else Mixture(tr).cuda()
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=7e-5 if name=="flow" else 7e-4,weight_decay=.0005,fused=True)
    gen=torch.Generator(device="cuda").manual_seed(seed)
    best=np.inf
    metrics=[]
    for t in range(1,1001):
        net.train()
        ix=torch.randint(len(tr.a),(32,),device="cuda",generator=gen)
        e=loss(name,net,tr,ix)
        opt.zero_grad(set_to_none=True);e.backward()
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step()
        if t%200==0:
            net.eval()
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(4829)
                v=torch.stack([loss(name,net,val,i) for i in torch.arange(len(val.a),device="cuda").split(32)]).mean().item()
            metrics.append([t,v])
            if v<best:
                saved={k:x.detach().clone() for k,x in net.state_dict().items() if
                       name=="mixture" or k.startswith(("predictor.","action_encoder."))}
                best=v
    assert np.isfinite(best)
    if name=="mixture":
        net.load_state_dict(saved)
    else:
        for key in ("predictor","action_encoder"):
            getattr(net,key).load_state_dict({k[len(key)+1:]:v for k,v in saved.items() if k.startswith(key+".")})
    torch.save(saved,CACHE/f"{name}_{seed}.pt")
    return net.eval(),metrics


@torch.inference_mode()
def predict(name,net,decode,data,mode):
    torch.manual_seed(4830)
    pred=[]
    for i in range(0,len(data.a),16):
        z=data.z[i:i+16]
        h=observations(z,torch.full((len(z),),mode=="probe",device="cuda"))
        a=data.a[i:i+16]
        if name=="flow":
            with torch.autocast("cuda",dtype=torch.bfloat16):
                act=net.action_encoder(action_tensor("reacher",a)).repeat_interleave(SAMPLES,0)
                future=net.predict(h.repeat_interleave(SAMPLES,0)[:,:,None],act,num_steps=8)
                future=future.reshape(len(a),SAMPLES,5,192).float()
        else:
            future=net.sample(h,a,mode) if name=="retrieval" else net.sample(h,a)
        peak=decode(future).amax(-1)
        dist=(peak-data.goal[i:i+len(z),None]).abs()
        pred.append(torch.stack([(dist<=BAND).float().mean(1),dist.mean(1)],-1))
    return torch.cat(pred).cpu().numpy().reshape(-1,4,2)


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    ensure_reacher_checkpoint()
    bank={}
    visual=load("reacher")
    splits={s:Data(s,visual,bank) for s in ("train","val",*DOMAINS)}
    tr,val=splits["train"],splits["val"]
    read,val_error=reader(tr,val)
    torch.save(read.state_dict(),CACHE/"reader.pt")
    meta={"source_val_angle_MAE_deg":val_error,"fit":{},"observed_future":{}}
    with torch.no_grad():
        for d in DOMAINS:
            x=splits[d]
            ang=read(x.z[:,2:])
            with np.load(CACHE/f"{d}.npz") as records:
                gt=records["peak"]
            meta["observed_future"][d]=dict(
                angle_MAE_deg=float(torch.rad2deg((ang-x.angles[:,2:]).abs()).mean()),
                five_times_peak_MAE_deg=float(np.rad2deg(abs(x.angles[:,2:].amax(1).cpu().numpy()-gt)).mean()),
                peak_MAE_deg=float(np.rad2deg(abs(ang.amax(1).cpu().numpy()-gt)).mean()))
    raw={}
    for name in ("retrieval","mixture","flow"):
        for seed in (SEEDS[:1] if name=="retrieval" else SEEDS):
            print("start",name,seed,flush=True)
            if name=="retrieval":
                net=Retrieval(tr)
            else:
                net,meta["fit"][f"{name}_{seed}"]=train(name,tr,val,seed)
            for mode in MODES:
                for d in DOMAINS:
                    key=f"{name}_{seed}_{mode}_{d}"
                    if d=="shift" and mode=="photo":
                        assert torch.equal(splits["source"].z[:,0],splits["shift"].z[:,0])
                        assert torch.equal(splits["source"].goal,splits["shift"].goal)
                        raw[key]=raw[f"{name}_{seed}_{mode}_source"]
                    else:
                        raw[key]=predict(name,net,read,splits[d],mode)
            np.savez_compressed(HERE/"forecasts.npz",**raw)
            (HERE/"run.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
            print("finished",name,seed,flush=True)
