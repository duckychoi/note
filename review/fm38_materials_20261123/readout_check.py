"""Compare the original and sequence event readers on the same predicted button trajectories."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from analyse import DOMAINS,policy,bootstrap
from benchmark import Data,Mixture,Probe,Retrieval,SEEDS,metrics,sample_paths
from public_model import HERE,load,vision
from worlds import ROOT,labels,render


class Window(nn.Module):
    def __init__(self,data):
        super().__init__()
        self.register_buffer("center",data.z.mean((0,1)))
        self.register_buffer("scale",data.z.std((0,1)).clamp_min(.02))
        self.net=nn.Sequential(nn.Linear(576,192),nn.SiLU(),nn.Dropout(.1),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,z):
        return self.net(((z-self.center)/self.scale).flatten(-2,-1)).squeeze(-1)


def train_window(tr,val):
    torch.manual_seed(3881)
    model=Window(tr).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0006)
    x=torch.cat([tr.z[:,i:i+3] for i in range(4)])
    y=torch.cat([tr.y[:,i:i+3].all(1) for i in range(4)])
    pos,neg=y.nonzero().flatten(),(~y).nonzero().flatten()
    best=-1
    for step in range(1,1001):
        model.train()
        idx=torch.cat([row[torch.randint(len(row),(96,),device="cuda")] for row in (pos,neg)])
        loss=F.binary_cross_entropy_with_logits(model(x[idx]),y[idx].float())
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0:
            model.eval()
            with torch.no_grad():
                logits=model(val.z[:,-3:])
                for threshold in (-1.,-.5,0.,.5,1.):
                    v=metrics(logits>threshold,val.y[:,-3:].all(1))["balanced"]
                    if v>best:
                        best=v
                        saved=({k:w.clone() for k,w in model.state_dict().items()},threshold,step)
    model.load_state_dict(saved[0])
    return model.eval(),saved[1],{"threshold":saved[1],"update":saved[2],"val_balanced_accuracy":best}


@torch.inference_mode()
def evaluate(tr,probe,window,threshold):
    visual=load("press")
    truth={}
    probe_perf={}
    raw={}
    starts=[]
    for d in DOMAINS:
        with np.load(ROOT/f"press_check_{d}.npz") as data:
            states=data["path"]
            starts.append(states[:,0])
            flag=labels("press",data)[:,-3:].all(1)
        truth[d]=flag.reshape(-1,4)
        raw["truth_"+d]=truth[d]
        # Encode the observed true future, solely to diagnose perception.
        images=[render("press",x) for x in states[:,-3:].reshape(-1,7)]
        true_z=torch.cat([vision(visual,np.stack(images[i:i+64])) for i in range(0,len(images),64)]).reshape(-1,3,192)
        y=torch.as_tensor(flag,device="cuda")
        probe_perf[d]={"frames":metrics(probe.hit(true_z,"press").all(1),y),
                       "sequence":metrics(window(true_z)>threshold,y)}
    assert np.array_equal(*starts)
    images=np.stack([render("press",x) for x in starts[0][::4]])
    z=torch.cat([vision(visual,images[i:i+64]) for i in range(0,len(images),64)]).repeat_interleave(4,0)
    actions=torch.arange(4,device="cuda").repeat(len(z)//4)
    policy_results={}
    for name in ("flow","mixture","retrieval"):
        items=[]
        for seed in (SEEDS[:1] if name=="retrieval" else SEEDS):
            if name=="retrieval":
                model=Retrieval(tr)
            else:
                weights=torch.load(ROOT/f"press_{name}_{seed}.pt",weights_only=True)
                if name=="mixture":
                    model=Mixture(tr).cuda().eval()
                    model.load_state_dict(weights)
                else:
                    model=load("press")
                    for k in ("predictor","action_encoder"):
                        getattr(model,k).load_state_dict({s[len(k)+1:]:v for s,v in weights.items() if s.startswith(k+".")})
            torch.manual_seed(3910)
            batch=[]
            for i in range(0,len(z),16):
                paths=sample_paths("press",name,model,z[i:i+16],actions[i:i+16])[:,:,-3:]
                old=probe.hit(paths,"press").all(-1).float().mean(1)
                new=(window(paths)>threshold).float().mean(1)
                batch.append(torch.stack([old,new],1))
            p=torch.cat(batch).reshape(-1,4,2).cpu().numpy()
            raw[f"{name}_{seed}"]=p
            item={}
            for i,kind in enumerate(("frames","sequence")):
                item[kind]={}
                for d,y in truth.items():
                    u,enter,choice=policy(p[None,:,:,i],y)
                    item[kind][d]={"value":float(u.mean()),"attempt_rate":float(enter.mean()),
                                   "forced_success":float(y[np.arange(len(y)),choice[0]].mean())}
            items.append(item)
            del model
        policy_results[name]=items
    np.savez_compressed(HERE/"readout_check_scores.npz",**raw)
    return probe_perf,policy_results


def intervals():
    with np.load(HERE/"readout_check_scores.npz") as d:
        diffs={}
        for i,kind in enumerate(("frames","sequence")):
            for domain in DOMAINS:
                y=d["truth_"+domain]
                def by_state(name):
                    p=np.stack([d[k][:,:,i] for k in d if k.startswith(name+"_")])
                    return policy(p,y)[0].mean((0,2))
                diffs[kind+"_"+domain]=by_state("flow")-by_state("retrieval")
    ranges=bootstrap(np.column_stack(list(diffs.values())),comparisons=4)
    return {k:[float(values.mean()),*r.tolist()] for (k,values),r in zip(diffs.items(),ranges)}


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    tr,val=Data("press","train"),Data("press","val")
    old=torch.load(ROOT/"press_probe.pt",weights_only=True)
    probe=Probe(tr).cuda().eval()
    probe.load_state_dict(old["state"])
    probe.use_class=old["use_class"]
    window,threshold,meta=train_window(tr,val)
    torch.save({"state":window.state_dict(),"threshold":threshold},ROOT/"press_window.pt")
    readers,methods=evaluate(tr,probe,window,threshold)
    (HERE/"readout_check.json").write_text(json.dumps(dict(fit=meta,readers=readers,methods=methods,
                                                 flow_minus_retrieval=intervals()),indent=2),encoding="utf8")
