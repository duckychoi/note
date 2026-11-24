"""Exploratory output-space test. Fit compact conditional flow and deterministic regressors to angles."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run import CACHE,HERE,SEEDS,Data,load,make_planner,prior_for
from controls import features,fit_ridge
from analyse_gate import ALPHAS,score,summarize,bootstrap


class AngleResidual(nn.Module):
    def __init__(self,width,flow):
        super().__init__()
        self.flow=flow
        self.net=nn.Sequential(nn.Linear(width+(6 if flow else 0),128),nn.SiLU(),
                               nn.Linear(128,128),nn.SiLU(),nn.Linear(128,5))

    def objective(self,x,y,rng):
        if not self.flow:
            return F.mse_loss(self.net(x),y)
        noise=torch.randn(y.shape,generator=rng)
        time=torch.rand(len(y),1,generator=rng)
        middle=torch.lerp(noise,y,time)
        v=self.net(torch.cat((x,middle,time),1))
        return F.mse_loss(v,y-noise)

    @torch.no_grad()
    def residual(self,x):
        if not self.flow:
            return self.net(x)[:,None].numpy()
        rng=torch.Generator().manual_seed(53820)
        output=[]
        for batch in x.split(64):
            batch=batch.repeat_interleave(32,0)
            state=torch.randn(len(batch),5,generator=rng)
            for step in range(8):
                time=torch.full((len(batch),1),step/8)
                state=state+self.net(torch.cat((batch,state,time),1))/8.
            output.append(state.reshape(-1,32,5))
        return torch.cat(output).numpy()


def read_data():
    bank={}
    visual=load("reacher")
    splits={s:Data("damp_"+s,visual,bank) for s in ("train","val","angle_holdout")}
    net=make_planner(splits["train"])
    data={}
    for s,x in splits.items():
        p=prior_for(net,x,"probe")
        with torch.no_grad():
            observed=net.read_angle(x.z).cpu().numpy()
        with np.load(CACHE/f"damp_{s}.npz") as d:
            data[s]={"observed":observed,"k":p.strength.cpu().numpy(),"a":d["action"],
                     "prior":p.angles.cpu().numpy(),"peak":d["peak"],"goal":d["goal"],"truth":d["angles"][:,2:]}
    return data


def eval_paths(paths,d):
    return score(paths,d["truth"],d["peak"],d["goal"])


def run():
    data=read_data()
    tr,val,test=(data[s] for s in ("train","val","angle_holdout"))
    X=features(tr)
    mu,sd=X.mean(0),X.std(0).clip(.001)
    norm=lambda d:torch.as_tensor((features(d)-mu)/sd,dtype=torch.float32)
    x,xval,xtest=(norm(d) for d in (tr,val,test))
    y=tr["observed"][:,2:]-tr["prior"]
    bias,scale=y.mean(0),y.std(0).clip(.005)
    y=torch.as_tensor((y-bias)/scale,dtype=torch.float32)
    ridge=fit_ridge(tr)
    case={"truth":test["truth"],"peak":test["peak"],"goal":test["goal"],"prior":test["prior"],
          "ridge":test["prior"]+ridge.predict(features(test))}
    states={k:eval_paths(case[k],test) for k in ("prior","ridge")}
    res={"checkpoints":{},"alpha":{},"metrics":{},"contrasts":{}}
    for method in ("mlp","angle_flow"):
        trials=[]
        for seed in SEEDS:
            torch.manual_seed(seed)
            net=AngleResidual(x.shape[1],method=="angle_flow")
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0005)
            batches=torch.Generator().manual_seed(seed)
            noise=torch.Generator().manual_seed(10000+seed)
            best=np.inf
            for step in range(1,1601):
                ix=torch.randint(len(x),(128,),generator=batches)
                loss=net.objective(x[ix],y[ix],noise)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step%200==0:
                    delta=bias+scale*net.residual(xval)
                    err=np.abs(val["prior"]+delta.mean(1)-val["truth"]).mean()
                    if err<best:
                        best=err
                        state={k:v.detach().clone() for k,v in net.state_dict().items()}
                        res["checkpoints"][f"{method}_{seed}"]=step
            net.load_state_dict(state)
            vdelta=bias+scale*net.residual(xval)
            vs={a:eval_paths(val["prior"][:,None]+a*vdelta,val) for a in ALPHAS}
            alpha=max(ALPHAS,key=lambda a:(vs[a]["success"].mean(),-vs[a]["mae"].mean(),-a))
            res["alpha"][f"{method}_{seed}"]=alpha
            delta=bias+scale*net.residual(xtest)
            paths=test["prior"][:,None]+alpha*delta
            case[f"{method}_{seed}"]=paths
            trials.append(eval_paths(paths,test))
        states[method]={k:np.stack([t[k] for t in trials]).mean(0) for k in ("mae","success","peak_mae","reachable")}
    compare={}
    for other in ("prior","ridge","mlp"):
        compare[other+"_mae"]=states[other]["mae"]-states["angle_flow"]["mae"]
        compare[other+"_success"]=states["angle_flow"]["success"]-states[other]["success"]
    ci=bootstrap(np.column_stack(list(compare.values())),comparisons=6)
    res["contrasts"]={k:[float(d.mean()),*b.tolist()] for (k,d),b in zip(compare.items(),ci)}
    res["metrics"]={k:summarize(v) for k,v in states.items()}
    res["reachable"]=int(states["prior"]["reachable"].sum())
    res["n"]=len(states["prior"]["reachable"])
    np.savez_compressed(HERE/"angle_holdout.npz",**case)
    (HERE/"angle_results.json").write_text(json.dumps(res,indent=2),encoding="utf8")
    print(res["metrics"],res["contrasts"],flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    run()
