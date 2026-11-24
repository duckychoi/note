"""Branch one MuJoCo state into three impulses and evaluate learned future-distribution heads."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr
import torch
from torch import nn

from contact36 import SIGMA, model

HERE = Path(__file__).resolve().parent
ACTIONS = np.array([-.8,0,.8])
REGION = slice(8,11)  # future frames 20,21,22, with frame zero the initial height
SAFE = np.array([.05,.15,.30,.50])
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def dataset(damping,seed,n,all_actions):
    rng = np.random.default_rng(seed)
    models = [model(x) for x in damping]
    inputs,actions,outputs = [],[],[]
    for _ in range(n):
        m = models[rng.integers(len(models))]
        d = mujoco.MjData(m)
        d.qpos[0],d.qvel[0] = rng.uniform(.38,1.65),rng.uniform(-1,.35)
        prefix = [d.qpos[0]]
        for _ in range(11):
            mujoco.mj_step(m,d,nstep=10)
            prefix.append(d.qpos[0])
        noisy = np.array(prefix)+rng.normal(0,SIGMA,len(prefix))
        state = np.empty(mujoco.mj_stateSize(m,SPEC))
        mujoco.mj_getState(m,d,state,SPEC)
        chosen = range(3) if all_actions else [rng.integers(3)]
        for a in chosen:
            branch = mujoco.MjData(m)
            mujoco.mj_setState(m,branch,state,SPEC)
            branch.qvel[0] += ACTIONS[a]
            suffix = []
            for _ in range(12):
                mujoco.mj_step(m,branch,nstep=10)
                suffix.append(branch.qpos[0])
            inputs.append(noisy)
            actions.append(a)
            outputs.append(suffix)
    return np.asarray(inputs),np.asarray(actions),np.asarray(outputs)


def mdl(i,o):
    return nn.Sequential(nn.Linear(i,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,o))


def mixture(output):
    pi,mu,logsd = torch.split(output,[4,48,48],dim=1)
    return pi.log_softmax(1),mu.reshape(-1,4,12),logsd.reshape(-1,4,12).clamp(-4,2)


def train(context,y,method,seed):
    torch.manual_seed(seed)
    joint = method=="flow"
    net = mdl(28,12) if joint else mdl(15,100)
    opt = torch.optim.Adam(net.parameters(),lr=.0015)
    for _ in range(2000):
        idx = torch.randint(len(y),(192,))
        z,c = y[idx],context[idx]
        if joint:
            t = torch.rand(len(z),1)
            noise = torch.randn_like(z)
            x = (1-t)*noise+t*z
            err = net(torch.cat((x,c,t),1))-(z-noise)
            loss = err.square().mean()
        else:
            lp,mu,ls = mixture(net(c))
            loglike = -.5*((z[:,None]-mu)*(-ls).exp()).square()-ls
            loss = -(lp+loglike.sum(2)).logsumexp(1).mean()/12
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def stats(prob,truth):
    # Every row contains three physical potential outcomes from the same pre-action state.
    p = prob.reshape(-1,3)
    winner = p.argmax(1)
    best = p.max(1)
    earned = np.where(best[:,None]>SAFE,truth[np.arange(len(p)),winner,None],SAFE)
    return dict(value=float(earned.mean()),brier=float(np.square(p-truth).mean()),
                value_by_safe=earned.mean(0).tolist())


def event(y):
    return (y[...,REGION]>=.10)&(y[...,REGION]<=.23)


def event_bounds(p):
    return np.maximum(0,p.sum(1)-2),p.min(1)


@torch.no_grad()
def learned(net,method,ctx,mean,sd,seed):
    torch.manual_seed(seed)
    if method=="mdn":
        lp,mu,ls = mixture(net(ctx))
        w = lp.exp().numpy()
        center = mu.numpy()*sd+mean
        scale = ls.exp().numpy()*sd
        inside = (ndtr((.23-center)/scale)-ndtr((.10-center)/scale))[:,:,REGION]
        marginal = np.einsum("nc,nct->nt",w,inside)
        return (w*inside.prod(2)).sum(1),marginal.prod(1),*event_bounds(marginal)
    lists = [[],[],[],[]]
    for batch in ctx.split(32):
        c = batch.repeat_interleave(96,0)
        x = torch.randn(len(c),12)
        for j in range(16):
            t = torch.full((len(c),1),j/16)
            x += net(torch.cat((x,c,t),1))/16
        hits = event(x.reshape(len(batch),96,12).numpy()*sd+mean)
        marginal = hits.mean(1)
        for result,value in zip(lists,(hits.all(2).mean(1),marginal.prod(1),*event_bounds(marginal))):
            result.extend(value)
    return tuple(np.array(z) for z in lists)


def main(bounds_test=False):
    source = (.18,.4,.75)
    test_base = 94200 if bounds_test else 93200
    tests = {k:[dataset(v,s,64,True) for s in range(test_base,test_base+8)] for k,v in
             [("source",source),("new",(.12,.55,.95))]}
    rows = []
    for seed in range(93000,93003):
        x,a,y = dataset(source,seed,3072,False)
        xmean,xsd,mean,sd = x.mean(0),x.std(0),y.mean(0),y.std(0)
        def context(obs,actions):
            return torch.from_numpy(np.column_stack(((obs-xmean)/xsd,np.eye(3)[actions])).astype("float32"))
        ctx = context(x,a)
        target = torch.from_numpy(((y-mean)/sd).astype("float32"))
        heads = {m:train(ctx,target,m,seed) for m in ("flow","mdn")}
        for name,groups in tests.items():
            for i,(obs,act,future) in enumerate(groups):
                truth = event(future).all(1).reshape(-1,3)
                probs = {}
                for m,net in heads.items():
                    probs[m] = learned(net,m,context(obs,act),mean,sd,seed+i+100)
                probs["nearest32"] = tuple(np.empty(len(future)) for _ in range(4))
                for action in range(3):
                    ix = a==action
                    d = cdist((obs[act==action]-xmean)/xsd,(x[ix]-xmean)/xsd)
                    pairs = np.argpartition(d,31,axis=1)[:,:32]
                    hits = event(y[ix][pairs])
                    marginal = hits.mean(1)
                    for result,vals in zip(probs["nearest32"],(hits.all(2).mean(1),marginal.prod(1),*event_bounds(marginal))):
                        result[act==action] = vals
                for joint,_,lower,upper in probs.values():
                    assert np.all((joint>=lower-1e-6)&(joint<=upper+1e-6))
                rules = ("joint","independent","lower","upper") if bounds_test else ("joint","independent")
                rows.append(dict(seed=seed,test_seed=test_base+i,materials=name,
                    event_rates=truth.mean(0).tolist(),oracle_value=float(np.maximum(truth.max(1)[:,None],SAFE).mean()),
                    methods={m:{k:stats(p,truth) for k,p in zip(rules,pair)} for m,pair in probs.items()}))
    return dict(actions=ACTIONS.tolist(),safe_values=SAFE.tolist(),train_n=3072,test_states_per_material=512,
                train_seeds=[93000,93001,93002],flow_steps=16,flow_samples_per_action=96,results=rows)


if __name__ == "__main__":
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounds-test",action="store_true")
    bounds_test = ap.parse_args().bounds_test
    name = "bounds38.json" if bounds_test else "action38.json"
    (HERE/name).write_text(json.dumps(main(bounds_test),indent=2,allow_nan=False)+"\n",encoding="utf-8")
