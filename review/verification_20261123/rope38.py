"""Separate history-length extrapolation from rollout length in trained temporal attention."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def sequences(rng,batch,length):
    phase = rng.uniform(-np.pi,np.pi,(batch,1))
    omega = rng.uniform(.22,.50,(batch,1))
    action = rng.choice((-1.,0.,1.),(batch,length),p=(.2,.6,.2))
    prev = np.column_stack((np.zeros(batch),np.cumsum(action[:,:-1],axis=1)))
    true = np.cos(phase+omega*np.arange(length)+.08*prev)
    obs = true + .03*rng.standard_normal(true.shape)
    return [torch.tensor(x,dtype=torch.float32) for x in (obs,action,true)]


class Predictor(nn.Module):
    def __init__(self,method):
        super().__init__()
        self.method = method
        self.embed = nn.Linear(2,32)
        self.qkv = nn.Linear(32,96)
        self.out = nn.Linear(32,32)
        self.ff = nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        self.ln = nn.LayerNorm(32)
        self.read = nn.Linear(32,1)
        if method=="learned":
            self.position = nn.Parameter(torch.randn(32,32)*.02)
        self.register_buffer("freq",10000**(-torch.arange(0,8,2)/8))

    def rotate(self,x):
        angles = torch.arange(x.shape[2])[:,None]*self.freq
        a,b = x[...,::2],x[...,1::2]
        return torch.stack((a*angles.cos()-b*angles.sin(),b*angles.cos()+a*angles.sin()),dim=-1).flatten(-2)

    def forward(self,obs,act):
        b,t = obs.shape
        x = self.embed(torch.stack((obs,act),dim=-1))
        if self.method=="learned":
            x = x+self.position[:t]
        q,k,v = self.qkv(self.ln(x)).reshape(b,t,3,4,8).permute(2,0,3,1,4).unbind(0)
        if self.method=="rope":
            q,k = self.rotate(q),self.rotate(k)
        att = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x = x+self.out(att.transpose(1,2).reshape(b,t,32))
        return obs+self.read(x+self.ff(x)).squeeze(-1)


def trained(method,seed):
    torch.manual_seed(seed)
    net = Predictor(method)
    rng = np.random.default_rng(seed+101)
    opt = torch.optim.AdamW(net.parameters(),lr=.001)
    for _ in range(1200):
        obs,act,_ = sequences(rng,64,9)
        pred = net(obs[:,:8],act[:,:8])
        loss = (pred[:,3:]-obs[:,4:9]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.inference_mode()
def evaluate(net):
    obs,act,true = sequences(np.random.default_rng(105200),512,65)
    def error(x,y):
        return float((x-y).square().mean().sqrt())
    fixed = net(obs[:,24:32],act[:,24:32])[:,-1]
    longer = net(obs[:,:32],act[:,:32])[:,-1]
    past = obs[:,24:32]
    output = {}
    for h in range(1,33):
        step = net(past,act[:,23+h:31+h])[:,-1]
        past = torch.cat((past[:,1:],step[:,None]),1)
        if h in (1,8,32):
            output[h] = error(step,true[:,31+h])
    return dict(one_step_last8=error(fixed,true[:,32]),one_step_all32=error(longer,true[:,32]),
                fixed_context_rollout=output,parameters=sum(p.numel() for p in net.parameters()))


if __name__=="__main__":
    torch.set_num_threads(2)
    rows = {s:{m:evaluate(trained(m,s)) for m in ("none","learned","rope")} for s in range(105000,105003)}
    (HERE/"rope38.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
