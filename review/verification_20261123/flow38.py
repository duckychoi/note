"""Train one-step and joint conditional neural flows on the matched-pair worlds."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from planning38 import all_starts, utility_regret

HERE = Path(__file__).resolve().parent
NFE = 32


def vector_field(dims,conditions):
    return nn.Sequential(nn.Linear(dims+conditions+1,64),nn.SiLU(),nn.Linear(64,64),
                         nn.SiLU(),nn.Linear(64,dims))


def dataset(seed):
    rng = np.random.default_rng(seed)
    groups = []
    for seq in ("+++---","+-----+++++-"):
        patterns = all_starts(seq)
        signs = patterns[rng.integers(len(patterns),size=1024)]
        groups.append(signs*np.abs(rng.normal(size=signs.shape)))
    return torch.from_numpy(np.concatenate(groups)).float(),torch.arange(2).repeat_interleave(1024).float()[:,None]


def train(x,world,seed,joint):
    torch.manual_seed(seed)
    net = vector_field(4,1) if joint else vector_field(1,2)
    optimizer = torch.optim.Adam(net.parameters(), lr=.002)
    for _ in range(1800):
        idx = torch.randint(len(x),(256,))
        context = world[idx]
        if joint:
            target = x[idx]
        else:
            t_idx = torch.randint(3,(len(idx),))
            target = x[idx,t_idx+1,None]
            context = torch.cat((context,x[idx,t_idx,None]),1)
        noise = torch.randn_like(target)
        t = torch.rand(len(idx),1)
        mid = (1-t)*noise+t*target
        loss = (net(torch.cat((mid,context,t),1))-(target-noise)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net.eval()


@torch.inference_mode()
def sample(net,context,dim):
    y = torch.randn(len(context),dim)
    for i in range(NFE):
        t = torch.full((len(y),1),i/NFE)
        y = y + net(torch.cat((y,context,t),1))/NFE
    return y


def run(seed):
    x,world = dataset(seed)
    rows = []
    for method in ("one_step","joint_four"):
        joint = method=="joint_four"
        net = train(x,world,seed+12000,joint)
        for w in (0,1):
            torch.manual_seed(seed+13000+w)
            context = torch.full((4096,1),float(w))
            if joint:
                path = sample(net,context,4)
            else:
                steps = [torch.randn(len(context),1)]  # exact initial marginal, shared by both worlds
                for _ in range(3):
                    steps.append(sample(net,torch.cat((context,steps[-1]),1),1))
                path = torch.cat(steps,1)
            sign = path.numpy()>0
            event_p = float(np.mean((sign[:,1:]==sign[:,:-1]).all(1)))
            rows.append(dict(method=method,world=w,event_p=event_p,regret=utility_regret(event_p,w/3,.2),
                        end_sign_match=float(np.mean(sign[:,0]==sign[:,-1])),
                        stay=float(np.mean(sign[:,1:]==sign[:,:-1])),
                        parameters=sum(p.numel() for p in net.parameters()),nfe=NFE*(1 if joint else 3)))
    return rows


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = {seed:run(seed) for seed in range(51200,51205)}
    (HERE/"flow38.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
