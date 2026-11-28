"""Exploratory joint depth-and-jump L1 gain and independent value for stereo pairs."""
import copy
import hashlib
import json

import cv2
import numpy as np
import torch

from fit import bank, predict, DepthNet, terms, CKPT, SEEDS, med
from acquire import HERE

ARMS=("pair_uniform","pair_utility","pair_truth")
OFFSETS=((0,4),(4,0))


def ends(arr,dy,dx):
    h,w=arr.shape[-2:]
    return arr[...,:h-dy,:w-dx],arr[...,dy:,dx:]


def edges(data,prior):
    for dy,dx in OFFSETS:
        a,b=ends(prior,dy,dx)
        s,t=ends(data["stereo"],dy,dx)
        v,w=ends(data["ok"],dy,dx)
        l,r=ends(data["spread"],dy,dx)
        measured=t-s
        expected=b-a
        valid=v&w&(np.abs(measured)>=.05)
        group=4*np.digitize(np.abs(measured-expected),[.2,.6])+2*(np.sign(measured)!=np.sign(expected))+(np.maximum(l,r)>1.5)
        za,zb=ends(data["truth"],dy,dx)
        yield (a,b,s,t,za,zb),group,valid


def gather(net,data):
    prior=predict(net,data["x"])
    obs=list(edges(data,prior))
    values=np.concatenate([np.stack([r[ok] for r in rows]) for rows,_,ok in obs],axis=1)
    return values,np.concatenate([g[ok] for _,g,ok in obs]),sum(ok.size for _,_,ok in obs)


def optimal(vals):
    a,b,s,t,za,zb=vals
    d1,d2=s-a,t-b
    residual=np.concatenate((za-a,zb-b,(zb-za)-(b-a)))
    step=np.concatenate((d1,d2,d2-d1))
    weights=np.concatenate((.5*np.abs(d1),.5*np.abs(d2),np.abs(d2-d1)))
    ok=weights>0
    return float(np.clip(med(residual[ok]/step[ok],weights[ok]),0,1)) if ok.any() else 0.


def risk(a,b,za,zb):
    return .5*(abs(a-za)+abs(b-zb))+abs((b-a)-(zb-za))


def risks(vals,k):
    a,b,s,t,za,zb=vals
    return risk(a+k*(s-a),b+k*(t-b),za,zb)


def calibrate(teacher,seed):
    vals,g,_=gather(teacher,bank(seed,1,32))
    global_k=optimal(vals)
    gain=np.full(12,global_k,np.float32)
    for i in range(12):
        mask=g==i
        if mask.sum()>=128:
            gain[i]=optimal(vals[:,mask])
    vals,g,_=gather(teacher,bank(seed,2,32))
    delta=risks(vals,0)-risks(vals,gain[g])
    counts=np.bincount(g,minlength=12)
    u=(np.bincount(g,weights=delta,minlength=12)+128*delta.mean())/(counts+128)
    u=u.astype(np.float32)
    positive=u[g][u[g]>0]
    cap=float(np.quantile(positive,.9)) if len(positive) else 1.
    vals,g,total=gather(teacher,bank(seed,4,16))
    before,after=risks(vals,0),risks(vals,gain[g])
    partition=np.array_split(np.argsort(u[g],kind="stable"),4)
    audit=dict(prior_mm=float(1000*before.mean()),corrected_mm=float(1000*after.mean()),
         fixed_mm=float(1000*risks(vals,global_k).mean()),fraction=len(g)/total,
         quartiles_true_mm=[float((before[ii]-after[ii]).mean()*1000) for ii in partition])
    return gain,u,cap,audit


def train_pair(teacher,data,seed,gain,u,cap):
    prior=predict(teacher,data["x"])
    targets={arm:[] for arm in ARMS}
    def gpu(a):
        return torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32)).cuda()
    for i,(vals,g,valid) in enumerate(edges(data,prior)):
        a,b,s,t,za,zb=vals
        k=gain[g]
        ca,cb=gpu(a+k*(s-a)),gpu(b+k*(t-b))
        weight=gpu(np.where(valid,np.clip(u[g]/cap,0,1),0))
        targets["pair_utility"].append((ca,cb,weight))
        targets["pair_uniform"].append((ca,cb,gpu(valid)))
        targets["pair_truth"].append((gpu(za),gpu(zb),weight))
    logz=gpu(np.log(data["truth"]))
    for arm in ARMS:
        net=copy.deepcopy(teacher).train()
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001)
        rng=np.random.default_rng(seed+81)
        indices=[]
        for step in range(192):
            idx=rng.integers((step//48+1)*len(logz)//4,size=4)
            indices.extend(idx)
            y=net(data["x"][idx])[:,0]
            loss=terms(y-logz[idx])
            depth=y.exp()
            num=den=0
            for (dy,dx),(ca,cb,w) in zip(OFFSETS,targets[arm]):
                p,q=ends(depth,dy,dx)
                weight=w[idx]
                num=num+(weight*risk(p,q,ca[idx],cb[idx])).flatten(1).sum(1)
                den=den+weight.flatten(1).sum(1)
            loss=loss+.35*(num/den.clamp_min(1e-6)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)
        checksum=hashlib.sha256(np.int32(indices).tobytes()).hexdigest()
        torch.save(net.state_dict(),CKPT/f"{seed}_{arm}.pt")
        yield arm,net.eval(),checksum


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS:
        ledger=json.loads((HERE/f"seed{seed}.json").read_text())
        teacher=DepthNet(False).cuda()
        teacher.load_state_dict(torch.load(CKPT/f"{seed}_prior_net.pt",weights_only=True))
        teacher.eval()
        gain,u,cap,audit=calibrate(teacher,seed)
        scale=bank(seed,3,16)
        full=bank(seed,5,128)
        z=scale["truth"][:,::4,::4].ravel()
        scales={}
        for arm,net,checksum in train_pair(teacher,full,seed,gain,u,cap):
            assert checksum==ledger["batch_hashes"]["replay"]
            p=predict(net,scale["x"])[:,::4,::4].ravel()
            scales[arm]=med(z/p,p)
        output=dict(seed=seed,gain=gain.tolist(),utility=u.tolist(),cap=cap,scales=scales,audit=audit,batch_hash=checksum)
        (HERE/f"pair{seed}.json").write_text(json.dumps(output,indent=1),encoding="utf8")
        print("trained pair arms",seed,audit,flush=True)


if __name__=="__main__":
    main()
