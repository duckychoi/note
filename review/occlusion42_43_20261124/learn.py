"""Matched-budget order supervision on a single dense metric depth output."""
import copy
import hashlib
import json
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT,STORE,SEEDS,ARMS,save
from junctions import render
from fit import DepthNet,bank,train,terms,med,predict,observe,tensor_input
from paired import gather,optimal,risks,edges,ends,risk,OFFSETS


def gpu(a):
    return torch.as_tensor(np.ascontiguousarray(a),device="cuda")


def fit_tables(teacher,seed):
    vals,g,_=gather(teacher,bank(seed,1,32))
    gain=np.full(12,optimal(vals),np.float32)
    for j in range(12):
        if (g==j).sum()>=128:
            gain[j]=optimal(vals[:,g==j])
    vals,g,_=gather(teacher,bank(seed,2,32))
    delta=risks(vals,0)-risks(vals,gain[g])
    u=((np.bincount(g,weights=delta,minlength=12)+128*delta.mean())/(np.bincount(g,minlength=12)+128)).astype(np.float32)
    pos=u[g][u[g]>0]
    return gain,u,float(np.quantile(pos,.9)) if len(pos) else 1.


def mixed(legacy,seed,lines):
    data={k:v.clone() if torch.is_tensor(v) else v.copy() for k,v in legacy.items()}
    queries=np.zeros((len(data["x"]),96,3),np.int32)
    for n in range(64):
        frames,fx,B=render(seed,21,n//2,n%2,lines)
        matches=observe(frames[0][0],frames[1][0],fx,B,0,48)
        for side,((im,z,qs),ob) in enumerate(zip(frames,matches)):
            i=2*(2*n+1)+side
            data["x"][i]=tensor_input(im,fx).cuda()
            data["truth"][i]=z
            for k,value in zip(("stereo","ok","grad","spread"),ob):
                data[k][i]=value
            queries[i,:len(qs),:2]=qs[:,:2]
            queries[i,:len(qs),2]=1
    return data,queries


def targets(prior,data,gain,u,cap):
    out=[]
    for vals,g,valid in edges(data,prior):
        a,b,s,t,_,_=vals
        k=gain[g]
        weight=np.where(valid,np.clip(u[g]/cap,0,1),0).astype(np.float32)
        out.append((gpu(a+k*(s-a)),gpu(b+k*(t-b)),gpu(weight)))
    return out


def fit_student(teacher,data,qs,order,seed,tg):
    net=copy.deepcopy(teacher).train()
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001)
    logz=gpu(np.log(data["truth"]))
    rng=np.random.default_rng(seed+81)
    qs=gpu(qs.astype(np.int64)) if order else None
    ids=[]
    for step in range(192):
        sel=rng.integers((step//48+1)*len(logz)//4,size=4)
        ids.extend(sel)
        p=net(data["x"][sel])[:,0]
        loss=terms(p-logz[sel])
        depth=p.exp()
        num=den=0
        for (dy,dx),(ca,cb,w) in zip(OFFSETS,tg):
            pa,pb=ends(depth,dy,dx)
            weight=w[sel]
            num=num+(weight*risk(pa,pb,ca[sel],cb[sel])).flatten(1).sum(1)
            den=den+weight.flatten(1).sum(1)
        loss=loss+.35*(num/den.clamp_min(1e-6)).mean()
        if order:
            q=qs[sel]
            def differences(arr):
                flat=arr.flatten(1)
                return flat.gather(1,q[:,:,1])-flat.gather(1,q[:,:,0])
            signed=differences(p)*differences(logz[sel]).sign()
            loss=loss+.15*(F.softplus(-signed/.15)*q[:,:,2]).sum()/q[:,:,2].sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    return net.eval(),hashlib.sha256(np.int32(ids).tobytes()).hexdigest()


def audit_set(seed):
    rows=[]
    for i in range(8):
        states=[]
        for state in (0,1):
            fr,f,_=render(seed,22,i,state,True)
            im,z,q=fr[0]
            q=q[q[:,2]==1,:2]
            states.append((tensor_input(im,f),z.ravel()[q[:,1]]-z.ravel()[q[:,0]],q,im))
        x0,t0,q0,im0=states[0]
        x1,t1,q1,im1=states[1]
        assert np.array_equal(q0,q1) and np.all(t0*t1<0)
        rows.append((torch.stack((x0,x1)).cuda(),np.stack((t0,t1)),q0))
        if i==0 and seed==SEEDS[0]:
            panel=cv2.cvtColor(np.hstack((im0,im1)),cv2.COLOR_RGB2BGR)
            cv2.imencode(".png",cv2.resize(panel,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))[1].tofile(ROOT/"synthetic_reversal.png")
    return rows


def audit_model(net,audit):
    states=[]
    for x,truth,q in audit:
        p=predict(net,x).reshape(2,-1)
        signed=(p[:,q[:,1]]-p[:,q[:,0]])*truth
        states.append(signed)
    signed=np.concatenate(states,axis=1)
    return dict(order=float(((signed>0)+.5*(signed==0)).mean()),both=float((signed>0).all(0).mean()),
                flipped=float((signed[0]*signed[1]>0).mean()),n=signed.shape[1],ties=float((signed==0).mean()))


def run():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    ckpt=STORE/"models"
    ckpt.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        teacher=DepthNet(False).cuda()
        train(teacher,bank(seed,0,48),80,seed,"warm")
        teacher.eval()
        gain,u,cap=fit_tables(teacher,seed)
        scale=bank(seed,3,16)
        zcal=scale["truth"][:,::4,::4].ravel()
        audit=audit_set(seed)
        old=bank(seed,5,128)
        ledger=dict(seed=seed,constant=float(np.median(zcal)),scales={},audit={},batch={},rank_queries={})
        for typ in ("legacy","plain","lines"):
            data,q=(old,None) if typ=="legacy" else mixed(old,seed,typ=="lines")
            tg=targets(predict(teacher,data["x"]),data,gain,u,cap)
            for arm in [a for a in ARMS if a.startswith(typ)]:
                model,h=fit_student(teacher,data,q,arm.endswith("order"),seed,tg)
                ledger["batch"][arm]=h
                ledger["audit"][arm]=audit_model(model,audit)
                ledger["rank_queries"][arm]=int(q[:,:,2].sum()) if arm.endswith("order") else 0
                p=predict(model,scale["x"])[:,::4,::4].ravel()
                ledger["scales"][arm]=med(zcal/p,p)
                torch.save(model.state_dict(),ckpt/f"{seed}_{arm}.pt")
            del tg,data
        assert len(set(ledger["batch"].values()))==1
        save(f"train_{seed}.json",ledger)
        print("trained",seed,ledger["audit"]["lines_order"],flush=True)


if __name__=="__main__":
    run()
