"""Distil epipolar disparity intervals to an RGB surface head; same-grid SGBM control."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from native_direction_data import OUT,SEED,cache_path,sha,write
from native_direction_fit import ARMS,LIMIT,direction,load,network,region_target,unpack,valid
from native_direction_eval import aggregate,dataframes,predict,score
from native_direction_data import Native

def items(inv,tap,split):
    for row in inv["rows"]:
        if row[0]!=split:
            continue
        data=load(OUT/inv["folder"],row)
        with np.load(cache_path(OUT/tap["folder"],row)) as tr:
            iy,ix=tr["iy"],tr["ix"]
            support=(iy,ix)
            samples=dict(tap=(tr["tap1"],tr["tap2"]),sgbm=(data["d1"][support],data["d2"][support]))
        yield row[1],data,support,samples

def calibration(inv,tap):
    residuals={name:defaultdict(list) for name in ("tap","sgbm")}
    for scene,d,p,samples in items(inv,tap,"cal"):
        z=d["z"][p]
        actual=d["A"][p]/np.maximum(z,1e-9)
        for teacher,(one,two) in samples.items():
            ok=valid(z)&np.isfinite(one)&np.isfinite(two)
            if ok.any():
                r=np.maximum(np.minimum(one,two)-actual,actual-np.maximum(one,two))
                residuals[teacher][scene].append(np.maximum(0,r[ok]))
    widths={};groups={}
    for name,scenes in residuals.items():
        qs=sorted(float(np.quantile(np.concatenate(rs),.9)) for rs in scenes.values())
        widths[name]=qs[min(len(qs),math.ceil(.9*(len(qs)+1)))-1]
        groups[name]=len(qs)
    return widths,groups

def observations(inv,tap,k,widths):
    raw=[]
    stats={key:Counter() for key in widths}
    counts=Counter()
    for scene,d,p,pairs in items(inv,tap,"fit"):
        truth=d["z"];ok=valid(truth)
        if not ok.any():
            continue
        x=d["X"];lab=d["lab"];base=d["base"]
        t=np.log(np.where(ok,truth,1)/(k*base))
        yd,wd=region_target(lab,ok,t,len(x))
        teachers={}
        for name,(one,two) in pairs.items():
            dl=np.minimum(one,two)-widths[name]
            du=np.maximum(one,two)+widths[name]
            d0=d["A"][p]/(k*base[p])
            matched=np.isfinite(du)&(d0>0)&(du>0)
            lo=np.log(np.maximum(d0,1e-10)/np.maximum(du,1e-10))
            hi=np.where(dl>0,np.log(np.maximum(d0,1e-10)/np.maximum(dl,1e-10)),np.inf)
            sel=matched&((lo>0)|(hi<0))
            s=stats[name]
            s["queries"]+=sel.size;s["matched"]+=int(matched.sum());s["directed"]+=int(sel.sum())
            eligible=sel&ok[p];s["directed_gt"]+=int(eligible.sum())
            s["correct"]+=int((eligible&(((lo>0)&(t[p]>0))|((hi<0)&(t[p]<0)))).sum())
            s["both_gt"]+=int((matched&ok[p]).sum())
            s["interval_hit"]+=int((matched&ok[p]&(t[p]>=lo)&(t[p]<=hi)).sum())
            teachers[name]=(lab[p][sel],lo[sel].clip(-LIMIT,LIMIT),hi[sel].clip(-LIMIT,LIMIT),
                            np.full(sel.sum(),1/sel.size))
        counts[scene]+=1
        raw.append((scene,x,yd,wd/wd.sum(),teachers))
    X=[];Y=[];W=[];track={k:[] for k in widths};offset=0
    for scene,x,y,w,teachers in raw:
        X.append(x);Y.append(y);W.append(w/counts[scene])
        for name,(idx,lo,hi,tw) in teachers.items():
            track[name].append((idx.astype(np.int64)+offset,lo,hi,tw/counts[scene]))
        offset+=len(x)
    tensors=lambda v:torch.from_numpy(np.concatenate(v).astype(np.float32))
    traj={}
    for name,v in track.items():
        indices,low,high,weights=zip(*v)
        traj[name]=(torch.from_numpy(np.concatenate(indices).astype(np.int64)),tensors(low),tensors(high),tensors(weights))
    return tensors(X),tensors(Y),tensors(W),traj,{k:dict(v) for k,v in stats.items()}

def optimize(X,Y,D,tracks):
    mu=(X*D[:,None]).sum(0)/D.sum()
    sd=(((X-mu)**2*D[:,None]).sum(0)/D.sum()).sqrt().clamp_min(.001)
    X=((X-mu)/sd).clamp(-8,8);D=D/D.mean()
    states={}
    for arm,teacher in (("dense",None),("stereo","sgbm"),("joint","tap")):
        torch.manual_seed(SEED)
        model=network(X.shape[1])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        rng=torch.Generator().manual_seed(SEED)
        tq=torch.Generator().manual_seed(SEED+1)
        if teacher:
            idx,lo,hi,tw=tracks[teacher]
            if len(idx)==0:
                raise ValueError("No identified training queries")
            tw=tw/tw.mean()
        for epoch in range(60):
            for ids in torch.randperm(len(X),generator=rng).split(1024):
                pred=LIMIT*model(X[ids]).flatten().tanh()
                loss=(D[ids]*(pred-Y[ids])**2).mean()
                if teacher:
                    query=torch.randint(len(idx),(len(ids),),generator=tq)
                    ph=LIMIT*model(X[idx[query]]).flatten().tanh()
                    violation=(lo[query]-ph).relu().square()+(ph-hi[query]).relu().square()
                    loss=.5*(loss+(tw[query]*violation).mean())
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        states[arm]=model.state_dict()
    return dict(mu=mu,sd=sd,states=states)

def fit():
    torch.set_num_threads(2)
    inv=json.loads((OUT/"inventory.json").read_text());tap=json.loads((OUT/"tap-inventory.json").read_text())
    k=json.loads((OUT/"fixed-scale.json").read_text())["k"]
    w,groups=calibration(inv,tap)
    X,Y,D,obs,stats=observations(inv,tap,k,w)
    print("Matched native teachers",w,stats,flush=True)
    h=optimize(X,Y,D,obs);models=unpack(h)
    amps=np.array([0.,.25,.5,1.])
    dev={a:defaultdict(list) for a in ARMS}
    for row in inv["rows"]:
        if row[0]!="val":
            continue
        d=load(OUT/inv["folder"],row)
        ok=valid(d["z"])
        if not ok.any():
            continue
        q=k*d["base"][ok]/d["z"][ok]
        for arm,v in direction(d["X"],d["area"],h,models).items():
            c=v[d["lab"][ok]]
            dev[arm][row[1]].append([np.abs(q*np.exp(a*c)-1).mean() for a in amps])
    scores={arm:np.mean([np.mean(rs,axis=0) for rs in g.values()],axis=0).tolist() for arm,g in dev.items()}
    choice={a:float(amps[np.argmin(v)]) for a,v in scores.items()}
    h.update(k=k,choice=choice,ckpt=inv["ckpt"])
    path=OUT/"tap-students.pt";temp=path.with_suffix(".partial");torch.save(h,temp);temp.replace(path)
    checksum=sha(path)
    write(OUT/"tap-students-selection.json",dict(k=k,choice=choice,dev=scores,widths=w,cal_scenes=groups,
          teacher_counts=stats,arms=dict(dense="dense GT",stereo="GT+SGBM interval",joint="GT+TAP interval"),
          tap_checkpoint=tap["checkpoint"],student=checksum,code=sha(Path(__file__))))
    return h,checksum

def diagnostic(base,labels,k,z,mask):
    # Real labels supply a feasible direction only for this explicitly marked oracle.
    mask=mask&valid(z)
    area=np.bincount(labels.ravel(),minlength=int(labels.max())+1)
    ids=labels[mask]
    q=k*base[mask].astype(float)/z[mask]
    active=np.bincount(ids,minlength=len(area))>0
    gradient=np.bincount(ids,weights=np.sign(1-q)*q,minlength=len(area))/np.maximum(area,1)
    gradient[active]-=np.average(gradient[active],weights=area[active])
    gradient[~active]=0
    gradient/=max(np.abs(gradient).max(),1e-12)
    if abs(area@gradient)>1e-8*area.sum():
        raise ValueError("Diagnostic altered global scale")
    offsets=gradient[ids]
    return min(float(np.mean(abs(q*np.exp(amp*offsets)-1))) for amp in (0,.025,.05,.0875,.175,.35))


def evaluate(h,checksum):
    reader=Native();models=unpack(h)
    result={}
    for name in ("hammer","ibims","middlebury","nyu"):
        rows=[]
        old=json.loads((OUT/(name+"-rows.json")).read_text())["rows"]
        baseline={r["key"]:r for r in old}
        for key,group,rgb,z,mask in dataframes(name):
            base,lab,hs=predict(reader,h,models,rgb)
            stats=score(base,lab,hs,h["k"],z,mask)
            if stats:
                stats["loss"]["gt_direction_diagnostic"]=diagnostic(base,lab,h["k"],z,mask)
                prior=baseline[key]
                if (prior["group"]!=group or prior["n"]!=stats["n"] or prior["size"]!=stats["size"]
                    or not np.isclose(stats["loss"]["fixed"],prior["loss"]["fixed"],rtol=1e-7,atol=1e-8)):
                    raise ValueError("Different fixed-scale depth/model/valid mask")
                rows.append(dict(key=key,group=group,**stats))
        if {r["key"] for r in rows}!=set(baseline):
            raise ValueError("Evaluation frame sets differ")
        result[name]=aggregate(rows,name)
        write(OUT/("tap-"+name+"-rows.json"),dict(student=checksum,rows=rows))
        write(OUT/"tap-native-results.json",dict(student=checksum,datasets=result))
        print(name,result[name]["gain_pct"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.image and args.output:
        path=OUT/"tap-students.pt"
        meta=json.loads((OUT/"tap-students-selection.json").read_text())
        if sha(path)!=meta["student"]:
            raise ValueError("Student weights changed")
        heads=torch.load(path,map_location="cpu",weights_only=True)
        from native_direction_data import CKPT
        if sha(CKPT)!=heads["ckpt"]:
            raise ValueError("Depth checkpoint changed")
        with Image.open(args.image) as image:
            base,lab,hs=predict(Native(),heads,unpack(heads),np.array(image.convert("RGB")))
        np.save(args.output,heads["k"]*base.astype(float)*np.exp(hs["joint"][lab]))
    elif not args.image and not args.output:
        heads,checksum=fit()
        evaluate(heads,checksum)
    else:
        parser.error("Specify both image and output")
