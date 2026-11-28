"""One native RGB frame at inference, with separate scoring and no target-scale fitting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from depth_full_eval import load_sample
from eval_stereo_students import hammer_scenes
from native_direction_data import CKPT,Native,OUT,PREV,SEED,nearest,sha,write
from native_direction_fit import ARMS,direction,unpack,valid
from scale_surface_evaluate import frames

def weights():
    record=json.loads((OUT/"selection.json").read_text())
    path=OUT/"student.pt"
    if sha(path)!=record["student_sha256"] or sha(CKPT)!=record["ckpt"]:
        raise ValueError("Frozen heads or network mismatch")
    return torch.load(path,map_location="cpu",weights_only=True),record["student_sha256"]

def predict(reader,heads,models,rgb):
    base,low,x=reader.extract(rgb)
    lab=nearest(low.astype(np.float32),base.shape).astype(np.int32)
    area=np.bincount(lab.ravel(),minlength=len(x))
    hs=direction(x,area,heads,models)
    for arm in ARMS:
        hs[arm]*=heads["choice"][arm]
    return base,lab,hs

def dataframes(dataset):
    if dataset=="hammer":
        with ZipFile(PREV.parent/"HAMMER.zip") as arc:
            for group,names in sorted(hammer_scenes(arc).items()):
                for name in sorted(names):
                    rgb,z,ok,_=load_sample(arc,name,1)
                    yield name,group,rgb,z,ok
    else:
        yield from frames(dataset)

def score(base,labels,hs,k,z,ok):
    if base.shape!=z.shape or base.shape!=ok.shape:
        raise ValueError("Native prediction and target must match")
    ok=ok&valid(z)
    if not ok.any():
        return None
    lab=labels[ok]
    gt=z[ok].astype(float);b=base[ok].astype(float)
    q=k*b/gt;t=-np.log(q)
    absolute=lambda v:float(np.abs(v-1).mean())
    losses={"raw":absolute(b/gt),"fixed":absolute(q),
        "legacy":absolute((b+np.clip(b*(.8051849577771368-1),-.25,.25))/gt)}
    counts=np.bincount(lab,minlength=len(hs["joint"]))
    trend=np.bincount(lab,weights=np.where(q<1,q,np.where(q>1,-q,0)),minlength=len(counts))
    ties=np.bincount(lab,weights=np.where(q==1,q,0),minlength=len(counts))
    detail={}
    for arm,h in hs.items():
        c=h[lab]
        updated=q*np.exp(c)
        losses[arm]=absolute(updated)
        actual=np.bincount(lab,weights=np.abs(q-1)-np.abs(updated-1),minlength=len(counts))
        chosen=(np.abs(h)>1e-9)&(counts>0)
        correct=chosen & (np.sign(h)*trend>ties)
        avg=float(c.mean())
        scale=2*t.mean()*avg-avg**2
        total=np.mean(2*t*c-c*c)
        detail[arm]=dict(correct=int(correct.sum()),selected=int(chosen.sum()),regions=int((counts>0).sum()),
            lower_loss=int((chosen&(actual>0)).sum()),covered_pixels=int(counts[chosen].sum()),
            mean_log=avg,scale_loggain=float(scale),shape_loggain=float(total-scale))
    return dict(n=int(ok.sum()),size=list(base.shape),loss=losses,detail=detail)

def aggregate(rows,dataset):
    groups=defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    ids=sorted(groups)
    names=rows[0]["loss"]
    group_scores={g:{a:float(np.mean([r["loss"][a] for r in groups[g]])) for a in names} for g in ids}
    mean={a:float(np.mean([group_scores[g][a] for g in ids])) for a in names}
    ref=np.array([group_scores[g]["fixed"] for g in ids])
    cand=np.array([group_scores[g]["joint"] for g in ids])
    quant=None
    if dataset!="nyu":
        idx=np.random.default_rng(SEED).integers(0,len(ids),(10000,len(ids)))
        ratios=100*(1-cand[idx].mean(1)/ref[idx].mean(1))
        quant=np.quantile(ratios,[.025,.05,.975]).tolist()
    details={}
    for arm in ARMS:
        counters={k:sum(r["detail"][arm][k] for r in rows) for k in
                  ("correct","selected","regions","lower_loss","covered_pixels")}
        def group_avg(k):
            return float(np.mean([np.mean([r["detail"][arm][k] for r in groups[g]]) for g in ids]))
        details[arm]=dict(**counters,accuracy=counters["correct"]/counters["selected"] if counters["selected"] else None,
            selected_fraction=counters["selected"]/counters["regions"],
            avg_mask_log=group_avg("mean_log"),scale_loggain=group_avg("scale_loggain"),
            shape_loggain=group_avg("shape_loggain"))
    return dict(frames=len(rows),groups=len(ids),scores=mean,details=details,
                gain_pct=100*(1-mean["joint"]/mean["fixed"]),group_quantiles_pct=quant,
                group_scores=group_scores if dataset!="nyu" else None,improved=int((cand<ref).sum()),
                sizes=sorted(set(tuple(r["size"]) for r in rows)),pixels=sum(r["n"] for r in rows))

def main():
    heads,hashval=weights()
    reader=Native();models=unpack(heads)
    results={}
    output=OUT/"middlebury-native"
    output.mkdir(exist_ok=True)
    for dataset in ("middlebury","hammer","ibims","nyu"):
        rows=[];missing=[]
        for key,group,rgb,z,ok in dataframes(dataset):
            base,lab,hs=predict(reader,heads,models,rgb)
            if dataset=="middlebury":
                temp=output/(key+".part.npz")
                np.savez_compressed(temp,base=base,h=hs["joint"][lab].astype(np.float32),
                                    k=heads["k"],sha256=hashval)
                temp.replace(output/(key+".npz"))
            item=score(base,lab,hs,heads["k"],z,ok)
            if item is None:
                missing.append(key)
                continue
            rows.append(dict(key=key,group=group,**item))
            if len(rows)%200==0:
                print(dataset,len(rows),flush=True)
        results[dataset]=dict(**aggregate(rows,dataset),empty_mask=missing)
        write(OUT/(dataset+"-rows.json"),dict(rows=rows,student=hashval))
        write(OUT/"native-results.json",dict(datasets=results,student=hashval,official_check=reader.check))
        print(dataset,len(rows),results[dataset]["gain_pct"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.image and args.output:
        heads,_=weights()
        with Image.open(args.image) as image:
            b,l,h=predict(Native(),heads,unpack(heads),np.array(image.convert("RGB")))
        np.save(args.output,heads["k"]*b*np.exp(h["joint"][l]))
    elif not args.image and not args.output:
        main()
    else:
        parser.error("Specify both image and output")
