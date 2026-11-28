"""Post-test render-label diagnostic using actual changes in direct-light occlusion."""
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from depth_full_data import hypersim_intrinsics
from edge_motion43 import (OUT,RENDERS,Model,edge,grid,halo,infer,inverse,neighbors,resize,
                           save_json,sha,CKPT)


def score(rgb,a,b,z,valid,entity,sa,sb):
    sz=grid(rgb)
    ok=resize(valid.astype(np.uint8),sz,True)>0
    zz=np.log(np.where(ok,resize(z,sz,True),1))
    protect=halo(edge(zz,ok,np.log(1.05)),2)|halo(~ok,2)
    shadows=[edge(resize(s.astype(np.uint8),sz,True).astype(np.float32),ok,.5) for s in (sa,sb)]
    masks=[halo(shadows[k],2)&~halo(shadows[1-k],2)&~protect for k in (0,1)]
    base=np.log(.5*(a.astype(float)+b))
    low=resize(base.astype(np.float32),sz)
    other=[resize(np.log(v),sz)-low for v in (b,a)]
    rhs=np.zeros_like(low)
    for axis in (0,1):
        p,q=neighbors(axis,1,1)
        h=sum(np.diff(d,axis=axis)*m[p] for d,m in zip(other,masks))
        rhs[p]-=h
        rhs[q]+=h
    correction=resize(inverse(rhs),z.shape[::-1]).astype(float)
    true_log=np.log(np.where(valid,z,1.)).astype(float)
    terms=np.zeros(3)
    n=0
    for axis in (0,1):
        p,q=neighbors(axis,16,4)
        keep=valid[p]&valid[q]&(entity[p]==entity[q])
        r=(true_log[q]-true_log[p]-base[q]+base[p])[keep]
        u=(correction[q]-correction[p])[keep]
        terms += np.array([(r*r).sum(),(r*u).sum(),(u*u).sum()])
        n+=len(r)
    return dict(moments=(terms/n).tolist(),fraction=float((masks[0]|masks[1]).mean()))


def main():
    model=Model()
    rows=defaultdict(list)
    for row in json.loads((RENDERS/"scenes.json").read_text()):
        folder=RENDERS/"renders"/row["scene"]
        with np.load(folder/"passes.npz") as f:
            z,ent=f["z"],f["entity"]
            valid=f["valid"]&(z>=.25)&(z<=30)
            ss=[f[f"shadow{i}"] for i in range(3)]
        fx=hypersim_intrinsics(row["scene"],z.shape[1],z.shape[0])[0,0]
        for mat in ("matte","coat"):
            imgs=[np.array(Image.open(folder/f"{mat}-{i}.png")) for i in range(3)]
            depths=[infer(model,img,fx) for img in imgs]
            for i,j in ((0,1),(0,2),(1,2)):
                rows[row["split"]].append(dict(scene=row["scene"],material=mat,lights=[i,j],
                    **score(imgs[i],depths[i],depths[j],z,valid,ent,ss[i],ss[j])))
    # Each scene contributes exactly six pairs and is the only member of its family.
    moments={k:np.mean([r["moments"] for r in rs],axis=0) for k,rs in rows.items()}
    _,a,c=moments["train"]
    lam=float(np.clip(a/c,0,1))
    m,a,c=moments["val"]
    save_json(OUT/"label-diagnostic.json",dict(strength=lam,val_half=float(m),
             val_corrected=float(m-2*lam*a+lam**2*c),val_gain=float(100*(2*lam*a-lam**2*c)/m),
             val_fraction=float(np.mean([r["fraction"] for r in rows["val"]])),rows=rows,model=sha(CKPT)))


if __name__=="__main__":
    main()
