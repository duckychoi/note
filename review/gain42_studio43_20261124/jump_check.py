"""Exploratory boundary contrast, to distinguish good absolute scale from resolved edges."""
import json
from collections import defaultdict

import cv2
import numpy as np
import torch

from experiment import SEEDS
from footage import ARCHIVE, DEV, TEST, read
from scenes import CACHE, HERE, DepthNet


def pairs(z,band,inside):
    valid=(z>=.2)&(z<=5)&band
    indices=np.arange(z.size).reshape(z.shape)
    left,right=[],[]
    for axis in (0,1):
        a=[slice(None)]*2
        b=[slice(None)]*2
        a[axis],b[axis]=slice(2,None),slice(None,-2)
        a,b=tuple(a),tuple(b)
        ok=valid[a]&valid[b]&(inside[a]!=inside[b])&(np.abs(z[a]-z[b])>=.05)
        left.extend(indices[a][ok])
        right.extend(indices[b][ok])
    return np.asarray(left,int),np.asarray(right,int)


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    extra=json.loads((HERE/"extra_manifest.json").read_text())["scenes"]
    out={}
    for label,scenes,archive,n in [("basic",TEST,ARCHIVE,12),("extra",extra,CACHE/"supplement.zip",8)]:
        dataset=read(scenes,archive,n)
        keep=[]
        for x,z,band,inside,scene,frame,_ in dataset:
            a,b=pairs(z,band,inside)
            if len(a)>=16:
                truth=z.ravel()[a]-z.ravel()[b]
                keep.append((x,truth,a,b,scene,frame))
        if not keep:
            out[label]={"frames":0}
            continue
        inputs=torch.stack([r[0] for r in keep]).cuda()
        summary={}
        for name in ("constant","zero/replay","zero/gain","few/replay","few/gain"):
            by_seed=[]
            for seed in SEEDS if name!="constant" else [0]:
                if name!="constant":
                    model=DepthNet(False).cuda().eval()
                    model.load_state_dict(torch.load(CACHE/"checkpoints"/f"{seed}_{name.replace('/','_')}.pt",weights_only=True))
                    with torch.no_grad():
                        pred=torch.cat([model(b)[:,0].exp() for b in inputs.split(8)]).cpu().numpy()
                groups=defaultdict(list)
                for i,(_,truth,a,b,s,_) in enumerate(keep):
                    change=np.zeros_like(truth) if name=="constant" else pred[i].ravel()[a]-pred[i].ravel()[b]
                    sign=(np.sign(change)==np.sign(truth)).astype(float)+.5*(change==0)
                    groups[s].append([float(1000*np.abs(change-truth).mean()),float(sign.mean())])
                by_seed.append({s:np.mean(scores,axis=0).tolist() for s,scores in groups.items()})
            scores=np.array([list(row.values()) for row in by_seed]).mean(axis=(0,1))
            summary[name]={"mae_mm":scores[0],"sign_accuracy":scores[1],"by_seed_scene":by_seed}
        out[label]=dict(frames=len(keep),scenes=sorted({r[4] for r in keep}),
                        pair_count=sum(len(r[1]) for r in keep),results=summary)
    (HERE/"jump_check.json").write_text(json.dumps(out,indent=1),encoding="utf8")


if __name__=="__main__":
    main()
