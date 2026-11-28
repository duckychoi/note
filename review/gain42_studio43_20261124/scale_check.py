"""Post-hoc: use the exact same real points as a metric scale, plus a constant-depth control."""
from collections import defaultdict
import json

import cv2
import numpy as np
import torch

from experiment import SEEDS
from footage import ARCHIVE, DEV, TEST, read
from scenes import CACHE, HERE, DepthNet


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    anchors=read(DEV)
    ax=torch.stack([r[0] for r in anchors]).cuda()
    constant=float(np.median(np.exp(np.concatenate([r[3] for r in anchors]))))
    extra=json.loads((HERE/"extra_manifest.json").read_text())["scenes"]
    out={"post_hoc":True,"constant_m":constant,"valid_points":sum(len(r[1]) for r in anchors),"sets":{}}
    for name,scenes,archive,n in [("main",TEST,ARCHIVE,12),("extra",extra,CACHE/"supplement.zip",8)]:
        rows=read(scenes,archive,n)
        x=torch.stack([r[0] for r in rows]).cuda()
        values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for arm in ("constant","uniform","source","replay","gain"):
            for seed in SEEDS if arm!="constant" else [0]:
                if arm=="constant":
                    estimates=np.full((len(rows),*rows[0][1].shape),constant,np.float32)
                else:
                    model=DepthNet(False).cuda().eval()
                    model.load_state_dict(torch.load(CACHE/"checkpoints"/f"{seed}_zero_{arm}.pt",weights_only=True))
                    with torch.no_grad():
                        calibration=model(ax)[:,0].cpu().numpy()
                        offsets=np.concatenate([r[3]-calibration[i,r[1],r[2]] for i,r in enumerate(anchors)])
                        shift=float(np.median(offsets))
                        estimates=torch.cat([(model(b)[:,0]+shift).exp() for b in x.split(8)]).cpu().numpy()
                frame_groups=defaultdict(lambda:defaultdict(list))
                for pred,(_,depth,band,_,scene,_,_) in zip(estimates,rows):
                    valid=(depth>=.2)&(depth<=5.)
                    err=np.abs(pred-depth)*1000
                    for metric,mask in [("whole",valid),("thin",valid&band)]:
                        if mask.sum()>=(16 if metric=="thin" else 1):
                            frame_groups[metric][scene].append(float(err[mask].mean()))
                for metric,groups in frame_groups.items():
                    for scene,fs in groups.items():
                        values[arm][metric][scene].append(float(np.mean(fs)))
        out["sets"][name]={arm:{metric:{"mean":float(np.mean([np.mean(v) for v in groups.values()])),
                                                 "by_scene_seed":dict(groups)} for metric,groups in metrics.items()}
                           for arm,metrics in values.items()}
    (HERE/"scale_check.json").write_text(json.dumps(out,indent=1),encoding="utf8")


if __name__=="__main__":
    main()
