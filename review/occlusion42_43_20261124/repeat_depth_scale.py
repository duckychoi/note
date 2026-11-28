"""Within-scene diagnostic of focal length and scale under identical physical brackets."""
from collections import defaultdict
import json

import cv2
import numpy as np

from repeat_capture_acquire import OUT
from repeat_capture_render import FOCAL, BASELINE, LOG_MIN, VIEWS, geometry, load_gray, sample
from repeat_encoder import quadrants


def main():
    core=json.loads((OUT/"results.json").read_text())
    groups=defaultdict(list)
    rows={}
    for scene,row in core["by_scene"].items():
        if row["design_family"]:
            continue
        src=OUT/"physical"/scene
        x,disp,dmin,inv,xyz,normal,good=geometry(src)
        y=np.arange(x.shape[0])[:,None]
        enc=np.load(OUT/"encoder"/(scene+".npz"))
        render=np.load(OUT/"renders"/(scene+".npz"))
        pred=cv2.resize(enc["inverse"],(x.shape[1],x.shape[0]),interpolation=cv2.INTER_LINEAR)
        ids,support=render["source_x"],render["valid"].astype(np.float32)
        good &= np.logical_or.reduce(quadrants(*x.shape))
        for i,v in enumerate(VIEWS):
            gx=x-v*disp
            good &= (sample(support[i],gx)>.999)&(abs(sample(ids[i],gx)-x)<.6)
        # Reference and diagnostic pixels occupy disjoint 32x32 tiles.
        tiles=(x.astype(int)//32+y//32)%2==0
        fit=good & tiles & (pred>1e-5)
        if not fit.any():
            raise ValueError(f"Empty scale calibration in {scene}")
        scale=float(np.median(inv[fit]/pred[fit]))
        ratio=float(enc["focal"])/(FOCAL*enc["inverse"].shape[1]/x.shape[1])
        depths=dict(model=pred,known_focal=pred*ratio,reference_scale=pred*scale,truth=inv)
        keep=good & ~tiles
        coords={}
        for k,rho in depths.items():
            coords[k]=[x-v*(FOCAL*BASELINE*rho-dmin) for v in VIEWS]
            for i,gx in enumerate(coords[k]):
                keep &= sample(support[i],gx)>.999
        if not keep.any():
            raise ValueError(f"Empty matched diagnostic in {scene}")
        energies={k:[] for k in depths}
        for exp in (1,2):
            contrasts=[]
            for v in (3,5):
                images=[load_gray(src/f"Illum{l}/Exp{exp}/view{v}.png") for l in (1,2)]
                contrasts.append(np.log(np.clip(images[0],LOG_MIN,1))-np.log(np.clip(images[1],LOG_MIN,1)))
            for k,(a,b) in coords.items():
                u=(sample(contrasts[1],b)-sample(contrasts[0],a))[keep]
                energies[k].append(np.mean(u.astype(float)**2))
        metrics={k:float(np.mean(vals)) for k,vals in energies.items()}
        rows[scene]=dict(scale=scale,focal_ratio=ratio,fit=int(fit.sum()),eval=int(keep.sum()),mse=metrics)
        groups[row["family"]].append(rows[scene])
    rms={k:float(np.sqrt(np.mean([np.mean([s["mse"][k] for s in grp]) for grp in groups.values()])))
         for k in depths}
    improved={k:sum(float(np.mean([s["mse"][k]-s["mse"]["model"] for s in grp]))<0 for grp in groups.values())
              for k in ("known_focal","reference_scale")}
    (OUT/"scale-diagnostic.json").write_text(json.dumps(dict(rms=rms,improved_families=improved,
                  pixels=sum(s["eval"] for s in rows.values()),scenes=rows),indent=1),encoding="utf8")
    print(rms,improved,flush=True)


if __name__=="__main__":
    main()
