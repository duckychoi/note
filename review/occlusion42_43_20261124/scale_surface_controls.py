"""Post-hoc mechanism checks, without fitting any parameters to real depth."""
from collections import defaultdict
import json

import numpy as np
from PIL import Image

from scale_surface_evaluate import frames
from scale_surface_rgb import CAP, OUT, Reader, infer_all, nearest, read_heads, resize_rgb, sha, write


def constant_from_training(weights):
    folder=OUT/("fit-"+weights["fit_signature"][:16])
    per_scene=[]
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as data:
            if data["N"][3]:
                per_scene.append(float(data["G"][0,-1]/data["G"][0,0]))
    if len(per_scene)!=weights["scenes"]:
        raise ValueError("Incomplete synthetic fit statistics")
    return float(np.clip(np.mean(per_scene),-.5,.5))


def measure(dataset,width,reader,heads,constant):
    standard=json.loads((OUT/(dataset+"-rows.json")).read_text())["rows"]
    rows={}
    for key,group,photo,truth,valid in frames(dataset):
        small=resize_rgb(photo)
        shape=small.shape[:2]
        if width==small.shape[1]:
            image=small
        else:
            im=Image.fromarray(photo).convert("RGB")
            image=np.array(im.resize((width,round(width*im.height/im.width)),Image.Resampling.BILINEAR))
        rep=reader.extract(image)
        outputs=infer_all(rep,heads)
        base=outputs["base"]
        outputs["one_scale"]=base+np.clip(base*np.expm1(constant),-CAP,CAP)
        outputs["one_direction"]=base+np.sign(constant)*np.minimum(CAP,base/2)
        z=nearest(truth.astype(np.float32),shape).astype(float)
        mask=nearest(valid.astype(np.uint8),shape).astype(bool)&(z>=.25)&(z<=30)&np.isfinite(z)
        if not mask.any():
            continue
        out={k:nearest(v,shape)[mask] for k,v in outputs.items()}
        absrel={k:float(np.mean(np.abs(v/z[mask]-1))) for k,v in out.items()}
        if width==256 and abs(absrel["base"]-standard[key]["scores"]["base"]["absrel"])>1e-10:
            raise ValueError("Base depth or mask changed")
        rows[key]=dict(group=group,**absrel,
                       saturated=float(np.mean(np.isclose(abs(out["stereo_joint"]-out["base"]),CAP,atol=1e-8))))
    grouped=defaultdict(list)
    for row in rows.values():
        grouped[row["group"]].append(row)
    stats={k:float(np.mean([np.mean([r[k] for r in g]) for g in grouped.values()]))
           for k in next(iter(rows.values())) if k!="group"}
    return dict(width=width,score_width=256,frames=len(rows),groups=len(grouped),means=stats)


def main():
    weights=json.loads((OUT/"weights.json").read_text())
    constant=constant_from_training(weights)
    reader=Reader()
    heads=read_heads()
    results={}
    for dataset,width in (("ibims",256),("ibims",512),("middlebury",256),("nyu",256)):
        key=f"{dataset}_{width}"
        results[key]=measure(dataset,width,reader,heads,constant)
        write(OUT/"mechanism-controls.json",dict(constant_log_scale=constant,
              constant_ratio=float(np.exp(constant)),heads_sha256=sha(OUT/"weights.json"),results=results))
        print(key,results[key]["means"],flush=True)


if __name__=="__main__":
    main()
