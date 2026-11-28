"""Real paired-lighting probe. Set ROBI_OBJECT to Chrome_screw or Gear before running."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"physical43_20261124"))
sys.path.insert(0,str(HERE.parents[2]/"research-sources/ml-depth-pro-2/src"))
from depth_pro_2 import PRESETS, create_model_and_transforms
from depth_pro_2.model import resolve_checkpoint
from robi_diagnostic import calibration, grid, pose, read_depth, read_image
from robi_files import CACHE, OBJECT, cached, selection


def photo_names(row):
    root = row["raw"][0].split("/Depth/")[0]
    return [f"{root}/{mode}/LEFT_view_{row['ids'][0]}.bmp" for mode in ("Stereo","StereoPattern")]


def anchor_comparison(x,y,ordinary,pattern,native,truth,frame_shape):
    h,w = frame_shape
    queries = (x%16==0)&(y%16==0)
    test = (~queries)&(truth>0)
    difference = np.abs(np.log(ordinary/pattern))
    grid_pools, centers = [],[]
    for right,lower in ((False,False),(True,False),(False,True),(True,True)):
        pool = np.flatnonzero(queries & ((x>=w/2)==right) & ((y>=h/2)==lower))
        distance = (x[pool]-(3+2*right)*w/8)**2 + (y[pool]-(3+2*lower)*h/8)**2
        grid_pools.append(pool[np.argsort(distance,kind="stable")])
        centers.append(int(grid_pools[-1][0]))
    def measure(idx):
        idx = np.asarray(idx)
        usable = idx[native[idx]>0]
        factor = np.exp(np.median(np.log(native[usable]/ordinary[usable]))) if len(usable) else 1.
        return [float(np.abs(factor*ordinary[test]-truth[test]).mean()),float(len(usable))]
    rng = np.random.default_rng(43163)
    random = [measure([rng.choice(pool) for pool in grid_pools]) for _ in range(128)]
    center_idx = np.asarray(centers)
    good = center_idx[native[center_idx]>0]
    seed_calibration = np.log(native[good]/ordinary[good])
    comparisons = dict(no_anchor=[float(np.abs(ordinary[test]-truth[test]).mean()),0],
                center=measure(centers),random=np.mean(random,axis=0).tolist(),
                high=measure([p[np.argmax(difference[p])] for p in grid_pools]),
                low=measure([p[np.argmin(difference[p])] for p in grid_pools]))
    return comparisons,seed_calibration,ordinary[test],truth[test]


def probe(row,model,transform,repeat):
    names = photo_names(row)
    k = calibration("Ensenso","LEFT")
    kd = calibration("Ensenso")
    assert all(k[key]==kd[key] for key in ("fx","fy","cx","cy","width","height"))
    gt_name,native_name = row["gt"][0],row["raw"][0]
    poses = [pose(n) for n in (gt_name,native_name,*names)]
    assert all(np.allclose(poses[0],p,rtol=0,atol=1e-6) for p in poses)
    gt = read_depth(gt_name,"Ensenso")
    raw = read_depth(native_name,"Ensenso")
    x,y = grid(gt)
    photos = [read_image(name,cv2.IMREAD_GRAYSCALE) for name in names]
    assert all(p.shape==gt.shape for p in photos)
    assert np.any(photos[0]!=photos[1])
    def predict(photo):
        tensor = transform(cv2.cvtColor(photo,cv2.COLOR_GRAY2RGB))
        return model.infer(tensor,f_px=k["fx"])["depth"].float().cpu().numpy()[y,x].astype(float)*1000
    with torch.inference_mode():
        ordinary,pattern = [predict(p) for p in photos]
        control = float(np.max(np.abs(ordinary-predict(photos[0])))) if repeat else None
    assert np.all(np.isfinite(ordinary)&np.isfinite(pattern)&(ordinary>0)&(pattern>0))
    z = gt[y,x]
    known = z>0
    base,alt,truth = ordinary[known],pattern[known],z[known]
    delta = np.log(alt/base)
    anchors,seed_log,test_predictions,test_truth = anchor_comparison(x,y,ordinary,pattern,raw[y,x],z,gt.shape)
    replay = CACHE/"depthpro2_probe"
    replay.mkdir(exist_ok=True)
    preset = os.environ.get("DEPTHPRO_PRESET","depthpro2-s")
    np.savez_compressed(replay/f"{preset}_{row['scene']}_{row['ids'][0]}.npz",pred=test_predictions,gt=test_truth)
    result = dict(scene=row["scene"],view=row["ids"][0],reference_points=int(known.sum()),
                  control_max_mm=control,median_response_mm=float(np.median(np.abs(base-alt))),
                  pattern_over_ordinary=float(np.exp(np.median(delta))),
                  response_after_scale_pct=float(100*np.expm1(np.median(np.abs(delta-np.median(delta))))),
                  anchors=anchors,center_valid_log_ratios=seed_log.tolist())
    for mode,pred in (("ordinary",base),("pattern",alt)):
        gain = np.exp(np.median(np.log(pred/truth)))
        result[mode]=dict(ratio=float(gain),mean_abs_mm=float(np.mean(np.abs(pred-truth))),
                          aligned_mean_abs_mm=float(np.mean(np.abs(pred/gain-truth))))
    return result


if __name__=="__main__":
    rows = [r for r in selection() if r["sensor"]=="Ensenso"]
    files = {n for r in rows for name in photo_names(r)+[r["raw"][0],r["gt"][0]]
             for n in (name,name.rsplit(".",1)[0]+".json")}
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(cached,sorted(files)))
    model_name = os.environ.get("DEPTHPRO_PRESET","depthpro2-s")
    preset = replace(PRESETS[model_name],fov_checkpoint_uri=None)
    model,transform = create_model_and_transforms(preset,device="cuda")
    results = [probe(row,model,transform,i==0) for i,row in enumerate(rows)]
    assert results[0]["control_max_mm"]==0
    with resolve_checkpoint(preset.checkpoint_uri).open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint,"sha256").hexdigest()
    log = dict(object=OBJECT,preset=model_name,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
               weights_sha256=digest,
               input_hashes={n:hashlib.sha256(path.read_bytes()).hexdigest() for n,path in zip(sorted(files),paths)},
               rows=results)
    suffix = "" if model_name=="depthpro2-s" else "_"+model_name.split("-")[-1]
    (HERE/(OBJECT.lower()+suffix+".json")).write_text(json.dumps(log,indent=2),encoding="utf8")
    print(OBJECT,len(results),np.mean([r["median_response_mm"] for r in results]),flush=True)
