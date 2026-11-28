"""Single-view YCB-Video evaluation. Frames are nested within videos, not iid."""
import json
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

from experiment import ARMS, SEEDS
from footage import TEST, read
from scenes import CACHE, HERE, DepthNet


def summarize(rows, prefix=""):
    rng = np.random.default_rng(420043)
    summary = {}
    for metric in ("whole","thin","inner","outer"):
        grouped = defaultdict(list)
        for row in rows:
            if row["counts"][metric] >= (1 if metric=="whole" else 16):
                grouped[row["scene"]].append(row)
        scenes = sorted(grouped)
        if not scenes:
            summary[metric]={"scenes":[]}
            continue
        resample = rng.integers(0,len(scenes),(20000,len(scenes)))
        results = {}
        for budget in ("zero","few"):
            vals = {}
            for name in ARMS:
                vals[name] = np.array([[np.mean([r["results"][f"{seed}/{budget}/{name}"][metric]
                                    for r in grouped[s]]) for s in scenes] for seed in SEEDS])
            contrast = {}
            for name in ARMS:
                delta = vals["gain"]-vals[name]
                units = delta.mean(axis=0)
                samples = units[resample].mean(1)
                contrast[name]=dict(delta=float(units.mean()),upper95=float(np.quantile(samples,.95)),
                                    ci95=np.quantile(samples,[.025,.975]).tolist(),wins=int((units<0).sum()),
                                    seed_delta=delta.mean(1).tolist(),scene_delta=units.tolist())
            results[budget] = {"means":{n:v.mean() for n,v in vals.items()},"gain_minus":contrast,
                               "seed_means":{n:v.mean(1).tolist() for n,v in vals.items()},
                               "scene_means":{n:v.mean(0).tolist() for n,v in vals.items()}}
        summary[metric]=dict(scenes=scenes,n_frames=[len(grouped[s]) for s in scenes],
                  n_pixels=sum(r["counts"][metric] for s in scenes for r in grouped[s]),results=results)
    (HERE/(prefix+"summary.json")).write_text(json.dumps(summary,indent=1),encoding="utf8")


def run(extra=False):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    prefix = "extra_" if extra else ""
    if extra:
        chosen = json.loads((HERE/"extra_manifest.json").read_text())["scenes"]
        frames = read(chosen,CACHE/"supplement.zip",8)
    else:
        frames=read(TEST)
    show_masks(frames,prefix)
    rows=[]
    for _,depth,band,inside,scene,frame,ids in frames:
        valid = (depth>=.2)&(depth<=5.)
        counts = {n:int(mask.sum()) for n,mask in masks(valid,band,inside).items()}
        rows.append(dict(scene=scene,frame=frame,object_ids=ids,counts=counts,results={}))
    inputs=torch.stack([r[0] for r in frames]).cuda()
    for seed in SEEDS:
        for budget in ("zero","few"):
            for arm in ARMS:
                net=DepthNet(False).cuda()
                path=CACHE/"checkpoints"/f"{seed}_{budget}_{arm}.pt"
                net.load_state_dict(torch.load(path,weights_only=True))
                net.eval()
                with torch.no_grad():
                    preds=torch.cat([net(x)[:,0].exp() for x in inputs.split(8)]).cpu().numpy()
                for i,(pred, frame) in enumerate(zip(preds,frames)):
                    _,depth,band,inside,_,_,_ = frame
                    valid=(depth>=.2)&(depth<=5.)
                    error=1000*np.abs(pred-depth)
                    rows[i]["results"][f"{seed}/{budget}/{arm}"] = {
                        n:float(error[mask].mean()) for n,mask in masks(valid,band,inside).items() if mask.any()}
    (HERE/(prefix+"frames.json")).write_text(json.dumps(rows,separators=(",",":")),encoding="utf8")
    summarize(rows,prefix)


def masks(valid, band, inside):
    return dict(whole=valid,thin=band&valid,inner=inside&valid,outer=band&~inside&valid)


def show_masks(frames,prefix):
    groups = defaultdict(list)
    for row in frames:
        _,depth,band,_,scene,_,_ = row
        if ((depth>=.2)&(depth<=5.)&band).sum() >= 16:
            groups[scene].append(row)
    if not groups:
        return
    panels = []
    for scene, rows in sorted(groups.items()):
        x,depth,band,inside,_,frame,_ = rows[len(rows)//2]
        bgr = cv2.cvtColor(np.uint8(np.clip((x[:3].numpy().transpose(1,2,0)+1)*127.5,0,255)),cv2.COLOR_RGB2BGR)
        valid = (depth>=.2)&(depth<=5.)
        bgr[inside&valid] = (40,230,255)
        bgr[band&~inside&valid] = (255,205,30)
        cv2.rectangle(bgr,(0,0),(164,22),(22,22,22),-1)
        cv2.putText(bgr,f"Video {scene} / {frame}",(5,16),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        panels.append(bgr)
    if len(panels)%2:
        panels.append(np.zeros_like(panels[0]))
    image = np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)])
    cv2.imencode(".png",image)[1].tofile(HERE/(prefix+"thin_masks.png"))


if __name__=="__main__":
    run(extra="--extra" in sys.argv)
