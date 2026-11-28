"""Audit the old stereo failures and compare a geometry-only choice at identical anchors."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from controlled_observations import RUN, SOURCE, wait_for_ram, write_json
from depth_full_data import Hypersim
from score_controlled_observations import fit, score
from select_stereo_pairs import OUT, coarse, visibility
from stereo_controlled_observations import CONDITIONS, cases_for, rectification, rectified_cameras, stereo_pair


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def stats(rows, anchor=False):
    cases = [[c for c in r["cases"] if not anchor or c["side"]==0] for r in rows]
    n = sum(c["eligible"] for group in cases for c in group)
    total = sum(c["full_valid"] for group in cases for c in group)
    return dict(scenes=len(rows),observed_scenes=sum(any(c["eligible"] for c in g) for g in cases),
                observed=n,valid=total,percent=100*n/total)


def diagnose(expanded=False):
    data = Hypersim("validation_24",4,True)
    rows = []
    for file in sorted(RUN.glob("dev-*.json")):
        sample=read(OUT/f"pairs-dense-{read(file)['scene']}.json") if expanded else read(file)
        s,c,a,b=sample["row"]
        first,second,pose=data.read_pair((s,c,a,b))
        k,size,*_ = coarse(s,c,a)
        rect,why = rectified_cameras([k,k],size,pose)
        previous=read(OUT/f"cases-dense-{s}.json") if expanded else read(RUN/"stereo_cases"/file.name)
        row=dict(scene=s,old_pair=[s,c,a,b],rectification=why,old_counts=stats([previous]))
        if pose is not None:
            row["baseline_m"]=float(np.linalg.norm(pose[:,3]))
            row["lateral_m"]=float(np.linalg.norm(pose[:2,3]))
            row["lateral_fraction"]=row["lateral_m"]/max(1e-12,row["baseline_m"])
            displacement=-pose[:,:3].T@pose[:,3]
            row["anchor_axis_fractions"]=np.abs(displacement/max(1e-12,row["baseline_m"])).tolist()
            row["frame_gap"]=abs(a-b)
            overlap, search = visibility(coarse(s,c,a),coarse(s,c,b),pose,rect)
            row.update(geometry_overlap=overlap,searchable_fraction=search)
            if why == "rectified_focal":
                _,_,p0,p1,*_=cv2.stereoRectify(k,None,k,None,size,pose[:,:3].astype(float),
                    pose[:,3:].astype(float),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
                axis=np.argmax(np.abs(p1[:2,3]))
                row["focal_ratio"]=float(p1[axis,axis]/k[0,0])
        if rect is not None:
            geometry=rectification((first,second),pose)
            lut=np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)
            imgs=[[v[0],lut[v[0]]] for v in (first,second)]
            masks={}
            for l in (0,1):
                for r in (0,1):
                    trace=[]
                    stereo_pair([imgs[0][l],imgs[1][r]],geometry,trace)
                    masks[l,r]=trace
            count=np.zeros(5,dtype=np.int64)
            for side,item in enumerate((first,second)):
                for l in (0,1):
                    other=(l,1-l) if side==0 else (1-l,l)
                    for level,(x,y) in enumerate(zip(masks[l,l][side],masks[other][side])):
                        count[level]+=np.count_nonzero(x & y & item[2][::2,::2])
            stages=("inside_rectified_canvas","source_map","forward_disparity","reverse_correspondence","metric_range")
            row["two_light_funnel"]=dict(zip(stages,count.tolist()))
            row["first_empty_stage"]=next((key for key,n in zip(stages,count) if n==0),"none")
        if row["old_counts"]["observed"]:
            row["first_failure"]="observed"
        elif why=="ok":
            row["first_failure"]="no_two_light_matches"
        else:
            row["first_failure"]=why
        rows.append(row)
    reasons=dict(Counter(r["first_failure"] for r in rows))
    write_json(OUT/("dense_failures.json" if expanded else "old_failures.json"),dict(reasons=reasons,scenes=rows))
    print(reasons,flush=True)


def collect(dense=False):
    import torch
    sys.path.insert(0,str(SOURCE/"ml-depth-pro-2"/"src"))
    from depth_pro_2 import PRESETS,create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint
    dataset=Hypersim("validation_24",4,True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.hub.set_dir(str(SOURCE/"ml-depth-pro-2-cache"/"hub"))
    preset=replace(PRESETS["depthpro2-s"],fov_checkpoint_uri=None)
    with resolve_checkpoint(preset.checkpoint_uri).open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    model,transform=create_model_and_transforms(preset,device="cpu")
    model.eval()
    lut=np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)
    for split in (("dev",) if dense else ("train","dev")):
        label="dense" if dense else split
        for row in read(OUT/f"pairs-{label}.json"):
            scene=row[0]
            path=OUT/f"cases-{label}-{scene}.json"
            if path.exists():
                if read(path)["row"]!=row:
                    raise ValueError(f"Stale pair at {path}")
                continue
            wait_for_ram()
            original=read(RUN/f"{split}-{scene}.json")
            if digest!=original["sha256"] or row[:3]!=original["row"][:3]:
                raise ValueError("Different anchor/model")
            cached=RUN/"predictions"/f"{split}-{scene}.npy"
            pair=dataset.read_pair(row)
            lights=[[item[0],lut[item[0]]] for item in pair[:2]]
            with torch.inference_mode():
                predictions=[]
                for i in (0,1):
                    if cached.exists() and (i==0 or row==original["row"]):
                        predictions.append(np.load(cached,mmap_mode="r")[i])
                    else:
                        predictions.append(np.stack([model.infer(transform(im),f_px=float(pair[i][3][0,0]))
                                                     ["depth"].cpu().numpy() for im in lights[i]]))
            cases=cases_for(pair[:2],lights,pair[2],np.stack(predictions))
            write_json(path,dict(scene=scene,row=row,cases=cases))
        print("scored",split,flush=True)


def dense():
    import csv
    from collections import defaultdict
    from hypersim_fetch import ROOT, scene_camera_files
    from select_stereo_pairs import choose,shortlist
    candidates=defaultdict(list)
    with (ROOT/"splits.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["split_partition_name"]=="val" and r["included_in_public_release"]=="True":
                candidates[r["scene_name"],r["camera_name"]].append(int(r["frame_id"]))
    rows=[]
    for file in sorted(RUN.glob("dev-*.json")):
        s,c,a,b=read(file)["row"]
        dest=OUT/f"pairs-dense-{s}.json"
        if not dest.exists():
            wait_for_ram()
            frames=candidates[s,c]
            if a not in frames or b not in frames:
                raise ValueError("Anchor and fallback must be public validation frames")
            needed=[(c,f) for _,f,_,_ in shortlist(s,c,a,frames)]
            scene_camera_files(s,needed)
            chosen=choose(s,c,a,b,frames)
            write_json(dest,dict(chosen,eligible_trajectory_frames=len(frames),shortlisted=needed))
        rows.append(read(dest)["row"])
    write_json(OUT/"pairs-dense.json",rows)
    collect(dense=True)
    outcomes=[read(p) for p in sorted(OUT.glob("cases-dense-*.json"))]
    old=[read(p) for p in sorted((RUN/"stereo_cases").glob("dev-*.json"))]
    anchors=[dict(r,cases=[c for c in r["cases"] if c["side"]==0]) for r in outcomes]
    assert len(outcomes)==46 and all(a["scene"]==b["scene"] and
        [c["full_valid"] for c in a["cases"] if c["side"]==0]==[c["full_valid"] for c in b["cases"]]
        for a,b in zip(old,anchors))
    fitted=read(OUT/"stereo_results.json")
    gains=(fitted["pooled_gain"],np.array(fitted["fixed_gain"]),np.array(fitted["pixel_gain"]))
    write_json(OUT/"dense_development.json",dict(coverage=stats(outcomes),anchor_coverage=stats(anchors),
        anchor_report=score(anchors,gains,"dev",CONDITIONS,with_fallback=True),
        both_report=score(outcomes,gains,"dev",CONDITIONS,with_fallback=True)))
    diagnose(expanded=True)


def report():
    load = lambda paths:[read(p) for p in sorted(paths)]
    base=load((RUN/"stereo_cases").glob("dev-*.json"))
    train=load(OUT.glob("cases-train-*.json"))
    dev=load(OUT.glob("cases-dev-*.json"))
    real=load((RUN/"stereo_cases").glob("real-*.json"))
    if [len(v) for v in (base,train,dev,real)] != [46,48,46,24]:
        raise ValueError("Incomplete splits")
    if {r["scene"] for r in train}&{r["scene"] for r in dev}:
        raise ValueError("Calibration and development overlap")
    gains=fit(train,CONDITIONS)
    result=dict(fixed_gain=gains[1].tolist(),pixel_gain=gains[2].tolist(),pooled_gain=gains[0],
                coverage={},reports={})
    for name,rows in (("old",base),("selected",dev),("calibration",train),("real",real)):
        result["coverage"][name]={"both_positions":stats(rows),"same_anchor":stats(rows,True)}
    for old,new in zip(base,dev):
        a=[c["full_valid"] for c in old["cases"] if c["side"]==0]
        b=[c["full_valid"] for c in new["cases"] if c["side"]==0]
        if old["scene"]!=new["scene"] or a!=b:
            raise ValueError("Fixed anchor support differs")
    for name,rows,split in (("dev",dev,"dev"),("real",real,"real")):
        for label,side in (("both",None),("anchor",0)):
            subset=[dict(r,cases=[c for c in r["cases"] if side is None or c["side"]==side]) for r in rows]
            result["reports"][f"{name}_{label}"]=score(subset,gains,split,CONDITIONS,with_fallback=True)
    result["student_exploration"]=all(result["reports"][domain]["summary"]["equal_condition_mean"][metric]>0
        for domain in ("dev_both","real_both") for metric in ("reduction_vs_condition_fixed_pct","reduction_vs_shuffled_pct"))
    write_json(OUT/"stereo_results.json",result)
    print(result["coverage"],flush=True)
    for k,v in result["reports"].items():
        print(k,v["summary"]["equal_condition_mean"],flush=True)


def isolate(expanded=False):
    old=sorted((RUN/"stereo_cases").glob("dev-*.json"))
    new=sorted(OUT.glob("cases-dense-*.json" if expanded else "cases-dev-*.json"))
    groups=[[dict(r,cases=[c for c in r["cases"] if c["side"]==0]) for r in map(read,files)]
            for files in (old,new)]
    if len(groups[0])!=46 or [r["scene"] for r in groups[0]]!=[r["scene"] for r in groups[1]]:
        raise ValueError("Missing paired anchor")
    bootstrap=np.random.default_rng(4342).integers(46,size=(10000,46))
    counts=np.array([[[sum(c[key] for c in r["cases"]) for key in ("eligible","full_valid")]
                       for r in g] for g in groups])
    if not np.array_equal(counts[0,:,1],counts[1,:,1]):
        raise ValueError("Different denominator for fixed anchors")
    bootcounts=counts[:,bootstrap].sum(axis=2)
    change=100*(bootcounts[1,:,0]-bootcounts[0,:,0])/bootcounts[0,:,1]
    out=dict(coverage_delta_percentage_points_ci95=np.percentile(change,[2.5,97.5]).tolist(),
             more_same_fewer=[int(fn(counts[1,:,0],counts[0,:,0]).sum()) for fn in (np.greater,np.equal,np.less)],
             fixed_anchor_pair_selection={})
    for label,folder in (("original_fit",RUN),("selected_fit",OUT)):
        fitted=read(folder/"stereo_results.json")
        gains=(fitted["pooled_gain"],np.array(fitted["fixed_gain"]),np.array(fitted["pixel_gain"]))
        scores=[score(group,gains,"dev",CONDITIONS,with_fallback=True) for group in groups]
        risk=np.array([s["scene_log_mse"] for s in scores])[:,:,-1,:]
        if not np.allclose(risk[0,:,0],risk[1,:,0]):
            raise ValueError("Depth errors before fusion changed for the same anchors")
        effect={}
        for method in ("condition_fixed","adaptive"):
            j=scores[0]["method_order"].index(method)
            error=risk[:,:,j]
            sampled=error[:,bootstrap].mean(axis=2)
            effect[method]=dict(old_mse=float(error[0].mean()),new_mse=float(error[1].mean()),
                relative_reduction_pct=100*(1-error[1].mean()/error[0].mean()),
                paired_ci95=np.percentile(100*(1-sampled[1]/sampled[0]),[2.5,97.5]).tolist())
        out["fixed_anchor_pair_selection"][label]=effect
    write_json(OUT/("dense_pair_selection_effect.json" if expanded else "pair_selection_effect.json"),out)
    print(out)


def figure():
    from PIL import Image,ImageDraw,ImageFont
    data=Hypersim("validation_24",4,True)
    changes=[]
    for file in sorted(OUT.glob("cases-dev-*.json")):
        new=read(file)
        s=new["scene"]
        old=read(RUN/f"dev-{s}.json")["row"]
        counts=[stats([r],anchor=True) for r in (read(RUN/"stereo_cases"/f"dev-{s}.json"),new)]
        changes.append((counts[1]["percent"]-counts[0]["percent"],old,new["row"]))
    increased=sorted(r for r in changes if r[0]>0)
    _,old,new=increased[(len(increased)-1)//2]
    s=old[0]
    pred=np.load(RUN/"predictions"/f"dev-{s}.npy",mmap_mode="r")[0,0,::2,::2]
    lut=np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)
    panels=[]
    for row in (old,new):
        a,b,pose=data.read_pair(row)
        geometry=rectification([a,b],pose)
        valid=a[2][::2,::2]&np.isfinite(pred)&(pred>=.25)&(pred<=30)
        mask=np.zeros_like(valid)
        if geometry is not None:
            mask=valid.copy()
            for neighbor in (b[0],lut[b[0]]):
                mask &= stereo_pair([a[0],neighbor],geometry)[0][1]
        overlay=a[0].copy()
        pixelmask=mask.repeat(2,0).repeat(2,1)[:overlay.shape[0],:overlay.shape[1]]
        overlay[pixelmask]=[19,255,139]
        panels.append((a[0],b[0],overlay,mask.sum(),valid.sum()))
    canvas=Image.new("RGB",(1050,665),"white")
    draw=ImageDraw.Draw(canvas)
    font=lambda size:ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",size)
    draw.text((14,6),"같은 첫 사진, 서로 다른 두 번째 사진",fill="#13212b",font=font(28))
    draw.text((15,49),f"Hypersim  {s} / {old[1]}    ·    관측률이 오른 22장면 중 증가폭 가운데 사례",fill="#394650",font=font(16))
    for i,parts in enumerate(panels):
        y=90+i*280
        titles=[f"기준 사진  {old[2]}",f"{'기존' if i==0 else '재선정'} 이웃  {(old,new)[i][3]}",
                f"두 조명 공통 시차  {100*parts[3]/parts[4]:.2f}%"]
        for j,(rgb,label) in enumerate(zip(parts[:3],titles)):
            x=15+j*345
            draw.text((x,y),label,fill="#172a34",font=font(18))
            canvas.paste(Image.fromarray(rgb).resize((325,244),Image.Resampling.NEAREST),(x,y+30))
    draw.text((15,648),"초록색은 원래 기준 사진의 유효 지점 중 두 조명에서 시차 후보가 나온 위치다.",fill="#394650",font=font(12))
    canvas.save(OUT/"same_anchor_observations.png")


if __name__=="__main__":
    cv2.setNumThreads(1)
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x4000)
    OUT.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("diagnose","collect","report","isolate","dense","figure"))
    parser.add_argument("--expanded",action="store_true",help="Use all-trajectory neighbors for diagnosis or matched effect")
    args=parser.parse_args()
    if args.expanded and args.stage not in ("diagnose","isolate"):
        parser.error("--expanded applies to diagnose or isolate")
    action=globals()[args.stage]
    action(expanded=True) if args.expanded else action()
