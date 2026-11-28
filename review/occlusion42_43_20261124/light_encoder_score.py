"""Score locked single-image students and independently score the two-exposure teacher."""
import argparse
from collections import defaultdict
import json
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from depth_full_eval import load_sample, sample_names
from light_encoder_study import (ARMS, CKPT, K_SCALE, OUT, Model, alternate, centered, edges,
                                 load_scene, manifest, moments, sha, write, yield_memory)
from native_direction_data import PREV
from repeat_encoder import DESIGN, family
from scale_surface_evaluate import frames


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row["scores"])
    keys = next(iter(groups.values()))[0]
    per_group = {g:{k:float(np.mean([row[k] for row in items])) for k in keys}
                 for g,items in groups.items()}
    means = {k:float(np.mean([v[k] for v in per_group.values()])) for k in keys}
    return dict(n=len(rows), groups=per_group, means=means)


def exposure_teacher(model, first, second, z, mask, gains):
    _, _, q = model.baseline(first)
    _, _, other_q = model.baseline(second)
    _, d, r, bins, _ = edges(first, second, q, other_q, z, mask)
    if not r.numel():
        return None
    scores = {"original":float(r.square().mean()), "half":float((r-.5*d).square().mean())}
    for arm, gain in gains.items():
        scores[arm] = float((r-gain[bins]*d).square().mean())
    table = moments(d,r,bins)
    for i in range(4):
        scores[f"count_{i}"] = float(table[0,i])
    return scores,table.tolist()


def teacher_gain_diagnostics(rows):
    # Per-frame mean loss first, then equal group weights, exactly as for teacher scores.
    by_group = defaultdict(list)
    for row in rows:
        table = np.array(row["moments"])
        by_group[row["group"]].append(table / table[0].sum())
    tables = np.stack([np.mean(x,axis=0) for x in by_group.values()])
    def optimal(t):
        return np.clip(np.divide(t[1],t[2],out=np.zeros(4),where=t[2]>0),0,1)
    def risk(t,k):
        return float((t[3] - 2*k*t[1]+k*k*t[2]).sum())
    average = tables.mean(0)
    oracle = optimal(average)
    loo = [risk(t,optimal(np.mean(np.delete(tables,i,axis=0),axis=0))) for i,t in enumerate(tables)]
    return dict(retrospective_gain=oracle.tolist(), retrospective_risk=risk(average,oracle),
                leave_group_out_risk=float(np.mean(loo)), uncorrected_risk=risk(average,0.))


def middlebury_pairs():
    root = PREV.parent / "repeat-capture/physical"
    for src in sorted(root.iterdir()):
        if family(src.name) in set(map(family, DESIGN)):
            continue
        folders = [src/f"Illum{light}/Exp1" for light in (1,2)]
        exposures = [float((path/"exposure_ms.txt").read_text().strip()) for path in folders]
        if exposures[0] != exposures[1]:
            continue
        pair = [np.array(Image.open(path/"view1.png").convert("RGB")) for path in folders]
        disp = np.array(Image.open(src/"disp1.png"), dtype=np.float32)
        denom = disp + float((src/"dmin.txt").read_text())
        z = 3740*.16 / np.maximum(denom, 1e-8)
        mask = (disp>0)&(z>=.25)&(z<=30.)
        if pair[0].shape != pair[1].shape or z.shape != pair[0].shape[:2]:
            raise ValueError(src)
        yield src.name,family(src.name),pair[0],pair[1],torch.as_tensor(z,device="cuda"),torch.as_tensor(mask,device="cuda")


@torch.no_grad()
def teachers():
    model = Model()
    cal = json.loads((OUT/"calibration.json").read_text())
    gains = {k:torch.as_tensor(np.broadcast_to(cal[k],(4,)).copy(), device="cuda", dtype=torch.float32)
             for k in ("fixed","conditional")}
    def synthetic():
        for row in manifest():
            if row[0] == "val":
                rgb,z,mask,_ = load_scene(row)
                yield str(row[1:4]),row[1],*alternate(rgb,row),z,mask
    for name, pairs in (("synthetic",synthetic()),("physical",middlebury_pairs())):
        rows = []
        skipped = []
        for key,group,first,second,z,mask in pairs:
            yield_memory()
            orientations = [(first,second),(second,first)] if name=="physical" else [(first,second)]
            for i,(a,b) in enumerate(orientations):
                measurement = exposure_teacher(model,a,b,z,mask,gains)
                if measurement is None:
                    skipped.append([key,i])
                else:
                    scores,table = measurement
                    rows.append(dict(key=key,group=group,orientation=i,scores=scores,moments=table))
        report = dict(**summary(rows), skipped=skipped, rows=rows, calibration=sha(OUT/"calibration.json"),
                      diagnostic=teacher_gain_diagnostics(rows))
        write(OUT/f"teacher-{name}.json", report)
        print("teacher", name, report["n"], flush=True)


def images(dataset):
    if dataset=="synthetic":
        for row in manifest():
            if row[0]=="val":
                rgb,z,mask,_ = load_scene(row)
                yield str(row[1:4]),row[1],rgb,z,mask
    elif dataset=="diode":
        with ZipFile(PREV.parent/"DIODE.zip") as archive:
            for name in sample_names(archive,True):
                rgb,z,mask,_ = load_sample(archive,name,1)
                group = re.search(r"(?i)scene[_-]?\d+",name).group()
                yield name,group,rgb,z,mask
    elif dataset=="ibims":
        yield from frames("ibims")
    else:
        raise ValueError(dataset)


@torch.no_grad()
def predict(model, rgb):
    x, state, q = model.baseline(rgb)
    focal = model.net.fov.focal_length_px(x).float()
    raw = focal/(rgb.shape[1]*q)
    out = {"base":raw}
    means = []
    for arm in ARMS:
        h = centered(q,model.q(state,arm))
        means.append(abs(float(h.mean())))
        out[arm] = raw*h.exp()
    return out,max(means)


def score(predictions, z, mask):
    mask = torch.as_tensor(mask,device="cuda")
    z = torch.as_tensor(z,device="cuda")[mask].double()
    if z.numel()==0:
        return None
    scores = {}
    for arm, p in predictions.items():
        ratio = p[mask].double()/z
        log_error = (ratio*K_SCALE).log()
        scores[arm] = float((ratio*K_SCALE-1).abs().mean())
        scores[arm+"_k1"] = float((ratio-1).abs().mean())
        scores[arm+"_scale_mse"] = float(log_error.mean().square())
        scores[arm+"_shape_mse"] = float(log_error.var(unbiased=False))
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Invalid depth prediction")
    return scores


@torch.no_grad()
def evaluate(stage):
    model = Model(True)
    n = sum(r[0]=="fit" for r in manifest())
    step = n//2 if stage=="middle" else n
    path = OUT/f"encoder-{step}.pt"
    record = torch.load(path,weights_only=True,map_location="cuda")
    if record["config"][1]!=sha(OUT/"calibration.json"):
        raise ValueError("Student calibration mismatch")
    for arm in ARMS:
        model.arms[arm].load_state_dict(record["arms"][arm])
    weight_hash = sha(path)
    for dataset in (("synthetic",) if stage=="middle" else ("synthetic","diode","ibims")):
        rows, skipped = [], []
        max_drift = 0.
        for key,group,rgb,z,mask in images(dataset):
            yield_memory()
            p,drift = predict(model,rgb)
            max_drift = max(max_drift,drift)
            s = score(p,z,mask)
            if s is None:
                skipped.append(key)
                continue
            rows.append(dict(key=key,group=group,scores=s))
        result = dict(**summary(rows), rows=rows, skipped=skipped, official_max_error=model.check,
                      max_mean_log_update=max_drift, checkpoint=weight_hash, base=sha(CKPT),
                      step=step, updates=record["updates"], parameters=record["trainable"])
        write(OUT/f"{stage}-{dataset}.json", result)
        print(stage,dataset,len(rows),{a:result["means"][a] for a in ("base",*ARMS)},flush=True)


@torch.no_grad()
def references():
    # These historical training recipes are reference levels, outside the matched three-arm trial.
    model = Model()
    paths = {"nyu_real":PREV.parent/"full-comparison-4243/real/best.pt",
             "silog":CKPT.parent.parent/"silog/best.pt"}
    outputs = {}
    for name,path in paths.items():
        mismatch = model.net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),strict=False)
        if mismatch.unexpected_keys or any(not key.startswith("fov.") for key in mismatch.missing_keys):
            raise ValueError(mismatch)
        for dataset in ("diode","ibims"):
            rows = []
            for key,group,rgb,z,mask in images(dataset):
                yield_memory()
                x = model.transform(rgb)
                prediction = model.net.infer(x)
                raw = prediction["focallength_px"]/(rgb.shape[1]*prediction["inverse_depth"].float().clamp_min(1e-5))
                s = score({name:raw},z,mask)
                if s:
                    rows.append(dict(key=key,group=group,scores=s))
            outputs[f"{name}-{dataset}"] = dict(**summary(rows),rows=rows,checkpoint=sha(path))
            write(OUT/"external-recipes.json",outputs)
            print("reference",name,dataset,len(rows),flush=True)


@torch.inference_mode()
def published_ibims():
    import os
    import sys
    from light_encoder_study import PREVIOUS
    from compare_hypersim_metric import load_depth_anything, predict as da_predict
    src = PREVIOUS.parents[1]
    sys.path.insert(0,str(src/"ml-depth-pro-2/src"))
    os.environ["TORCH_HOME"] = str(src/"ml-depth-pro-2-cache")
    from depth_pro_2 import create_model_and_transforms, PRESETS
    from depth_pro_2.model import resolve_checkpoint
    output = {}
    for label in ("official","small","large"):
        if label=="official":
            net,transform = create_model_and_transforms("depthpro2-s",device="cuda")
            digest = sha(resolve_checkpoint(PRESETS["depthpro2-s"].checkpoint_uri))
        else:
            net,path = load_depth_anything(label)
            digest = sha(path/"model.safetensors")
        rows = []
        for key,group,rgb,z,mask in images("ibims"):
            yield_memory()
            if label=="official":
                result = net.infer(transform(rgb))
                raw = result["focallength_px"]/(rgb.shape[1]*result["inverse_depth"].float().clamp_min(1e-5))
            else:
                raw = torch.as_tensor(da_predict(net,rgb,z.shape),device="cuda")
            s = score({label:raw},z,mask)
            if s:
                rows.append(dict(key=key,group=group,scores=s))
        output[label] = dict(**summary(rows),rows=rows,checkpoint=digest)
        del net
        write(OUT/"published-ibims.json",output)
        print("published",label,len(rows),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stages",nargs="+",choices=("teachers","middle","final","references","published"))
    for stage in parser.parse_args().stages:
        if stage=="teachers":
            teachers()
        elif stage=="references":
            references()
        elif stage=="published":
            published_ibims()
        else:
            evaluate(stage)
