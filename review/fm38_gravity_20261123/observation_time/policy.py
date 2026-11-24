"""Fit source-only camera policies and freeze the camera time before evaluating new starts."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from threadpoolctl import threadpool_limits

from simulate import HERE,SNAP,TRAIN_SEEDS,TEST_SEEDS,CONFIRM_SEEDS
from swing import BAND,CACHE,PROBE_SPEED,SPEED,TIMES,raster
from models import AngleReader,load,vision
from controls import fit_ridge,features
sys.path.insert(0,str(Path(os.environ.get("LEWAM_SOURCE",HERE.parents[3]/"research-sources/le-wam"))/"src"))
from probe_planner import identify,swing_trajectory


class Camera:
    def __init__(self):
        self.net=load("reacher")
        stats=torch.load(CACHE/"reader.pt",weights_only=True)
        self.read=AngleReader(stats=(stats["center"],stats["scale"])).cuda().eval()
        self.read.load_state_dict(stats)
        self.bank={}

    @torch.inference_mode()
    def decode(self,angles):
        # The renderer draws identical RGB for identical integer bob centers.
        v=angles.ravel()
        keys=(np.rint(28+49*np.sin(v)).astype(int)+256)*1024+np.rint(30+49*np.cos(v)).astype(int)+256
        ids,first,inverse=np.unique(keys,return_index=True,return_inverse=True)
        missing=[(i,j) for i,j in zip(ids,first) if i not in self.bank]
        for k in range(0,len(missing),64):
            items=missing[k:k+64]
            im=np.stack([raster(v[j]) for _,j in items])
            outputs=self.read(vision(self.net,im)).cpu().numpy()
            self.bank.update({i:float(q) for (i,_),q in zip(items,outputs)})
        return np.array([self.bank[i] for i in ids])[inverse].reshape(angles.shape)


def read_set(name,camera,training=False):
    with np.load(HERE/f"{name}.npz") as f:
        data={"start":camera.decode(f["start"]),"snap":camera.decode(f["probe"]),"action":f["action"]}
        if training:
            data["y"]=camera.decode(f["future"]).reshape(-1,5)
        else:
            data["goal"]=f["goal"]
            data["success"]=abs(f["peak"]-f["goal"][:,None])<=BAND
    return data


def inputs(data,idx):
    n,branches=data["action"].shape
    a=data["action"].flatten()
    start=data["start"]
    if idx is None:
        return np.column_stack([np.repeat(start,branches),SPEED[a]])
    after=data["snap"][:,idx]
    k=identify(start,after,PROBE_SPEED,SNAP[idx])
    q0=np.repeat(start,branches)
    force=np.repeat(k,branches)
    prior=swing_trajectory(q0,SPEED[a],force,TIMES).astype(np.float32)
    return {"a":a,"k":force,"observed":np.column_stack([q0,np.repeat(after,branches)]),"prior":prior}


def direct_x(data):
    return np.column_stack([data["observed"],SPEED[data["a"]]])


def direct_fit(x,y):
    return make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),
                         StandardScaler(),Ridge(alpha=1.)).fit(x,y)


def fit_all(train):
    models={"static":direct_fit(inputs(train,None),train["y"]),"prior":[],"direct":[]}
    for j in range(len(SNAP)):
        p=inputs(train,j)
        targets={**p,"observed":np.column_stack([p["observed"],train["y"]])}
        models["prior"].append(fit_ridge(targets))
        models["direct"].append(direct_fit(direct_x(p),train["y"]))
    return models


def pick(predicted,data):
    peak=predicted.max(-1).reshape(data["action"].shape)
    return abs(peak-data["goal"][:,None]).argmin(1)


def decisions(models,data):
    out={}
    for method in ("prior","direct"):
        out[method]=np.empty((len(models),len(SNAP),len(data["start"])),np.uint8)
    out["static"]=np.stack([pick(m["static"].predict(inputs(data,None)),data) for m in models]).astype(np.uint8)
    for j in range(len(SNAP)):
        p=inputs(data,j)
        for i,m in enumerate(models):
            out["prior"][i,j]=pick(p["prior"]+m["prior"][j].predict(features(p)),data)
            out["direct"][i,j]=pick(m["direct"][j].predict(direct_x(p)),data)
    out["truth"]=data["success"]
    return out


def success(out,method):
    a=out[method]
    return out["truth"][np.arange(a.shape[-1]),a]


def fit(camera):
    models=[fit_all(read_set(f"train_{s}",camera,True)) for s in TRAIN_SEEDS]
    out=decisions(models,read_set("validation",camera))
    np.savez_compressed(HERE/"validation_choices.npz",**out)
    joblib.dump(models,HERE/"policies.joblib",compress=3)
    selected={}
    for method in ("prior","direct"):
        totals=success(out,method).sum((0,2))
        selected[method]={"all":int(totals.argmax()),"capped":int(totals[SNAP<=.55].argmax()),
                          "validation_counts":totals.tolist(),"validation_denominator":int(len(models)*len(out["truth"]))}
    selected["model_sha256"]=hashlib.sha256((HERE/"policies.joblib").read_bytes()).hexdigest()
    (HERE/"selection.json").write_text(json.dumps(selected,indent=2),encoding="utf8")
    print([(m,SNAP[selected[m]["all"]],SNAP[selected[m]["capped"]]) for m in ("prior","direct")])


def test(camera,confirmation=False):
    locked=json.loads((HERE/"selection.json").read_text(encoding="utf8"))
    assert hashlib.sha256((HERE/"policies.joblib").read_bytes()).hexdigest()==locked["model_sha256"]
    models=joblib.load(HERE/"policies.joblib")
    for seed in (CONFIRM_SEEDS if confirmation else TEST_SEEDS):
        for d in (("strong",) if confirmation else ("seen","strong")):
            name=f"{d}_{seed}"
            out=decisions(models,read_set(name,camera))
            np.savez_compressed(HERE/f"{name}_choices.npz",**out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("fit","test","confirmation"))
    stage=p.parse_args().stage
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    with threadpool_limits(limits=2):
        camera=Camera()
        if stage=="fit":
            fit(camera)
        else:
            test(camera,stage=="confirmation")
