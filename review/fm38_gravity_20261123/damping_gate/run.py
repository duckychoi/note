"""Fit on the declared damped demonstrations; save one set of draws for all correction strengths."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

from world import CACHE,HERE
from controls import fitted_drag_path,fit_drag,features,fit_ridge
sys.path.insert(0,str(HERE.parent/"probe_integration"))
from experiment import (Data,SEEDS,load,make_planner,training_sets,train_residual,prior_for,
                        actions,ensure_reacher_checkpoint)

CHECKPOINTS=CACHE/"damping_gate"
SPLITS=("val","seen","strong","clean")


@torch.inference_mode()
def draw(net,prior,data):
    batches=[]
    rng=torch.Generator(device="cuda").manual_seed(53815)
    for start in range(0,len(data.a),16):
        ix=slice(start,start+16)
        paths=net.forecast(prior.at(ix),actions(data.a[ix]),generator=rng)
        batches.append(paths.cpu().numpy())
    return np.concatenate(batches)


def run():
    CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    ensure_reacher_checkpoint()
    bank={}
    visual=load("reacher")
    splits={s:Data("damp_"+s,visual,bank) for s in ("train",*SPLITS)}
    del visual
    template=make_planner(splits["train"])
    train=training_sets(template,splits["train"])
    val=training_sets(template,splits["val"])
    prior={s:prior_for(template,data,"probe") for s,data in splits.items()}
    numeric={}
    raw={}
    for s,data in splits.items():
        with torch.no_grad():
            numeric[s]={"observed":template.read_angle(data.z).cpu().numpy(),
                        "k":prior[s].strength.cpu().numpy(),"a":data.a.cpu().numpy(),
                        "prior":prior[s].angles.cpu().numpy()}
        if s in SPLITS:
            with np.load(CACHE/f"damp_{s}.npz") as actual:
                assert np.array_equal(actual["action"].reshape(-1,4),np.tile(np.arange(4),(len(actual["action"])//4,1)))
                for key in ("state","action","angles","goal","peak","level","damping"):
                    raw[f"{s}_{key}"]=actual[key]
            raw[f"{s}_prior"]=numeric[s]["prior"]
    # Our two sets of fitted parameters see source training data only.
    drag,fit_errors=fit_drag(numeric["train"])
    ridge=fit_ridge(numeric["train"])
    log={"global_drag":drag,"training_MAE_by_drag_rad":fit_errors,"fit":{}}
    for s in SPLITS:
        raw[f"{s}_drag"]=fitted_drag_path(numeric[s],drag)
        raw[f"{s}_ridge"]=numeric[s]["prior"]+ridge.predict(features(numeric[s]))
    np.savez_compressed(HERE/"cases.npz",**raw)
    del template
    for seed in SEEDS:
        print("training",seed,flush=True)
        net,log["fit"][str(seed)]=train_residual(make_planner(splits["train"]),train,val,seed,
                                                 checkpoint_dir=CHECKPOINTS)
        result={s:draw(net,prior[s],splits[s]) for s in SPLITS}
        np.savez_compressed(HERE/f"flow_{seed}.npz",**result)
        (HERE/"fit.json").write_text(json.dumps(log,indent=2),encoding="utf8")
        print("saved",seed,flush=True)
        del net


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    run()
