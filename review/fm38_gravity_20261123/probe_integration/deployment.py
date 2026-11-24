"""Select a binary correction gate on source validation, then audit it on fresh paired starts."""
import json
import numpy as np
import torch
from experiment import BAND,CACHE,DOMAINS,HERE,SEEDS,Data,forecast,load_artifact,prior_for
from evaluate import choices


def test_modes(net, data, split):
    p=prior_for(net,data,"probe")
    with np.load(CACHE/f"{split}.npz") as f:
        y=(abs(f["peak"]-f["goal"])<=BAND).reshape(-1,4)
    rates={}
    raw={}
    for m,residual in (("prior",False),("flow",True)):
        prediction=forecast(net,p,data,residual)
        raw[m]=prediction
        rates[m]=float(y[np.arange(len(y)),choices(prediction)].mean())
    raw["truth"]=y
    return rates,raw


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    policy={}
    archive={}
    bank={}
    first=load_artifact(SEEDS[0])
    val=Data("val",first.jepa,bank)
    test={d:Data("gate_"+d,first.jepa,bank) for d in DOMAINS}
    for seed in SEEDS:
        net=first if seed==SEEDS[0] else load_artifact(seed)
        selection,raw=test_modes(net,val,"val")
        archive.update({f"val_{seed}_{k}":x for k,x in raw.items()})
        selected=selection["flow"]>selection["prior"]
        policy[str(seed)]={"use_residual":selected,"validation":selection,"test":{}}
        for d,x in test.items():
            stats,raw=test_modes(net,x,"gate_"+d)
            stats["chosen"]=stats["flow" if selected else "prior"]
            policy[str(seed)]["test"][d]=stats
            archive.update({f"{d}_{seed}_{k}":v for k,v in raw.items()})
    np.savez_compressed(HERE/"gate_scores.npz",**archive)
    (HERE/"deployment.json").write_text(json.dumps(policy,indent=2),encoding="utf8")
    print(json.dumps(policy))
