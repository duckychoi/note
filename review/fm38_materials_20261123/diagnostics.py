"""Describe the bottlenecks separately from the registered model comparisons."""
import json
import numpy as np
import torch
from analyse import SAFE
from benchmark import Data,Mixture,Probe,SEEDS,metrics
from public_model import HERE
from worlds import ROOT,TASKS


@torch.inference_mode()
def run(task):
    tr=Data(task,"train")
    saved=torch.load(ROOT/f"{task}_probe.pt",weights_only=True)
    probe=Probe(tr).cuda().eval()
    probe.load_state_dict(saved["state"])
    probe.use_class=saved["use_class"]
    out={}
    for domain in ("source","shift"):
        data=Data(task,domain)
        estimated=probe.hit(data.z[:,-3:],task).all(1)
        true=data.y[:,-3:].all(1)
        m=metrics(estimated,true)
        p=estimated.cpu().numpy().reshape(-1,4)
        y=true.cpu().numpy().reshape(-1,4)
        u=np.where(p.any(1)[:,None],y[np.arange(len(y)),p.argmax(1)][:,None],SAFE)
        m["value_using_actual_future_images"]=float(u.mean())
        out[domain]=m
    data=Data(task,"val")
    sizes=[]
    for seed in SEEDS:
        model=Mixture(tr).cuda().eval()
        model.load_state_dict(torch.load(ROOT/f"{task}_mixture_{seed}.pt",weights_only=True))
        logits,_,_=model(data.z[:,0],data.a)
        w=logits.softmax(-1)
        sizes.append({"mean_weight":w.mean(0).tolist(),
                      "mean_effective_components":float((1/w.square().sum(1)).mean())})
    out["mixture_components_source_val"]=sizes
    return out


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    (HERE/"diagnostics.json").write_text(json.dumps({t:run(t) for t in TASKS},indent=2),encoding="utf8")
