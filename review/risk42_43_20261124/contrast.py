"""Explain distance-to-constant on verified jumps, without choosing another predictor."""
import json
import numpy as np
import torch

from acquire import HERE,TEST
from measured import PROCESSED
from fit import SEEDS,CKPT,DepthNet,predict,tensor_input


def main():
    torch.set_num_threads(2)
    pairs=json.loads((HERE/"pair_scores.json").read_text())
    few=json.loads((HERE/"calibration.json").read_text())
    pair_scales={(r["seed"],r["budget"],r["arm"]):r["scale"] for r in pairs}
    arms=("replay","utility","pair_uniform","pair_utility")
    scenes={}
    for s in TEST:
        with np.load(PROCESSED/(s+".npz")) as d:
            tri=d["triples"]
            if len(tri)>=8:
                f,b=tri[:,0,None],tri[:,1:]
                z=d["z"].ravel()
                scenes[s]=(tensor_input(d["im0"],d["focal"])[None].cuda(),f,b,z[b]-z[f])
    assert not {"Sword1","Sword2"}<=scenes.keys()  # Every eligible scene is its own cluster.
    out={}
    for seed in SEEDS:
        source=json.loads((HERE/f"seed{seed}.json").read_text())
        for arm in arms:
            net=DepthNet(False).cuda()
            net.load_state_dict(torch.load(CKPT/f"{seed}_{arm}.pt",weights_only=True))
            for s,(im,f,b,t) in scenes.items():
                p=predict(net.eval(),im)[0].ravel()
                for budget in ("zero","few"):
                    scale=pair_scales.get((seed,budget,arm))
                    if scale is None:
                        scale=(source["scales"] if budget=="zero" else few[str(seed)]["real_scale"])[arm]
                    h=(p[b]-p[f])*scale
                    magnitude=float(np.mean(np.abs(h),dtype=np.float64)*1000)
                    credit=float(np.mean(np.minimum(t,np.maximum(0,h)),dtype=np.float64)*2000)
                    gap=float(np.mean(abs(t-h)-t,dtype=np.float64)*1000)
                    assert abs(gap-(magnitude-credit))<.001
                    rec=out.setdefault(budget,{}).setdefault(arm,[])
                    rec.append([magnitude,credit,gap,float((h>0).mean())])
    summary={b:{a:dict(zip(("jump_magnitude_mm","overlap_credit_mm","excess_over_constant_mm","order_accuracy"),np.mean(v,axis=0)))
                for a,v in arms_data.items()} for b,arms_data in out.items()}
    (HERE/"contrast.json").write_text(json.dumps(summary,indent=1),encoding="utf8")
    print(json.dumps(summary,indent=1))


if __name__=="__main__":
    main()
