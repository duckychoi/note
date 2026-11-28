"""Keep the exploratory comparison and its scores separate from the frozen gate."""
import json
import numpy as np
import torch

from acquire import HERE,DEV,TEST
from measured import PROCESSED
from evaluate import metrics
from paired import ARMS
from fit import SEEDS,CKPT,DepthNet,med,predict,tensor_input,bank


def main():
    torch.set_num_threads(2)
    images,ground={},{}
    for s in DEV+TEST:
        with np.load(PROCESSED/(s+".npz")) as f:
            images[s]=tensor_input(f["im0"],f["focal"])[None].cuda()
            ground[s]={k:f[k] for k in (("x","y","z") if s in DEV else ("valid","triples","z"))}
    zcal=np.concatenate([ground[s]["z"] for s in DEV])
    scores=[]
    for seed in SEEDS:
        rec=json.loads((HERE/f"pair{seed}.json").read_text())
        for arm in (*ARMS,"warm_scale"):
            net=DepthNet(False).cuda()
            checkpoint="prior_net" if arm=="warm_scale" else arm
            net.load_state_dict(torch.load(CKPT/f"{seed}_{checkpoint}.pt",weights_only=True))
            net.eval()
            if arm=="warm_scale":
                source=bank(seed,3,16)
                synth=predict(net,source["x"])[:,::4,::4].ravel()
                scale0=med(source["truth"][:,::4,::4].ravel()/synth,synth)
                del source
            else:
                scale0=rec["scales"][arm]
            pred={s:predict(net,x)[0] for s,x in images.items()}
            pcal=np.concatenate([pred[s][ground[s]["y"],ground[s]["x"]] for s in DEV])
            for budget,scale in (("zero",scale0),("few",med(zcal/pcal,pcal))):
                for s in TEST:
                    p=scale*pred[s]
                    assert np.isfinite(p).all() and (p>0).all()
                    scores.append(dict(seed=seed,budget=budget,scene=s,arm=arm,scale=scale,**metrics(p,ground[s])))
    (HERE/"pair_scores.json").write_text(json.dumps(scores,allow_nan=False),encoding="utf8")
    from aggregate import summarize
    primary=json.loads((HERE/"scene_scores.json").read_text())
    summarize(primary+scores,"pair_summary",(("pair_utility",("constant","warm_scale","replay_raw","replay","utility","pair_uniform","pair_truth")),
               ("utility",("warm_scale",))),False)


if __name__=="__main__":
    main()
