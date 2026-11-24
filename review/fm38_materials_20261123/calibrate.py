"""Fresh paired test of material-specific action-rank prediction with 64 labeled probes per action."""
import json
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import brentq
import torch

from analyse import DOMAINS,PAIRS,policy
from benchmark import Data,Mixture,Probe,Retrieval,SEEDS,scores
from public_model import HERE,load,vision
from worlds import ROOT,TASKS,labels,render


def fit_offsets(log_odds,y):
    b=np.zeros(4)
    for a in range(4):
        def gradient(offset):
            return (expit(log_odds[:,a]+offset)-y[:,a]).sum()+offset/4
        bound=4*(len(y)+1)
        b[a]=brentq(gradient,-bound,bound)
        assert abs(gradient(b[a]))<1e-8
    return b


def signs(arr):
    return np.sign(np.stack([arr[...,i]-arr[...,j] for i,j in PAIRS],-1))


def assess(pred,actual):
    t=np.stack([signs(x) for x in actual])
    p=np.stack([signs(x) for x in pred])
    reversed_t=(t[0]*t[1])<0
    reversed_p=(p[0]*p[1])<0
    correct=reversed_t & reversed_p & (p[0]==t[0])
    return {"true_strict_pairs":int(reversed_t.sum()),"predicted_reversals":int(reversed_p.sum()),
            "direction_correct":int(correct.sum()),"sensitivity":float(correct.sum()/reversed_t.sum()),
            "precision":float(correct.sum()/reversed_p.sum()) if reversed_p.any() else None,
            "values":[float(policy(x[None],y)[0].mean()) for x,y in zip(pred,actual)]}


@torch.inference_mode()
def run(task):
    train=Data(task,"train")
    visual=load(task)
    state=torch.load(ROOT/f"{task}_probe.pt",weights_only=True)
    probe=Probe(train).cuda().eval()
    probe.load_state_dict(state["state"])
    probe.use_class=state["use_class"]
    obs={}
    truth={}
    raw={}
    for split in ("cal","fresh"):
        starts=[]
        truth[split]=[]
        for dom in DOMAINS:
            with np.load(ROOT/f"{task}_{split}_{dom}.npz") as data:
                starts.append(data["path"][:,0])
                truth[split].append(labels(task,data)[:,-3:].all(1).astype(int).reshape(-1,4))
        assert np.array_equal(*starts)
        frames=np.stack([render(task,x) for x in starts[0][::4]])
        obs[split]=torch.cat([vision(visual,frames[i:i+64]) for i in range(0,len(frames),64)]).repeat_interleave(4,0)
        raw["truth_"+split]=np.stack(truth[split])
    results={}
    for name in ("retrieval","mixture","flow","frequencies"):
        result=[]
        for seed in (SEEDS if name in ("flow","mixture") else SEEDS[:1]):
            if name=="frequencies":
                pred=[np.broadcast_to((c.sum(0)+.5)/(len(c)+1),y.shape) for c,y in zip(truth["cal"],truth["fresh"])]
            else:
                if name=="retrieval":
                    model=Retrieval(train)
                elif name=="mixture":
                    model=Mixture(train).cuda().eval()
                    model.load_state_dict(torch.load(ROOT/f"{task}_{name}_{seed}.pt",weights_only=True))
                else:
                    model=load(task)
                    state=torch.load(ROOT/f"{task}_{name}_{seed}.pt",weights_only=True)
                    for key in ("predictor","action_encoder"):
                        getattr(model,key).load_state_dict({k[len(key)+1:]:x for k,x in state.items() if k.startswith(key+".")})
                logits={}
                for split,z in obs.items():
                    torch.manual_seed(3901 if split=="cal" else 3902)
                    a=torch.arange(4,device="cuda").repeat(len(z)//4)
                    pr=torch.cat([scores(task,name,model,probe,z[i:i+16],a[i:i+16])[:,0]
                                  for i in range(0,len(z),16)]).reshape(-1,4).cpu().numpy()
                    logits[split]=logit((32*pr.astype(float)+.5)/33)
                    raw[f"log_odds_{split}_{name}_{seed}"]=logits[split]
                pred=[expit(logits["fresh"]+fit_offsets(logits["cal"],c)) for c in truth["cal"]]
                del model
            raw[f"{name}_{seed}"]=np.stack(pred)
            result.append(assess(pred,truth["fresh"]))
        results[name]=result
    np.savez_compressed(HERE/f"{task}_calibration.npz",**raw)
    return results


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    results={task:run(task) for task in TASKS}
    (HERE/"calibration.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    print(json.dumps(results))
