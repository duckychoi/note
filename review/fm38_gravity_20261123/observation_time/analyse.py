import json
import numpy as np
from simulate import HERE,SNAP,TEST_SEEDS


def load_results(name):
    with np.load(HERE/f"{name}_choices.npz") as f:
        truth=f["truth"]
        hits={m:truth[np.arange(len(truth)),f[m]] for m in ("prior","direct","static")}
        return hits,truth.any(1)


def intervals(matrices):
    rng=np.random.default_rng(58451)
    n_comparisons=matrices[0].shape[1]
    draws=np.empty((64000,n_comparisons))
    for i in range(0,len(draws),250):
        mean=0.
        for m in matrices:
            n=len(m)
            w=rng.multinomial(n,np.full(n,1/n),size=250)
            mean=mean+w@m/n
        draws[i:i+250]=mean/len(matrices)
    tail=.05/(2*n_comparisons)
    return np.quantile(draws,[tail,1-tail],axis=0).T


def evaluate():
    selection=json.loads((HERE/"selection.json").read_text())
    old=int(np.where(SNAP==.55)[0][0])
    early=int(np.where(SNAP==.24)[0][0])
    r={"reference":old,"early":early,"selection":selection,"times":SNAP.tolist(),"domains":{},"comparisons":{}}
    contrast={s:[] for s in TEST_SEEDS}
    contrast_names=[]
    for domain in ("seen","strong"):
        batches=[load_results(f"{domain}_{s}") for s in TEST_SEEDS]
        possible=np.concatenate([v for _,v in batches])
        data={"total":len(possible),"reachable":int(possible.sum()),"methods":{}}
        static=np.concatenate([b["static"] for b,_ in batches],axis=-1)
        data["no_probe"]=float(static.mean())
        for m in ("prior","direct"):
            hits=np.concatenate([b[m] for b,_ in batches],axis=-1)
            rates=hits.mean((0,2))
            data["methods"][m]={
                "rates":rates.tolist(),"conditional":hits[:,:,possible].mean((0,2)).tolist(),
                "random":float(rates.mean()),"posthoc_best":int(hits.sum((0,2)).argmax()),
                "per_training_set":hits.mean(-1).tolist(),
                "per_test_block":[b[m].mean((0,2)).tolist() for b,_ in batches]}
        for rule,j in (("all",selection["prior"]["all"]),("capped",selection["prior"]["capped"]),("fixed024",early)):
            name=f"{domain}_{rule}"
            contrast_names.append(name)
            for s,(b,_) in zip(TEST_SEEDS,batches):
                h=b["prior"].mean(0)
                contrast[s].append(h[j]-h[old])
            r["comparisons"][name]={"time":float(SNAP[j])}
        r["domains"][domain]=data
    mat=[np.column_stack(cols) for cols in contrast.values()]
    bounds=intervals(mat)
    for name,delta,(lo,hi) in zip(contrast_names,np.concatenate(mat).mean(0),bounds):
        row=r["comparisons"][name]
        row.update(delta=float(delta),ci=[float(lo),float(hi)],
                   improvement=bool(delta>=.03 and lo>0),
                   faster_noninferior=bool(.55-row["time"]>=.1-1e-10 and lo>-.02))
    h,_=load_results("validation")
    for m in ("prior","direct"):
        counts=h[m].sum((0,2))
        assert counts.tolist()==selection[m]["validation_counts"]
        assert counts.argmax()==selection[m]["all"]
        assert counts[SNAP<=.55].argmax()==selection[m]["capped"]
    return r


if __name__=="__main__":
    r=evaluate()
    (HERE/"results.json").write_text(json.dumps(r,indent=2),encoding="utf8")
    print(r["comparisons"])
