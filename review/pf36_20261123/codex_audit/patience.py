"""Matched ablation of the originally failed budget/seed, leaving the fresh holdout untouched."""
import json

from fresh import HERE, g, np, read
import pf36_g5 as original


def run():
    m, n, seed = 98304, 512, 61098
    lib = g.library(g.POOL,12,12,196608,52003)
    val = g.make_cases(51000,48,test=False)
    threshold = read("gate3.json")["config"]["xthr"]
    logs = []
    for patience in (5,15):
        cfg = dict(original.OLD,patience=patience,name=f"patience_{patience}")
        model, history = original.train5(lib,m,seed,cfg)
        val_score = {t:float(original.pf_eval(model,val,n,t,threshold,3)["energy"].mean()) for t in (1,2,4)}
        temp = min(val_score,key=val_score.get)
        raw = {}
        for i in range(4):
            case = g.make_cases(51200+i,48,test=True)
            values = original.pf_eval(model,case,n,temp,threshold,500+n+i)
            raw.update({f"{i}_{k}":v for k,v in values.items()})
        path = HERE/f"patience_{patience}.npz"
        np.savez_compressed(path,**raw)
        logs.append(dict(patience=patience, training_seed=seed,temperature=temp, val_energy=val_score,
                         energy=float(np.mean([raw[f"{i}_energy"] for i in range(4)])), **history))
        (HERE/"patience.json").write_text(json.dumps(logs,indent=2),encoding="utf8")
        print(patience,history["epochs"],history["best_val_nll"],logs[-1]["energy"],flush=True)
    shorter,longer=logs
    assert shorter["curve"]==longer["curve"][:len(shorter["curve"])]


if __name__=="__main__":
    run()
