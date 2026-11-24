"""Replay saved primary proposals on untouched cases against retrieval and two PF temperature choices."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PF_NTH", "12")
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE))
import pf36_g3 as g
import pf36_g4 as retrieval
import pf36_g4b as flow

SEEDS = tuple(range(56370, 56378))
METRICS = ("energy", "cover_x", "cover_z", "brier_bounce2", "brier_xpast")


def read(name):
    return json.loads((SOURCE/name).read_text(encoding="utf8"))


def reference(cases, temp, threshold, rng):
    parts = []
    for i in range(0, len(cases["mask"]), 12):
        subset = {k: v[i:i+12] for k, v in cases.items()}
        parts.append(g.run_pf_full(subset, 8192, temp, threshold, rng+i))
    return {k: np.concatenate([p[k] for p in parts]) for k in METRICS}


def main():
    torch.set_num_threads(4)
    cfg = read("gate3.json")["config"]
    threshold = cfg["xthr"]
    old_temp = cfg["temps"]["pf"]
    val = g.make_cases(51000, 48, test=False)
    scores = {t: float(reference(val,t,threshold,66100)["energy"].mean()) for t in (1,2,4)}
    chosen = min(scores, key=scores.get)
    (HERE/"pf_selection.json").write_text(json.dumps(dict(validation=scores, original=old_temp,
        chosen=chosen), indent=2), encoding="utf8")
    print("PF temperature locked", scores, chosen, flush=True)
    lib = g.library(g.POOL, 12, 12, 196608, 52003)
    runs = [r for r in read("gate5.json")["runs"] if r["role"] == "primary"]
    weights = torch.load(SOURCE/"gate5_primary_models.pt", map_location="cpu", weights_only=True)
    models = []
    for run in runs:
        model = flow.Proposal()
        model.load_state_dict(weights[f"{run['M']}_{run['train_seed']}"])
        models.append(model.eval())
    for i, seed in enumerate(SEEDS):
        case = g.make_cases(seed, 24, test=True)
        raw = {f"case_{key}": a for key, a in case.items()}
        lw, _ = retrieval.adaptive_lw(g.lib_logw(lib[0],case,1.),8)
        results = {"retrieval": retrieval.score_from_lw(lib,lw,case,threshold,66300+i)}
        original = reference(case,old_temp,threshold,66500+i*20)
        results["original"] = original
        results["tuned"] = original if old_temp == chosen else reference(case,chosen,threshold,66500+i*20)
        for run, model in zip(runs, models):
            n = run["N"]
            init = flow.proposal_init(model,case,n,0)
            results[f"flow_{run['train_seed']}"] = g.run_pf_full(case,n,run["pf_temp"],threshold,
                66900+i,init=init,resample_after=g.CUT-1)
        for method, values in results.items():
            raw.update({f"{method}_{k}": values[k] for k in METRICS})
        np.savez_compressed(HERE/f"fresh_{seed}.npz", **raw)
        print("saved", seed, flush=True)


if __name__ == "__main__":
    main()
