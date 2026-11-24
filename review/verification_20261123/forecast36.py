"""Generate the unobserved second half of a drop; compare same-size trajectory libraries."""
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from bounce36 import event_bank
from contact36 import SIGMA, model, trajectories

HERE = Path(__file__).resolve().parent
CUT = 12


def score(bank,y,obs,temp,seed):
    prefix = bank[:,:CUT]
    loglik = (obs[:,:CUT]@prefix.T - .5*np.square(prefix).sum(1))/SIGMA**2
    w = softmax(loglik/temp,axis=1)
    suffix = bank[:,CUT:]
    future = y[:,CUT:]
    event = future[:,-1]>.20
    pred = w@(suffix[:,-1]>.20)
    cdf = w.cumsum(1)
    rng = np.random.default_rng(seed)
    samples = np.stack([suffix[np.searchsorted(row,rng.uniform(size=128)).clip(max=len(bank)-1)] for row in cdf])
    # Independent sample pairs estimate the population energy score (distance in metres).
    distance = lambda a,b: np.sqrt(np.square(a-b).mean(axis=-1))
    energy = (distance(samples,future[:,None]).mean(1) -
              .5*distance(samples[:,:64],samples[:,64:]).mean(1))
    mode = suffix[loglik.argmax(1)]
    return dict(trajectory_energy=float(energy.mean()), brier=float(np.mean(np.square(pred-event))),
                mean_rmse=float(np.sqrt(np.square(w@suffix-future).mean())),
                map_brier=float(np.mean((mode[:,-1]>.20)!=event)),
                map_energy=float(distance(mode,future).mean()),event_rate=float(event.mean()))


def main():
    material = {k:[model(x) for x in damp] for k,damp in
                [("within_materials",(.18,.4,.75)),("new_materials",(.12,.55,.95))]}
    rng = np.random.default_rng(41000)
    full = event_bank()
    bank = {"analytic_impacts":full[rng.choice(len(full),2048,replace=False)],
            "simulator_retrieval":trajectories(material["within_materials"],41001,n=2048)[0]}
    val = [trajectories(material["within_materials"],s) for s in (41100,41101)]
    temp = {k:min((.5,1.,2.,4.),key=lambda t:np.mean(
        [score(b,y,obs,t,41190+i)["trajectory_energy"] for i,(y,obs,_) in enumerate(val)]))
            for k,b in bank.items()}
    rows = []
    for name,models in material.items():
        for seed in range(41200,41212):
            y,obs,_ = trajectories(models,seed)
            rows.append(dict(seed=seed,materials=name,methods={
                k:score(b,y,obs,temp[k],seed+1000) for k,b in bank.items()}))
    return dict(observed=CUT,unseen=12,threshold_m=0.20,library_size=2048,temperature=temp,results=rows)


if __name__ == "__main__":
    (HERE/"forecast36.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
