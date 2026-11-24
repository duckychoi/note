"""Exact k-window control counterexample, and finite-data recovery of the missing moment."""
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def exact(k, noise):
    states = np.array(list(itertools.product((-1,1),repeat=k)))
    ids = np.arange(2**k)
    law = {}
    for world,action in itertools.product((-1,1),repeat=2):
        parity = world*action*np.prod(states,axis=1)
        next_mass = np.zeros(2**k)
        success = 0.
        for flip,prob in [(1,1-noise),(-1,noise)]:
            last = flip*parity
            # Drop the oldest bit, append the generated one.
            dest = ((2*ids)%(2**k)) + (last>0)
            np.add.at(next_mass,dest,prob/len(ids))
            success += prob*np.mean(np.prod(states,axis=1)*last>0)
        law[f"{world}/{action}"] = dict(max_k_window_error=float(abs(next_mass-1/len(ids)).max()),
                                        path_success=float(success))
    return law


def estimation(k,noise,seed):
    rng = np.random.default_rng(seed)
    out = []
    for world in (-1,1):
        # Different episodes yield independent labeled windows for the estimate.
        start = rng.choice((-1,1),size=(32,k))
        actions = rng.choice((-1,1),size=32)
        eta = np.where(rng.random(32)<noise,-1,1)
        last = np.prod(start,axis=1)*world*actions*eta
        observed_success = np.prod(start,axis=1)*last>0
        estimate = np.mean(actions*(2*observed_success-1))
        chosen = 1 if estimate>=0 else -1
        success = 1-noise if chosen==world else noise
        out.append(dict(world=world,estimate=float(estimate),chosen_action=chosen,
                        achieved=float(success),regret=float((1-noise)-success)))
    return out


if __name__ == "__main__":
    results = {}
    for k in (2,4,8):
        for noise in (.1,.35):
            results[f"k{k}/noise{noise}"] = dict(exact=exact(k,noise),seeds={
                s:estimation(k,noise,s) for s in range(61200,61212)})
    (HERE/"horizon38.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
