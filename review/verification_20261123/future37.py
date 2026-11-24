"""Evaluate conditional samples on the full distribution using a matched training set."""
import contextlib
import io
import json
from pathlib import Path
import runpy

import numpy as np
from scipy.special import expit,ndtri
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]/"research-sources/generative-jepa/src/toy_generative_jepa.py"
with contextlib.redirect_stdout(io.StringIO()):
    orig = runpy.run_path(str(SRC))


def probability(c, varying):
    return expit(2.5*c) if varying else np.full_like(c,0.5,dtype=float)


def target(c,p,n=8192):
    u = (np.arange(n)+.5)/n
    left = u<1-p
    return np.r_[c-2+.18*ndtri(u[left]/(1-p)), c+2+.18*ndtri((u[~left]-(1-p))/p)]


def run(seed, varying):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-2,2,3000)
    z = c+np.where(rng.random(len(c))<probability(c,varying),2.,-2.)+.18*rng.standard_normal(len(c))
    z0 = rng.standard_normal(len(c))
    flow = orig["flow_sample"]
    flow.__globals__.update(C=c,Z=z,Z0=z0,U=z-z0,rng=np.random.default_rng(seed+1000))
    query = np.linspace(-1.5,1.5,5)
    rows = []
    for x in query:
        p = float(probability(x,varying))
        actual = target(x,p)
        w = np.exp(-.5*((c-x)/.25)**2)
        w /= w.sum()
        mean = float(w@z)
        dev = np.sqrt(w@(z-mean)**2)
        n = 384
        draws = dict(point=np.full(n,mean), gaussian=rng.normal(mean,dev,n),
                     conditional_resample=rng.choice(z,n,p=w), original_flow=flow(np.full(n,x)))
        for key,y in draws.items():
            rows.append(dict(context=float(x), method=key, w1=float(wasserstein_distance(actual,y)),
                             right_mass_error=float(abs(np.mean(y>x)-p)),
                             near_mode=float(np.mean(np.abs(np.abs(y-x)-2)<.6))))
    return rows


if __name__ == "__main__":
    data = {name:{seed:run(seed,varying) for seed in range(10120,10132)}
            for name,varying in [("balanced",False),("varying",True)]}
    (HERE/"future37.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n", encoding="utf-8")
