"""Out-of-sample event calibration and a direct data baseline on the catch task."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit,logit

from catch38 import WINDOWS,decision,in_band,model,probabilities,trajectories

HERE = Path(__file__).resolve().parent


def log_odds(p):
    return logit(np.clip(p,.0001,.9999))


def fit(p,y):
    weights = []
    for x,target in zip(log_odds(p).T,y.T):
        def obj(v):
            diff = v-np.array([0.,1.])
            linear = v[0]+v[1]*x
            error = expit(linear)-target
            loss = np.sum(np.logaddexp(0.,linear)-target*linear) + .5*(diff@diff)
            return loss,np.array([error.sum(),error@x])+diff
        result = minimize(obj,[0.,1.],jac=True,bounds=[(None,None),(0.,None)],method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(result.message)
        weights.append(result.x)
    return np.array(weights).T


def labels(y):
    return in_band(y[:,WINDOWS]).all(2)


def main():
    train = [model(x) for x in (.18,.4,.75)]
    test = [model(x) for x in (.12,.55,.95)]
    bank = trajectories(train,81001,n=2048)[0]
    cal = {}
    for name,group in [("source",train),("target",test)]:
        y,x,_ = trajectories(group,81100,n=768)
        prob,_ = probabilities(bank,x)
        cal[name] = (x[:,:12],labels(y),{m:fit(prob[m],labels(y)) for m in ("joint","independent")})
    rows = []
    for seed in range(81200,81224):
        y,x,_ = trajectories(test,seed,n=192)
        actual = labels(y)
        raw,_ = probabilities(bank,x)
        values = {f"raw_{m}":decision(raw[m],actual) for m in ("joint","independent")}
        for name,(prefix,target,coefs) in cal.items():
            nn = np.argpartition(cdist(x[:,:12],prefix),31,axis=1)[:,:32]
            values[f"{name}_direct32"] = decision(target[nn].mean(1),actual)
            for m,(a,b) in coefs.items():
                values[f"{name}_{m}"] = decision(expit(a+b*log_odds(raw[m])),actual)
        rows.append(dict(seed=seed,values=values))
    return dict(training_library=2048,labeled_calibration_episodes=768,realized_safe_value=.5,rows=rows)


if __name__ == "__main__":
    (HERE/"calibration38.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
