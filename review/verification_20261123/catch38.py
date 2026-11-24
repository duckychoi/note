"""Same per-frame forecasts, different path forecasts for a simulated catch decision."""
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from bounce36 import event_bank
from contact36 import SIGMA, model, trajectories

HERE = Path(__file__).resolve().parent
WINDOWS = np.array([np.arange(t,t+3) for t in (15,18,21)])
SAFE = np.array([.1,.3,.5,.7,.9])


def in_band(y):
    return (y>=.10)&(y<=.23)


def probabilities(bank,obs):
    prefix = bank[:,:12]
    logits = (obs[:,:12]@prefix.T - .5*np.square(prefix).sum(1))/(4*SIGMA**2)
    w = softmax(logits,axis=1)
    capture = in_band(bank[:,WINDOWS])
    marginal = (w@capture.reshape(len(bank),-1)).reshape(len(obs),3,3)
    joint = w@capture.all(2)
    iid = marginal.prod(2)
    neighbors = np.argpartition(logits,-32,axis=1)[:,-32:]
    estimates = dict(joint=joint,independent=iid, nearest32=capture.all(2)[neighbors].mean(1),
                     point=in_band((w@bank)[:,WINDOWS]).all(2).astype(float))
    direct_one = w@capture[:,:,0]
    return estimates,float(np.max(abs(direct_one-marginal[:,:,0])))


def decision(p,actual):
    action = p.argmax(1)
    best_p = p.max(1)
    outcome = actual[np.arange(len(actual)),action]
    value = np.where(best_p[:,None]>SAFE,outcome[:,None],SAFE)
    return dict(value=float(value.mean()),value_by_safe=value.mean(0).tolist(),
                brier=float(np.square(p-actual).mean()))


def evaluate(bank,y,obs):
    estimates,diff = probabilities(bank,obs)
    actual = in_band(y[:,WINDOWS]).all(2)
    result = {}
    for method,p in estimates.items():
        result[method] = decision(p,actual)
    # Both aggregators use *identical* one-frame probabilities, even at finite sample size.
    return dict(methods=result,one_frame_diff=diff,
                event_rate=actual.mean(0).tolist())


def main():
    train = [model(x) for x in (.18,.4,.75)]
    test = [model(x) for x in (.12,.55,.95)]
    full = event_bank()
    banks = dict(analytic=full[np.random.default_rng(71000).choice(len(full),2048,replace=False)],
                 retrieval=trajectories(train,71001,n=2048)[0])
    rows = []
    for seed in range(71200,71224):
        y,obs,_ = trajectories(test,seed,n=192)
        rows.append(dict(seed=seed,methods={name:evaluate(b,y,obs) for name,b in banks.items()}))
    return dict(windows=WINDOWS.tolist(),safe_values=SAFE.tolist(),band_m=[.10,.23],seeds=rows)


if __name__ == "__main__":
    (HERE/"catch38.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
