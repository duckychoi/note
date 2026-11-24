"""Which stochastic branch survives a one-dimensional representation budget?"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def data(rng,n,basis,amp,shared,random_actions):
    context = rng.choice((-1,1),n)
    a = rng.choice((-1,1),n) if random_actions else context*np.where(rng.random(n)<.90,1,-1)
    b = a*np.where(rng.random(n)<.7,1,-1)
    d1 = context*np.where(rng.random(n)<.95,1,-1)
    d2 = np.where(rng.random(n)<shared,d1,rng.choice((-1,1),n))
    def view(d):
        return (np.column_stack((b,amp*d))+.25*rng.normal(size=(n,2)))@basis.T
    return view(d1),view(d2),a,b


def whiten(x):
    eig,vec = np.linalg.eigh(np.cov(x.T))
    return (vec/np.sqrt(eig))@vec.T


def discriminant(x,label):
    return np.linalg.solve(np.cov(x.T),x[label==1].mean(0)-x[label==-1].mean(0))


def run(seed,amp,shared):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0,2*np.pi)
    basis = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    draw = lambda n,do:data(rng,n,basis,amp,shared,do)
    x,z,a,_ = draw(1024,False)
    xi,_,ai,_ = draw(1024,True)
    xc,_,_,bc = draw(64,True)
    xt,_,at,bt = draw(8192,True)
    wx,wz = whiten(x),whiten(z)
    cross = (x-x.mean(0)).T@(z-z.mean(0))/(len(x)-1)
    left,_,_ = np.linalg.svd(wx@cross@wz)
    methods = dict(pca=np.linalg.eigh(np.cov(x.T))[1][:,-1],two_view=wx@left[:,0],
                   logged_action=discriminant(x,a),randomized_action=discriminant(xi,ai),
                   supervised64=discriminant(xc,bc))
    out = {}
    for name,v in methods.items():
        # All unsupervised directions have a sign ambiguity. Use the same 64 labels to orient them.
        pos,neg = (xc[bc==1]@v).mean(),(xc[bc==-1]@v).mean()
        pred = np.where(((xt@v)-.5*(pos+neg))*np.sign(pos-neg)>0,1,-1)
        probs = [(pred[at==u]>0).mean() for u in (-1,1)]
        out[name] = dict(branch_accuracy=float(np.mean(pred==bt)),
                         action_probability_mae=float(np.mean(np.abs(np.asarray(probs)-[.3,.7]))))
    return out


if __name__ == "__main__":
    results = {f"amplitude{a}/shared{s}":{seed:run(seed,a,s) for seed in range(91200,91212)}
               for a in (1,4) for s in (0,1)}
    (HERE/"representation37.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
