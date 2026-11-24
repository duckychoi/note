"""Check an intrinsic order-sensitive rhythm score on held-out Minari trajectories."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent


def clips(root, env, q, joints):
    path = root/"minari/mujoco"/env/(q+"-v0")/"data/main_data.hdf5"
    with h5py.File(path) as f:
        eligible = [i for i in range(len(f)) if len(f[f"episode_{i}/rewards"]) >= (999 if q=="expert" else 384)]
        ids = np.random.default_rng(0 if q=="expert" else 211).permutation(eligible)
        ids = ids[:84] if q=="expert" else ids[:64]
        return np.stack([f[f"episode_{i}/observations"][128:384,2:2+joints] for i in ids]).astype(float),ids


def features(x):
    x = x-x.mean(1,keepdims=True)
    z = np.fft.rfft(x,axis=1)[:,1:]/x.shape[1]
    z = z.reshape(len(x),8,z.shape[1]//8,z.shape[-1])
    cross = np.einsum("nbfi,nbfj->nbij",z,z.conj())
    lag = np.einsum("nti,ntj->nij",x,np.roll(x,-1,axis=1))/x.shape[1]
    return dict(psd=np.diagonal(cross.real,axis1=-2,axis2=-1),zero_lag=np.einsum("nti,ntj->nij",x,x)/x.shape[1],
                signed_area=lag-lag.transpose(0,2,1),cross_spectrum=cross)


def distances(x,refs):
    return {k:np.linalg.norm((v-refs[k]).reshape(len(x),-1),axis=1)/np.linalg.norm(refs[k])
            for k,v in features(x).items()}


def audit(root,env,nj):
    expert,ids = clips(root,env,"expert",nj)
    sd = np.maximum(expert[:10].std(axis=(0,1)),.001)
    train,clean = expert[:10]/sd,expert[20:]/sd
    other = {q:clips(root,env,q,nj)[0]/sd for q in ("simple","medium")}
    out = dict(train_ids=ids[:10].tolist(),test_ids=ids[20:].tolist(),lengths={})
    for length in (32,128,256):
        x = clean[:,:length]
        rng = np.random.default_rng(721)
        shifts = rng.integers(0,length,size=(len(x),nj))
        split = np.stack([np.column_stack([np.roll(u[:,k],s[k]) for k in range(nj)]) for u,s in zip(x,shifts)])
        groups = dict(clean=x,common_shift=np.roll(x,7,axis=1),independent_shift=split,reverse=x[:,::-1],
                      **{q:u[:,:length] for q,u in other.items()})
        refs = {k:v.mean(0) for k,v in features(train[:,:length]).items()}
        scores = {g:distances(u,refs) for g,u in groups.items()}
        by_metric = {}
        for metric,d in scores["clean"].items():
            row = {}
            for g,vals in scores.items():
                v = vals[metric]
                if g in ("common_shift","independent_shift","reverse"):
                    diff = v-d
                    row[g] = dict(paired_order=float(np.mean((diff>1e-9)+.5*(np.abs(diff)<=1e-9))),
                                  median_difference=float(np.median(diff)))
                elif g in other:
                    ranks = rankdata(np.r_[v,d])
                    auc = (ranks[:len(v)].sum()-len(v)*(len(v)+1)/2)/(len(d)*len(v))
                    row[g] = dict(expert_closer_auc=float(auc),median_distance=float(np.median(v)),n=len(v))
                else:
                    row[g] = dict(median_distance=float(np.median(d)),n=len(d))
            by_metric[metric] = row
        out["lengths"][length] = by_metric
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("E:/0.Project/fm34_env"))
    root = parser.parse_args().root
    data = {e:audit(root,e,n) for e,n in [("halfcheetah",6),("hopper",3),("walker2d",6)]}
    (HERE/"coordination34.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
