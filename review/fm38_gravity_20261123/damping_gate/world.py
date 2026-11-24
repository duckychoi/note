"""Matched MuJoCo interventions. Change hinge viscosity only between test worlds."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from swing import CACHE, L, R, LEVELS, SPEED, PROBE_SPEED, build, observe

RATES={"clean":np.zeros(9),"seen":np.linspace(.6,1.4,9),"strong":np.linspace(2.,2.8,9)}
INERTIA=.2*(L*L+.4*R*R)


def generate(split,domain,n,seed):
    rng=np.random.default_rng(seed)
    starts=np.deg2rad(rng.uniform(-8,8,n))
    goals=np.deg2rad(rng.uniform(32,91,n))
    inclines=rng.integers(0,9,n)
    rates=rng.integers(0,9,n)
    logged=rng.integers(0,4,n)
    models={}
    rows=[]
    for i,(k,c) in enumerate(zip(inclines,rates)):
        key=(k,c)
        if key not in models:
            model=build(LEVELS["source"][k])
            model.dof_damping[0]=INERTIA*RATES[domain][c]
            models[key]=model
        model=models[key]
        probe=observe(model,starts[i],PROBE_SPEED,True)
        for a in ([logged[i]] if split=="train" else range(4)):
            q,peak=observe(model,starts[i],SPEED[a])
            rows.append((i,a,goals[i],k,RATES[domain][c],peak,[starts[i],probe,*q]))
    keys=("state","action","goal","level","damping","peak","angles")
    return {key:np.asarray(col) for key,col in zip(keys,zip(*rows))}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--angle-holdout",action="store_true")
    options=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    specs=(("train","seen",2048,53801),("val","seen",192,53802),
                                ("seen","seen",512,53803),("strong","strong",512,53803),
                                ("clean","clean",512,53803))
    if options.angle_holdout:
        specs=(("angle_holdout","strong",512,53819),)
    for split,domain,n,seed in specs:
        data=generate(split,domain,n,seed)
        np.savez_compressed(CACHE/f"damp_{split}.npz",**data)
        print(split,len(data["peak"]),flush=True)
