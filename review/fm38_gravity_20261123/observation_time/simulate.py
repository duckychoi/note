"""Take alternative snapshots along a single matched practice trajectory."""
import argparse
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"damping_gate"))
from world import INERTIA, RATES
from swing import DT, LEVELS, PROBE_SPEED, SPEED, build, observe

SNAP=np.array([.08,.16,.24,.32,.40,.48,.55,.64,.72,.88,1.04])
TRAIN_SEEDS=(58401,58402,58403)
TEST_SEEDS=(58411,58412)
CONFIRM_SEEDS=(58421,58422)


def practice(model,start):
    import mujoco
    data=mujoco.MjData(model)
    data.qpos[0]=start
    data.qvel[0]=PROBE_SPEED
    ticks=np.rint(SNAP/DT).astype(int)
    result=[]
    for steps in np.diff(ticks,prepend=0):
        mujoco.mj_step(model,data,nstep=int(steps))
        result.append(data.qpos[0])
    return result


def generate(seed,domain,n,training=False,grid=False):
    rng=np.random.default_rng(seed)
    start=np.deg2rad(rng.uniform(-8,8,n))
    goal=np.deg2rad(rng.uniform(32,91,n))
    level=rng.integers(0,9,n)
    drag=rng.integers(0,9,n)
    if grid:
        start,level,drag=np.meshgrid(np.deg2rad([-6.,0.,6.]),np.arange(9),np.arange(9),indexing="ij")
        start,level,drag=(x.ravel() for x in (start,level,drag))
        n=len(start)
        goal=np.zeros(n)  # Analysis evaluates every goal on each physical branch.
    action=rng.integers(0,4,(n,1)) if training else np.tile(np.arange(4),(n,1))
    models={}
    shots=np.empty((n,len(SNAP)))
    frames=np.empty((*action.shape,5))
    peak=np.empty(action.shape)
    for i,(k,c) in enumerate(zip(level,drag)):
        key=(k,c)
        if key not in models:
            m=build(LEVELS["source"][k])
            m.dof_damping[0]=INERTIA*RATES[domain][c]
            models[key]=m
        m=models[key]
        shots[i]=practice(m,start[i])
        for j,a in enumerate(action[i]):
            frames[i,j],peak[i,j]=observe(m,start[i],SPEED[a])
    return dict(start=start,goal=goal,level=level,damping=RATES[domain][drag],
                action=action,probe=shots,future=frames,peak=peak)


def save(name,**options):
    np.savez_compressed(HERE/f"{name}.npz",**generate(**options))
    print(name,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("development","test","grid","confirmation"))
    stage=p.parse_args().stage
    if stage=="development":
        for seed in TRAIN_SEEDS:
            save(f"train_{seed}",seed=seed,domain="seen",n=2048,training=True)
        save("validation",seed=58404,domain="seen",n=768)
    elif stage=="test":
        if not (HERE/"selection.json").exists():
            raise RuntimeError("Select the observation times on source validation first.")
        for seed in TEST_SEEDS:
            for domain in ("seen","strong"):
                save(f"{domain}_{seed}",seed=seed,domain=domain,n=1024)
    elif stage=="confirmation":
        if not (HERE/"strong_selection.json").exists():
            raise RuntimeError("Lock the timing selected on the initial shift data first.")
        for seed in CONFIRM_SEEDS:
            save(f"strong_{seed}",seed=seed,domain="strong",n=1024)
    else:
        for domain in ("seen","strong"):
            save(f"grid_{domain}",seed=0,domain=domain,n=243,grid=True)
