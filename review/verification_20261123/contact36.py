"""Test the fixed single-impulse model on noisy heights from MuJoCo sphere/plane contacts."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from physics36 import fit_contact, T, TIME

HERE = Path(__file__).resolve().parent
DT = .02
SIGMA = .015
BASE = -.5*9.81*(TIME*DT)**2


def model(damping):
    xml = f"""<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
    <default><geom solref=".015 {damping}" solimp=".95 .99 .001"/></default>
    <worldbody><geom type="plane" size="2 2 .1"/><body><joint type="slide" axis="0 0 1"/>
    <geom type="sphere" size=".06" mass=".25"/></body></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def trajectories(models, seed, n=96):
    rng = np.random.default_rng(seed)
    y = np.empty((n,T))
    first = np.full(n,-1)
    for i in range(n):
        m = models[rng.integers(len(models))]
        data = mujoco.MjData(m)
        data.qpos[0] = rng.uniform(.38,1.65)
        data.qvel[0] = rng.uniform(-1.,.35)
        y[i,0] = data.qpos[0]
        for t in range(1,T):
            for _ in range(10):
                mujoco.mj_step(m,data)
                if first[i]<0 and data.ncon:
                    first[i] = t
            y[i,t] = data.qpos[0]
    return y, y+rng.normal(0,SIGMA,y.shape), first


def rmse(x,y):
    return float(np.sqrt(np.mean(np.square(x-y))))


def main():
    models = [model(x) for x in (.18,.40,.75)]
    val = [trajectories(models,s) for s in (21000,21001)]
    widths = (5,7,9,11)
    selected = min(widths,key=lambda w:np.mean([rmse(y,savgol_filter(obs,w,2)) for y,obs,_ in val]))
    rows = []
    for seed in range(21200,21212):
        y,obs,first = trajectories(models,seed)
        hard,kink,onset = fit_contact(obs,BASE,SIGMA)
        hit = first>=0
        est = dict(raw=obs,hard=hard,impulse=kink,savgol=savgol_filter(obs,selected,2))
        rows.append(dict(seed=seed,contacts=int(hit.sum()),n=len(hit),
            errors={k:dict(all=rmse(x,y),contact=rmse(x[hit],y[hit]),
                           free_flight=rmse(x[~hit],y[~hit])) for k,x in est.items()},
            onset_within_one=float(np.mean((onset[hit]>=0)&(np.abs(onset[hit]-first[hit])<=1))),
            false_positive=float(np.mean(onset[~hit]>=0))))
    return dict(mujoco=mujoco.__version__,observation_dt=DT,noise_m=SIGMA,damping=[.18,.40,.75],
                validation_seeds=[21000,21001],savgol_candidates=widths,savgol_selected=selected,rows=rows)


if __name__ == "__main__":
    (HERE/"contact36.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
