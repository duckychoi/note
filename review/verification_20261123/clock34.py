"""Sensitivity of the existing medium-trajectory diagnostic to which clock labels the gait."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"priority_20261122"))
from rest_reference import scores,window
from phase_util import BPhaseFlow,PhaseRef,NQ


def main(root):
    diagnostic = json.loads((HERE.parents[1]/"results/v2/diag_rollout_predict.json").read_text())
    out = {}
    for env in ("HalfCheetah-v5","Hopper-v5","Walker2d-v5"):
        name = env.split("-")[0].lower()
        folder = root/"minari/mujoco"/name
        with h5py.File(folder/"expert-v0/data/main_data.hdf5") as f:
            eligible = [i for i in range(len(f)) if len(f[f"episode_{i}/rewards"])>=999]
            ids = np.random.default_rng(0).permutation(eligible)[:10]
            tr = [f[f"episode_{i}/observations"][:] for i in ids]
        with h5py.File(folder/"medium-v0/data/main_data.hdf5") as f:
            ids = [i for i in np.random.default_rng(1).permutation(len(f)) if len(f[f"episode_{i}/rewards"])>=300][:20]
            test = [f[f"episode_{i}/observations"][:] for i in ids]
        clean = np.concatenate([o[:-1] for o in tr]).astype("float32")
        mu = np.tile(clean.mean(0),2)
        sd = np.tile(clean.std(0)+.001,2)
        ref = PhaseRef(env,np.concatenate([x[:,1:NQ[env]] for x in tr]))
        omega = np.mean([(np.unwrap(ref.phase(x))[-50]-np.unwrap(ref.phase(x))[50])/(len(x)-100) for x in tr])
        prior = BPhaseFlow(1,len(mu),"film4")
        prior.load_state_dict(torch.load(root/"cache"/f"{env}_n10_phaseflow_film4.pt",map_location="cpu",weights_only=True))
        saved = diagnostic[env]["film4"]
        stand = saved["median_return"][saved["best_stand"]]
        returns = {"demo_phase":[],"external_phase":[]}
        discount = .99**np.arange(300)
        for x in test:
            w = window(x[:301])
            post = scores(prior.eval(),(w,ref.phase(x)[:300]),mu,sd)
            returns["demo_phase"].append(float(discount@post))
            trial = []
            for offset in 2*np.pi*(np.arange(4)+.125)/4:
                r = scores(prior,(w,offset+omega*np.arange(300)),mu,sd)
                trial.append(float(discount@r))
            returns["external_phase"].append(trial)
        # Integrate the four clock offsets within an episode before comparing episodes.
        blind = np.mean(returns["external_phase"],axis=1)
        out[env] = dict(returns=returns,ids=[int(i) for i in ids],stored_stand=stand,original_D=saved["D_medium"],
                        demo_D=float(np.mean(np.array(returns["demo_phase"])>stand)),
                        external_D=float(np.mean(blind>stand)))
    return out


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("E:/0.Project/fm34_env"))
    torch.set_num_threads(2)
    (HERE/"clock34.json").write_text(json.dumps(main(ap.parse_args().root),indent=2,allow_nan=False)+"\n",encoding="utf-8")
