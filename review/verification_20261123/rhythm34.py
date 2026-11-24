"""Extract the periodic part of each frozen policy, then check motion on fresh resets."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from motion34 import phase_actor

HERE = Path(__file__).resolve().parent


def harmonics(phase):
    angular = phase[:,None]*np.arange(1,5)
    return np.column_stack((np.ones(len(phase)),np.cos(angular),np.sin(angular)))


@torch.inference_mode()
def roll(envs,period,reset_base,policy=None,coeff=None):
    obs = np.stack([e.reset(seed=reset_base+i)[0] for i,e in enumerate(envs)])
    start = np.array([e.unwrapped.data.qpos[0] for e in envs])
    phase0 = np.random.default_rng(reset_base).uniform(0,2*np.pi,len(envs))
    actions,angles,joints = [],[],[]
    for t in range(500):
        phi = phase0 + t*2*np.pi/period
        if coeff is None:
            inputs = np.column_stack((obs,np.cos(phi),np.sin(phi))).astype("float32")
            act = policy(torch.from_numpy(inputs)).numpy()
        else:
            act = np.clip(harmonics(phi)@coeff,-1,1)
        actions.append(act)
        angles.append(phi)
        joints.append(obs[:,2:8])
        obs = np.stack([e.step(a)[0] for e,a in zip(envs,act)])
    final = np.array([e.unwrapped.data.qpos[0] for e in envs])
    return ((final-start)/(500*envs[0].unwrapped.dt), np.stack(joints,axis=1),
            np.array(angles),np.array(actions))


def descriptor(batch):
    # Compare phase-invariant joint spectra and cross-joint lag-one covariance.
    x = batch[:,128:384]
    x = x-x.mean(1,keepdims=True)
    fft = np.fft.rfft(x*np.hanning(256)[None,:,None],axis=1)[:,1:]
    bands = np.sqrt(np.square(abs(fft)).reshape(len(x),16,8,6).sum(2))
    gram = np.einsum("nti,ntj->nij",x[:,:-1],x[:,1:])/255
    return bands,gram


def style(x,refs):
    scores = {}
    for k,data,ref in zip(("spectrum","lagged_coordination"),descriptor(x),refs):
        scores[k] = (np.linalg.norm((data-ref).reshape(len(data),-1),axis=1)/np.linalg.norm(ref)).tolist()
    return scores


def main(root):
    with h5py.File(root/"minari/mujoco/halfcheetah/expert-v0/data/main_data.hdf5") as file:
        eligible = [i for i in range(len(file)) if len(file[f"episode_{i}/rewards"])>=999]
        order = np.random.default_rng(0).permutation(eligible)
        expert = np.stack([file[f"episode_{i}/observations"][:500,2:8] for i in order[20:68]])
    refs = [v.mean(0) for v in descriptor(expert[:24])]
    result = dict(expert_ids=order[20:68].tolist(),expert_test=style(expert[24:],refs),policies={})
    envs = [gym.make("HalfCheetah-v5") for _ in range(12)]
    try:
        for folder,tag in (("phase_hc","redesign"),("prereg","prereg")):
            prefix = root/folder/f"HalfCheetah-v5_{tag}"
            ckpt = torch.load(f"{prefix}_actors.pt",map_location="cpu",weights_only=True)
            for idx,(method,seed) in enumerate(ckpt["members"]):
                if method.endswith("rwin"):
                    continue
                meta = json.loads(Path(f"{prefix}_{method}_s{seed}.json").read_text())
                period = meta["period"]
                actor = phase_actor(ckpt["actor"],idx)[0]
                _,_,ph,act = roll(envs,period,81100,policy=actor)
                design = harmonics(ph[100:].ravel())
                a = act[100:].reshape(-1,6)
                coef = np.linalg.lstsq(design,a,rcond=None)[0]
                r2 = 1-np.square(a-design@coef).sum()/np.square(a-a.mean(0)).sum()
                entry = dict(action_r2=float(r2))
                for key in ("closed","open"):
                    v,x,_,_ = roll(envs,period,82100,policy=actor,coeff=coef if key=="open" else None)
                    entry[key] = dict(vx=v.tolist(),**style(x,refs))
                result["policies"][f"{method}/{seed}"] = entry
    finally:
        for env in envs:
            env.close()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("E:/0.Project/fm34_env"))
    torch.set_num_threads(1)
    (HERE/"rhythm34.json").write_text(json.dumps(main(ap.parse_args().root),indent=2,allow_nan=False)+"\n",encoding="utf-8")
