"""Compare the stationary reference with actual 300k Walker2d policy rollouts."""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata
import torch

from phase_null import make_env, FootTD, NQ
from phase_util import PhaseRef, BPhaseFlow
from walker_control import phase_actor


def episodes(root, environment, quality, ids=None):
    path = root / "minari/mujoco" / environment / f"{quality}-v0/data/main_data.hdf5"
    with h5py.File(path) as data:
        if ids is None:
            ids = [i for i in range(len(data)) if len(data[f"episode_{i}"]["rewards"]) >= 999]
            ids = np.random.default_rng(0).permutation(ids)[:10]
        return [data[f"episode_{i}"]["observations"][:] for i in ids]


def window(obs):
    return np.concatenate((obs[:-1], obs[1:]), axis=1)


@torch.inference_mode()
def rollouts(name, policy, period):
    env = make_env(name, train=policy is None)
    clips, phases = [], []
    for i in range(10):
        obs = env.reset(seed=20_000+i)[0]
        rows = [obs]
        clock = 2*np.pi*((i*.37)%1.) + (2*np.pi/period)*np.arange(1001)
        for step in range(250 if policy is None else 1000):
            if policy is None:
                action = np.zeros(env.action_space.shape)
            else:
                angle = clock[step]
                state = np.concatenate((obs, [np.cos(angle), np.sin(angle)]))
                action = policy(torch.from_numpy(state.astype("float32"))).numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            rows.append(obs)
            if terminated or truncated:
                break
        clips.append(window(np.stack(rows[50:]))[::5])
        phases.append(clock[50:len(rows)-1:5])
    observations = np.concatenate(clips)
    if policy is None:
        return observations, np.random.default_rng(81).uniform(-np.pi,np.pi,len(observations))
    return observations, np.concatenate(phases)


@torch.inference_mode()
def scores(model, group, mean, std):
    obs, phase = group
    generator = torch.Generator().manual_seed(781)
    x = torch.tensor((obs - mean) / std, dtype=torch.float32)[None]
    phi = torch.tensor(phase, dtype=torch.float32)[None]
    outputs = []
    lid = math.log(1000.)
    for offset in range(0, x.shape[1], 256):
        batch = x[:, offset:offset+256].repeat(1,32,1)
        labels = phi[:, offset:offset+256].repeat(1,32)
        time = -torch.expm1(-lid * torch.rand(labels.shape, generator=generator))
        noise = torch.randn(batch.shape, generator=generator)
        y = (1-time[...,None])*noise + time[...,None]*batch
        error = (model(y,time,labels) - (batch-noise)).square().sum(-1)
        outputs.append(-(error*lid*time).view(32,-1).mean(0).numpy())
    return np.concatenate(outputs)


def auc(left, right):
    ranks = rankdata(np.concatenate((left,right))).astype("float64")
    return float((ranks[:len(left)].sum() - len(left)*(len(left)+1)/2)/(len(left)*len(right)))


def contacts(root, env_id, folder):
    import mujoco

    env = make_env(env_id, False)
    env.reset(seed=0)
    detector = FootTD(env, env_id)
    model, data = env.unwrapped.model, env.unwrapped.data
    world_geoms = set(np.flatnonzero(model.geom_bodyid == 0).tolist())
    height_events = np.zeros(len(detector.gi), dtype=int)
    contact_events = np.zeros_like(height_events)
    gaps = [[] for _ in detector.gi]
    for obs in episodes(root, folder, "expert"):
        reconstructed_height = []
        touches = []
        for x in obs:
            data.qpos[:] = np.concatenate(([0.], x[:NQ[env_id]]))
            data.qvel[:] = x[NQ[env_id]:]
            mujoco.mj_forward(model, data)
            reconstructed_height.append(detector.z_live(data))
            contacts_now = [(int(c.geom1), int(c.geom2)) for c in data.contact]
            touches.append([any((a == foot and b in world_geoms) or (b == foot and a in world_geoms)
                                for a,b in contacts_now) for foot in detector.gi])
        flags = np.asarray(touches)
        height_events += [len(x) for x in detector.events(np.asarray(reconstructed_height))]
        for i in range(len(detector.gi)):
            events = np.flatnonzero(flags[1:, i] & ~flags[:-1, i]) + 1
            contact_events[i] += len(events)
            gaps[i].extend(np.diff(events[events >= 50]).tolist())
    env.close()
    return dict(height_threshold_count=height_events.tolist(), floor_contact_onsets=contact_events.tolist(),
                floor_contact_median_gap=[float(np.median(g)) if g else None for g in gaps])


def audit(root):
    name = "Walker2d-v5"
    training = episodes(root, "walker2d", "expert")
    all_obs = np.concatenate([x[:-1] for x in training]).astype("float32")
    mean = np.tile(all_obs.mean(0),2)
    std = np.tile(all_obs.std(0)+.001,2)
    ref = PhaseRef(name, np.concatenate([x[:, 1:NQ[name]] for x in training]))
    simple = episodes(root, "walker2d", "simple", np.random.default_rng(70).choice(100,10,replace=False))
    gait = (np.concatenate([window(x)[50::5] for x in simple]),
            np.concatenate([ref.phase(x)[50:-1:5] for x in simple]))
    phase_dir = root / "phase2"
    config = json.loads((phase_dir / f"{name}_pilot300_none_like_state_s0.json").read_text())
    period = config["period"]
    checkpoint = torch.load(phase_dir / f"{name}_pilot300_actors.pt", map_location="cpu", weights_only=True)
    groups = dict(simple=gait, zero_action=rollouts(name, None, period))
    for i,(method,seed) in enumerate(checkpoint["members"]):
        if method in ("none_like_state","film4_like_state"):
            policy = phase_actor(checkpoint["actor"], i)[0]
            groups[f"{method}_s{seed}"] = rollouts(name, policy, period)
    result = dict(groups={k:dict(windows=len(v[0]), height_median=float(np.median(v[0][:,0])),
                                vx_mean=float(v[0][:,NQ[name]].mean())) for k,v in groups.items()})
    result["simple_over_reference_auc"] = {}
    for cond in ("none","film4"):
        model = BPhaseFlow(1,34,cond)
        model.load_state_dict(torch.load(root / "cache" / f"{name}_n10_phaseflow_{cond}.pt",
                                         map_location="cpu",weights_only=True))
        values = {k:scores(model.eval(), v, mean, std) for k,v in groups.items()}
        result["simple_over_reference_auc"][cond] = {k:auc(values["simple"], v)
                                                    for k,v in values.items() if k!="simple"}
    result["halfcheetah_contacts"] = contacts(root,"HalfCheetah-v5","halfcheetah")
    result["walker2d_contacts"] = contacts(root,name,"walker2d")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/0.Project/fm34_env"))
    parser.add_argument("--output", type=Path, required=True)
    config = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(config.root)
    config.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
