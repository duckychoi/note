"""Train a clock-blind behavior-cloning control and actually roll it out in Walker2d."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from phase_null import assess, make_env, FootTD, NQ, wrap


FARAMA_SHA = "b7c5ca415c5da9538c3a6b50ee4de44350f120ddc41697c41a622f9bb5b2ce3b"


def public_actor(path):
    if sha256(path.read_bytes()).hexdigest() != FARAMA_SHA:
        raise ValueError("Unexpected Farama checkpoint contents")
    state = torch.load(path, weights_only=True, map_location="cpu")
    net = nn.Sequential(nn.Linear(17,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6),nn.Tanh())
    for layer, key in ((0,"latent_pi.0"),(2,"latent_pi.2"),(4,"mu")):
        net[layer].load_state_dict({k:state[f"actor.{key}.{k}"] for k in ("weight","bias")})
    return net.eval(), torch.zeros(17), torch.ones(17)


def phase_actor(state, member):
    # Convert one saved batched actor to the same deterministic CPU policy.
    net = nn.Sequential(nn.Linear(19,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(),
                        nn.Linear(256,6), nn.Tanh())
    for layer, key in ((0,"b.0"),(2,"b.2"),(4,"mu")):
        net[layer].load_state_dict(dict(weight=state[key+".W"][member].T, bias=state[key+".b"][member,0]))
    return net.eval(), torch.zeros(19), torch.ones(19)


def demonstrations(root):
    path = root / "minari/mujoco/walker2d/expert-v0/data/main_data.hdf5"
    with h5py.File(path) as f:
        eligible = [i for i in range(len(f)) if len(f[f"episode_{i}"]["rewards"]) >= 999]
        chosen = np.random.default_rng(0).permutation(eligible)[:10]
        episodes = [f[f"episode_{i}"] for i in chosen]
        obs = np.concatenate([e["observations"][:-1] for e in episodes]).astype("float32")
        actions = np.concatenate([e["actions"][:] for e in episodes]).astype("float32")
    return torch.from_numpy(obs), torch.from_numpy(actions), chosen


def train(obs, actions, seed):
    torch.manual_seed(seed)
    center, scale = obs.mean(0), obs.std(0, unbiased=False).clamp_min(.001)
    x = (obs-center)/scale
    net = nn.Sequential(nn.Linear(x.shape[1],128),nn.Tanh(),nn.Linear(128,128),
                        nn.Tanh(),nn.Linear(128,actions.shape[1]),nn.Tanh())
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    for _ in range(2000):
        ids = torch.randint(len(x),(128,))
        loss = (net(x[ids])-actions[ids]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net.eval(), center, scale


@torch.inference_mode()
def rollouts(net, center, scale, reference, period, reset_base=20000, command=None):
    name = "Walker2d-v5"
    envs = [make_env(name, False) for _ in range(24)]
    feet = [FootTD(e, name) for e in envs]
    states = np.stack([e.reset(seed=reset_base+i)[0] for i,e in enumerate(envs)]).astype("float32")
    alive = np.ones(24, bool)
    fell = np.zeros(24, bool)
    heights, speeds = [[] for _ in envs], [[] for _ in envs]
    returns = np.zeros(24)
    time = 0
    while alive.any():
        ids = np.flatnonzero(alive)
        inputs = states[ids]
        if command is not None:
            phase = 2*np.pi*((ids*.37)%1.) + 2*np.pi*time/period
            if command == "jump":
                jump_at = np.where(ids%6 < 3, 300, 600)
                delta = np.array((np.pi/2,-np.pi/2,np.pi))[ids%3]
                phase += delta*(time >= jump_at)
            inputs = np.column_stack((inputs,np.cos(phase),np.sin(phase))).astype("float32")
        actions = net((torch.from_numpy(inputs)-center)/scale).numpy()
        for i, action in zip(ids, actions):
            o, reward, term, trunc, _ = envs[i].step(action)
            states[i] = o
            heights[i].append(feet[i].z_live(envs[i].unwrapped.data))
            speeds[i].append(o[NQ[name]])
            fell[i] = term
            alive[i] = not (term or trunc)
            returns[i] += reward
        time += 1
    rows = []
    for i,env in enumerate(envs):
        touchdowns = feet[i].events(np.array(heights[i]))
        ordered = sorted((int(t)+1,reference["td_phase"][j]) for j,events in enumerate(touchdowns) for t in events)
        steps = np.array([t for t,_ in ordered])
        target = np.array([p for _,p in ordered])
        clock = 2*np.pi*((i*.37)%1.) + 2*np.pi*steps/period
        jt = (300,600)[(i%6)//3]
        shift = (np.pi/2,-np.pi/2,np.pi)[i%3]
        speed = float(np.mean(speeds[i]))
        walking = speed >= .5*reference["vx"]
        row = dict(reset=reset_base+i, speed=speed, length=len(speeds[i]), fell=bool(fell[i]),
                   walking=walking, env_return=returns[i])
        for key, angles in (("sham",clock-target), ("jump",clock+shift*(steps>=jt)-target)):
            row[key] = assess(steps,wrap(angles),jt,walking,fell[i],period,reference["td_tol"])
        # Both branches are judged against the SAME shifted clock. Equal foot weighting.
        agreement = []
        for j, events in enumerate(touchdowns):
            event_steps = np.asarray(events) + 1
            after = event_steps[(event_steps > jt) & (event_steps <= jt + 2*period)]
            if len(after):
                ang = 2*np.pi*((i*.37)%1.) + 2*np.pi*after/period + shift - reference["td_phase"][j]
                agreement.append(float(np.cos(ang).mean()))
        row["agreement"] = float(np.mean(agreement)) if len(agreement) == 2 else None
        rows.append(row)
        env.close()
    return rows


def paired(jumped, sham):
    differences = []
    prelocked = []
    for x,y in zip(jumped,sham,strict=True):
        assert x["reset"] == y["reset"]
        if all(r["walking"] and not r["fell"] and r["agreement"] is not None for r in (x,y)):
            delta = x["agreement"] - y["agreement"]
            differences.append(delta)
            if x["jump"]["prelocked"] and y["sham"]["prelocked"]:
                prelocked.append(delta)
    return dict(total=len(jumped), eligible=len(differences), prelocked_eligible=len(prelocked),
                mean_change=float(np.mean(differences)) if differences else None,
                prelocked_change=float(np.mean(prelocked)) if prelocked else None,
                jump_success=sum(r["jump"]["success"] for r in jumped),
                jump_walk=sum(r["walking"] for r in jumped))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("E:/0.Project/fm34_env"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--farama-weights", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    pivot = json.loads((args.root / "phase2/Walker2d-v5_pilot_amp_phase4_s0.json").read_text())
    obs, actions, ids = demonstrations(args.root)
    trials = {seed:rollouts(*train(obs, actions, seed), pivot["expert"], pivot["period"]) for seed in (31,32,33)}
    expert = public_actor(args.farama_weights)
    for base in (20000,21000,22000):
        trials[f"farama_{base}"] = rollouts(*expert, pivot["expert"], pivot["period"], reset_base=base)
    # A clock-blind deterministic actor has identical genuine and sham trajectories by construction.
    pairs = {k:paired(v,v) for k,v in trials.items() if str(k).startswith("farama")}
    trained = torch.load(args.root / "phase2/Walker2d-v5_pilot_actors.pt", weights_only=True,map_location="cpu")
    for i,(method,seed) in enumerate(trained["members"]):
        if method in ("rope_like_state","concat4_like_state","amp_phase4"):
            net = phase_actor(trained["actor"], i)
            contrast = [rollouts(*net, pivot["expert"], pivot["period"], command=cmd) for cmd in ("jump","sham")]
            pairs[f"{method}_s{seed}"] = paired(*contrast)
    means = {seed:dict(speed=np.mean([r["speed"] for r in seq]), walking=np.mean([r["walking"] for r in seq]),
                      fell=np.mean([r["fell"] for r in seq]), success=np.mean([r["jump"]["success"] for r in seq]))
             for seed,seq in trials.items()}
    args.output.write_text(json.dumps(dict(train_ids=ids.tolist(), farama_sha256=FARAMA_SHA,
                                          trials=trials, means=means, paired=pairs),
                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")
