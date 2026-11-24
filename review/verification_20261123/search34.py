"""Optimize four reward functions over the same open-loop policy class in real MuJoCo dynamics."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
from mujoco.rollout import Rollout
import numpy as np

from coordination34 import clips,distances,features

HERE = Path(__file__).resolve().parent
H = 192
SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS


def reset_state(env,seed):
    env.reset(seed=seed)
    m,d = env.unwrapped.model,env.unwrapped.data
    x = np.zeros(mujoco.mj_stateSize(m,SPEC))
    mujoco.mj_getState(m,d,x,SPEC)
    return x


def main(root,with_task=False):
    env = gym.make("HalfCheetah-v5")
    m,skip = env.unwrapped.model,env.unwrapped.frame_skip
    data = [mujoco.MjData(m) for _ in range(2)]
    engine = Rollout(nthread=len(data))
    try:
        clip,_ = clips(root,"halfcheetah","expert",6)
        sd = np.maximum(clip[:10].std((0,1)),.001)
        refs = {k:v.mean(0) for k,v in features(clip[:10,:128]/sd).items()}
        meta = json.loads((root/"phase_hc/HalfCheetah-v5_redesign_film4_like_plain_s0.json").read_text())
        period = meta["period"]
        def score(params,seed):
            start = reset_state(env,seed)
            phase = 2*np.pi*np.arange(H)/period + np.random.default_rng(seed).uniform(0,2*np.pi)
            basis = np.column_stack([np.ones(H),np.cos(phase),np.sin(phase),np.cos(2*phase),np.sin(2*phase)])
            act = np.tanh(np.einsum("th,nhc->ntc",basis,params.reshape(-1,5,6)))
            states,_ = engine.rollout(m,data,start[None],control=np.repeat(act,skip,axis=1))
            q = states[:,skip-1::skip,1:1+m.nq]
            z = distances(q[:,-128:,3:9]/sd,refs)
            z["velocity"] = (q[:,-1,0]-start[1])/(H*env.unwrapped.dt)
            z["upright_fraction"] = np.mean(np.cos(q[:,:,2])>0,axis=1)
            if with_task:
                task = -abs(z["velocity"]-1.2)-2*np.maximum(.9-z["upright_fraction"],0)
                z["task"] = task
                for k in ("psd","cross_spectrum"):
                    z["task_"+k] = task-.5*z[k]/(1+z[k])
            return z
        # Verify low-level batched dynamics against the actual Gymnasium entry point.
        start = reset_state(env,85400)
        check = np.random.default_rng(84).uniform(-.6,.6,(16,6))
        native,_ = engine.rollout(m,data,start[None],control=np.repeat(check[None],skip,axis=1))
        expected = []
        for a in check:
            env.step(a)
            expected.append(env.unwrapped.data.qpos.copy())
        reproduction = float(np.max(abs(native[0,skip-1::skip,1:1+m.nq]-expected)))
        assert reproduction<1e-8
        rows = []
        seed_base,reset_base = (85450,91000) if with_task else (85300,90000)
        methods = ("task","task_psd","task_cross_spectrum") if with_task else ("psd","signed_area","cross_spectrum","velocity")
        for seed in range(seed_base,seed_base+5):
            for objective in methods:
                rng = np.random.default_rng(seed)
                center,scale = np.zeros(30),np.full(30,.55)
                candidates = []
                direction = 1 if objective=="velocity" or with_task else -1
                for gen in range(5):
                    params = rng.normal(center,scale,(48,30))
                    avg = np.mean([score(params,reset_base+2*gen+j)[objective] for j in range(2)],axis=0)
                    elite = params[np.argsort(direction*avg)[-8:]]
                    center,scale = elite.mean(0),np.maximum(elite.std(0),.08)
                    candidates.extend([center.copy(),elite[-1].copy()])
                candidates = np.array(candidates)
                val = np.mean([score(candidates,reset_base+100+j)[objective] for j in range(4)],axis=0)
                chosen = candidates[np.argmax(direction*val)][None]
                evaluations = [score(chosen,reset_base+200+j) for j in range(12)]
                rows.append(dict(seed=seed,objective=objective,coefficients=chosen[0].tolist(),
                    metrics={k:[float(v[k][0]) for v in evaluations] for k in evaluations[0]}))
        return dict(numerical_gym_error=reproduction,period=period,horizon=H,rollout_steps_per_action=skip,
                    generations=5,population=48,train_resets_per_generation=2,validation_resets=4,
                    test_reset_seeds=list(range(reset_base+200,reset_base+212)),results=rows)
    finally:
        engine.close()
        env.close()


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--root",type=Path,default=Path("E:/0.Project/fm34_env"))
    arg.add_argument("--with-task",action="store_true")
    args = arg.parse_args()
    result = main(args.root,args.with_task)
    name = "task34.json" if args.with_task else "search34.json"
    (HERE/name).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n", encoding="utf-8")
