"""Can a Walker2d expert replay which never sees a clock pass the phase-jump gate?"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/v2/il"))
from common import make_env
from phase_util import FootTD, NQ, wrap


def assess(steps, errors, jump_time, walking, fell, period, tolerance):
    before = np.abs(errors[(steps >= 100) & (steps < jump_time)])
    after = np.flatnonzero(steps >= jump_time)
    streak = np.convolve((np.abs(errors[after]) < tolerance).astype(int), np.ones(3), mode="valid") if len(after) >= 3 else []
    hits = np.flatnonzero(np.asarray(streak) == 3)
    success = bool(walking and not fell and len(hits))
    lag = float((steps[after[hits[0]]] - jump_time)/period) if success else None
    return dict(success=success, cycles=lag, prelocked=bool(len(before)>=3 and np.all(before[-3:] < tolerance)))


def aggregate(rows):
    result = {}
    for key in ("jump", "sham", "prealigned_jump"):
        success = np.array([r[key]["success"] for r in rows])
        locked = np.array([r[key]["prelocked"] for r in rows])
        boot = np.random.default_rng(90).choice(success, (5000, len(rows))).mean(axis=1)
        result[key] = dict(pass_count=int(success.sum()), total=len(rows), rate=success.mean(),
                           ci95=np.quantile(boot,[.025,.975]).tolist(), prelocked_count=int(locked.sum()),
                           pass_and_prelocked=int((success & locked).sum()))
    return result


def experiment(root):
    env_id = "Walker2d-v5"
    # The saved benchmark contains the training-derived phase reference and threshold.
    saved = sorted((root / "phase2").glob(f"{env_id}_pilot_*_s*.json"))
    configs = [json.loads(p.read_text()) for p in saved]
    assert len(configs) == 12 and all("final" in j and j["evals"][-1]["step"] == 100000 for j in configs)
    reference = configs[0]["expert"]
    period = configs[0]["period"]
    assert all(j["expert"] == reference and j["period"] == period for j in configs)
    archive = root / "minari/mujoco/walker2d/expert-v0/data/main_data.hdf5"
    env = make_env(env_id, False)
    env.reset(seed=0)
    detector = FootTD(env, env_id)
    rows = []
    with h5py.File(archive) as store:
        eligible = [i for i in range(len(store)) if store[f"episode_{i}"]["rewards"].shape[0] >= 999]
        used = np.random.default_rng(0).permutation(eligible)[:20]
        ids = np.random.default_rng(891).choice(np.setdiff1d(np.arange(len(store)), used), 120, replace=False)
        for ix, episode in enumerate(ids):
            group = store[f"episode_{episode}"]
            obs = group["observations"][1:]
            fell = bool(np.any(group["terminations"][:]))
            walking = bool(obs[:, NQ[env_id]].mean() >= .5*reference["vx"])
            feet = detector.events(detector.z_from_obs(env.unwrapped.data, obs))
            ordered = sorted((int(t)+1, reference["td_phase"][foot]) for foot, ev in enumerate(feet) for t in ev)
            steps = np.array([p[0] for p in ordered])
            targets = np.array([p[1] for p in ordered])
            # Five balanced repetitions of the 24 clock settings from train_phase.evaluate.
            setting = ix % 24
            jump_time = (300,600)[(setting % 6)//3]
            jump = (np.pi/2, -np.pi/2, np.pi)[setting % 3]
            phase = 2*np.pi*((setting*.37) % 1.)
            error = phase + 2*np.pi*steps/period - targets
            shift = jump*(steps >= jump_time)
            # Sensitivity check: align the clock to the LAST three pre-jump landings.
            pre = error[(steps >= 100) & (steps < jump_time)][-3:]
            aligned = error - np.angle(np.mean(np.exp(1j*pre))) if len(pre) else error
            row = dict(episode=int(episode), jump_time=jump_time, jump_degrees=round(np.degrees(jump)),
                       walking=walking, fell=fell)
            for name, angles in (("jump",error+shift), ("sham",error), ("prealigned_jump",aligned+shift)):
                row[name] = assess(steps, wrap(angles), jump_time, walking, fell, period, reference["td_tol"])
            rows.append(row)
    env.close()
    pilot = {f"{j['args']['method']}_s{j['args']['seed']}":
             {k:j["final"]["jump"][k] for k in ("vx","walk_rate","fall_rate","success_rate")}
             for j in configs}
    return dict(period=period, expert=reference, episodes=rows, aggregate=aggregate(rows),
                steps_100k=pilot, source_pilot=saved[0].name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/0.Project/fm34_env"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(experiment(args.root), indent=2, allow_nan=False) + "\n", encoding="utf-8")
