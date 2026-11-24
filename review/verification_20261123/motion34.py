"""Evaluate frozen policies on paired MuJoCo resets and audit the registered endpoint."""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "priority_20261122"))
from walker_control import phase_actor


def registered(root):
    diagnostics = json.loads((HERE.parents[1] / "results/v2/diag_rollout_predict.json").read_text())
    rows = {}
    for path in sorted((root / "prereg").glob("*_s*.json")):
        data = json.loads(path.read_text())
        args = data["args"]
        env, method = args["env"], args["method"].split("_")[0]
        forecast = diagnostics[env][method]
        group = rows.setdefault(f"{env}/{method}", dict(prediction=forecast["prediction"],
                                old_prediction=forecast["old_prediction"], seeds={}))
        final = data.get("final")
        group["seeds"][args["seed"]] = dict(step=data["evals"][-1]["step"],
                                            final_vx=final["nominal"]["vx"] if final else None)
    for group in rows.values():
        seeds = group["seeds"]
        complete = set(seeds) == {0, 1, 2} and all(s["step"] == 300000 and s["final_vx"] is not None for s in seeds.values())
        group["observed"] = None
        if complete:
            speed = np.mean([s["final_vx"] for s in seeds.values()])
            group["mean_nominal_vx"] = float(speed)
            group["observed"] = "escape" if speed >= 1.0 else "stall"
    return rows


@torch.inference_mode()
def evaluate(policy, period, frozen):
    envs = [gym.make("HalfCheetah-v5") for _ in range(12)]
    try:
        obs = np.array([e.reset(seed=71100+i)[0] for i,e in enumerate(envs)])
        initial = np.array([e.unwrapped.data.qpos[0] for e in envs])
        offsets = np.random.default_rng(74531).uniform(-np.pi, np.pi, len(envs))
        for t in range(500):
            phase = offsets if frozen else offsets + 2*np.pi*t/period
            states = np.column_stack((obs, np.cos(phase), np.sin(phase))).astype("float32")
            actions = policy(torch.from_numpy(states)).numpy()
            obs = np.stack([e.step(a)[0] for e,a in zip(envs,actions)])
        final = np.array([e.unwrapped.data.qpos[0] for e in envs])
        return ((final-initial)/(500*envs[0].unwrapped.dt)).tolist()
    finally:
        for e in envs:
            e.close()


def main(root):
    result = dict(registered=registered(root), reset_seeds=list(range(71100,71112)), policies={})
    for folder, tag in [("phase_hc","redesign"),("prereg","prereg")]:
        prefix = root / folder / f"HalfCheetah-v5_{tag}"
        ckpt = torch.load(f"{prefix}_actors.pt", map_location="cpu", weights_only=True)
        for i,(name,seed) in enumerate(ckpt["members"]):
            if name.endswith("rwin"):
                continue
            metadata = json.loads(Path(f"{prefix}_{name}_s{seed}.json").read_text())
            assert metadata["evals"][-1]["step"] == 300000 and metadata["final"]
            policy = phase_actor(ckpt["actor"], i)[0]
            row = {key:evaluate(policy,metadata["period"],frozen) for key,frozen in [("clock",False),("frozen",True)]}
            row["half_expert_threshold"] = 0.5 * metadata["expert"]["vx"]
            result["policies"][f"{name}/{seed}"] = row
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("E:/0.Project/fm34_env"))
    parser.add_argument("--refresh-endpoints",action="store_true",
                        help="Refresh the registered run status in an existing evaluation.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = HERE/"motion34.json"
    if args.refresh_endpoints:
        result = json.loads(output.read_text())
        result["registered"] = registered(args.root)
    else:
        result = main(args.root)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
