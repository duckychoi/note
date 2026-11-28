"""Matched single-RGB transfer from three teachers on separate ETH3D fitting scenes."""
import argparse
from contextlib import nullcontext
import json
import time

import numpy as np
import psutil
import torch

import pf42_trials as prior
from pf42_b_teacher import train_student
from pf42_crossview import comparison, decide, entries, load
from pf42_crossview_data import DEPTH, FIT, OUT
from pf42_trial_data import Reader, gpu_slot
from run_stereo_students import capacity


def wait_for_memory():
    while psutil.virtual_memory().available < 8 * 2**30:
        time.sleep(4)


def make_fit(settings, device):
    folder = OUT / f"student-features-{device}"
    folder.mkdir(exist_ok=True)
    reader = None
    rows, distributions, frames = [], [], []
    for path in entries("fit", FIT):
        wait_for_memory()
        obs = load(path)
        keep = (np.isfinite(obs["peaks"]).sum(1) >= 3) & (np.isfinite(obs["cost"]).sum(1) >= 2).any(1)
        if not keep.any():
            continue
        x, y = obs["tally"][keep].astype(int).T
        feature_path = folder / path.name
        if feature_path.is_file():
            cached = load(feature_path)
            if str(cached["ref"]) != str(obs["ref"]) or not np.array_equal(cached["keep"], keep):
                raise ValueError(f"Stale features for {path.name}")
        else:
            if reader is None:
                if device == "cuda":
                    while not capacity():
                        time.sleep(4)
                reader = Reader(device)
            base, features = reader.sample(obs["rgb"], y, x)
            cached = dict(base=base, features=features, keep=keep, ref=obs["ref"])
            temp = feature_path.with_suffix(".partial.npz")
            np.savez_compressed(temp, **cached)
            temp.replace(feature_path)
        data = {key: obs[key][keep] for key in ("peaks", "cost", "visibility")}
        data["groups"] = obs["groups"]
        z = decide(data, "median", 0., 0.)
        rows.append(dict(base=cached["base"], features=cached["features"], y=z[:, None],
                         group=np.repeat(str(obs["scene"]), len(z))))
        distributions.append({"shared": decide(data, "shared", *settings["shared"], posterior=True)[1],
                              "calibrated_visibility": decide(data, "visibility", *settings["visibility"], posterior=True)[1],
                              "matched_visibility": decide(data, "visibility", *settings["shared"], posterior=True)[1]})
        frames.append(dict(scene=str(obs["scene"]), ref=str(obs["ref"]), points=len(z)))
        print("features", path.stem, len(z), device, flush=True)
    if not rows:
        raise ValueError("No fitting frames have support from three source observations")
    fit = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    if set(fit["group"]) != set(FIT) or not np.isfinite(fit["features"]).all():
        raise ValueError("All fitting scenes need finite features on common support")
    mass = {key: np.concatenate([d[key] for d in distributions]) for key in distributions[0]}
    targets = {key: (np.broadcast_to(np.log(DEPTH), value.shape), value) for key, value in mass.items()}
    return fit, targets, frames


def run(device):
    result = json.loads((OUT / "results.json").read_text(encoding="utf8"))
    if not result["student_gate"]:
        raise ValueError("The registered teacher comparison must pass before fitting the student")
    with gpu_slot() if device == "cuda" else nullcontext():
        fit, targets, frames = make_fit(result["settings"], device)
        if device == "cuda":
            torch.cuda.empty_cache()
    data = {n: prior.load(n) for n in ("synthetic", *prior.DATASETS)}
    data["fit"] = fit
    shared = targets.pop("shared")
    runs = train_student(data, shared, output=OUT, soft_controls=targets)
    names = tuple(runs[0]["scores"])
    means, differences, bootstrap = {}, {}, {}
    for key in prior.DATASETS:
        datasets = {arm: {group: float(np.mean([run["scores"][arm][key][group] for run in runs]))
                          for group in runs[0]["scores"][arm][key]} for arm in names}
        datasets["apple"] = prior.scores(data[key], 1.)
        means[key] = {arm: float(np.mean(list(scores.values()))) for arm, scores in datasets.items()}
        bootstrap[key] = {a: comparison(datasets["separated"], datasets[a])
                          for a in datasets if a != "separated"}
        differences[key] = [100 * (1 - np.mean(list(run["scores"]["separated"][key].values()))
                                      / np.mean(list(run["scores"]["direct"][key].values()))) for run in runs]
    stats = dict(means=means, comparison=bootstrap, per_seed_vs_direct=differences,
                 runs=runs, fit_frames=frames, synthetic_groups=len(set(data["synthetic"]["group"])),
                 synthetic_points=len(data["synthetic"]["base"]), device=device,
                 meets_3pct={key: val["direct"]["relative"] >= 3 for key, val in bootstrap.items()})
    (OUT / "student-results.json").write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf8")
    print(means, stats["meets_3pct"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    opts = parser.parse_args()
    torch.set_num_threads(2)
    device = ("cuda" if capacity() else "cpu") if opts.device == "auto" else opts.device
    run(device)
