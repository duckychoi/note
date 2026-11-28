"""Matched native-resolution Depth Pro 2-S adaptation and one-RGB evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from depth_full_train import supervised
from light_encoder_score import images, summary
from light_encoder_study import CKPT, Model, atomic_save, sha, write, yield_memory
from native_direction_eval import dataframes
from photometric_teacher import MATERIALS, N_LIGHTS, OUT, SOURCE, STRIDE, endpoints, rgb
from depth_full_data import hypersim_intrinsics
from run_stereo_students import gpu_slot


ARMS = {"augmentation": "d2", "photometric": "fixed", "ground_truth": "conditional"}
SEED = 261127
PASSES = 2


def manifest(split):
    return [r for r in json.loads((SOURCE / "scenes.json").read_text()) if r["split"] == split]


def photographs(rows):
    return [(row["scene"], material, light) for row in rows
            for material in MATERIALS for light in range(N_LIGHTS)]


def load_render(scene, material, light, training=False):
    folder = OUT / "renders" / scene
    image = rgb(folder / f"{material}-{light}.png")
    height, width = image.shape[:2]
    focal = hypersim_intrinsics(scene, width, height)[0, 0]
    with np.load(folder / "geometry.npz") as data:
        sl = slice(None, None, STRIDE if training else 1)
        z = data["z"][sl, sl]
        valid = data["valid"][sl, sl] & (z >= .25) & (z <= 30)
        if not training:
            return image, z, valid, focal
    with np.load(folder / "teacher.npz") as data:
        edges = [(data[f"{material}_{axis}_target"].astype(np.float32), data[f"{material}_{axis}_mask"])
                 for axis in (0, 1)]
    return image, z, valid, focal, edges


def tensor(array):
    return torch.as_tensor(np.ascontiguousarray(array), device="cuda")


def auxiliary_edges(logz, edges, exact):
    estimate, obs, gt = [], [], []
    for axis, (normal, mask) in enumerate(edges):
        p, q = endpoints(logz, axis)
        zp, zq = endpoints(exact, axis)
        keep = tensor(mask)
        estimate.append((q - p)[keep])
        obs.append(tensor(normal)[keep])
        gt.append((zq - zp)[keep])
    return torch.cat(estimate), torch.cat(obs), torch.cat(gt)


def train():
    pictures = photographs(manifest("train"))
    rng = np.random.default_rng(SEED)
    order = [pictures[i] for _ in range(PASSES) for i in rng.permutation(len(pictures))]
    signature = [sha(CKPT), sha(Path(__file__)), sha(Path(__file__).with_name("photometric_teacher.py")),
                 sha(Path(__file__).with_name("photometric_render.py")), sha(Path(__file__).with_name("PHOTOMETRIC_STEREO_PROTOCOL.md"))]
    for scene, _, _ in pictures[::len(MATERIALS) * N_LIGHTS]:
        signature.append(sha(OUT / "renders" / scene / "teacher.npz"))
    torch.manual_seed(SEED)
    model = Model(True)
    optimizers = {name: torch.optim.AdamW(model.arms[arm].parameters(), lr=3e-6, weight_decay=.01)
                  for name, arm in ARMS.items()}
    path = OUT / "latest.train.pt"
    done, log = 0, []
    if path.is_file():
        state = torch.load(path, weights_only=True, map_location="cuda")
        if state["signature"] != signature or state["order"] != order:
            raise ValueError("Resume inputs differ")
        done, log = state["done"], state["log"]
        for name, arm in ARMS.items():
            model.arms[arm].load_state_dict(state["weights"][name])
            optimizers[name].load_state_dict(state["optimizers"][name])
    for step, (scene, material, light) in enumerate(order[done:], done + 1):
        yield_memory()
        image, z, valid, focal, edges = load_render(scene, material, light, training=True)
        _, features = model.input(image)
        z, valid = tensor(np.where(valid, z, 1.)), tensor(valid)
        if not valid.any():
            raise ValueError(f"No valid depth for {scene}")
        target_log = z.log()
        row = dict(step=step, scene=scene, material=material, light=light)
        for name, arm in ARMS.items():
            predicted = model.q(features, arm)[::STRIDE, ::STRIDE]
            base = supervised(predicted, z, valid, focal / STRIDE, True)
            difference, pseudo, exact = auxiliary_edges(-predicted.log(), edges, target_log)
            add = base.new_zeros(())
            if name != "augmentation" and difference.numel():
                target = pseudo if name == "photometric" else exact
                add = F.smooth_l1_loss(difference, target, beta=.02)
            loss = base + .5 * add
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.arms[arm].parameters(), 1., error_if_nonfinite=True)
            optimizers[name].step()
            row[name] = [float(base.detach()), float(add.detach()), float(norm)]
            row["constraints"] = difference.numel()
        log.append(row)
        if step in (1, 4) or step % 32 == 0:
            print("train", step, "/", len(order), {k: row[k] for k in ARMS}, flush=True)
            weights = {name: model.arms[arm].state_dict() for name, arm in ARMS.items()}
            atomic_save(path, dict(done=step, order=order, log=log, signature=signature, weights=weights,
                                   optimizers={k: v.state_dict() for k, v in optimizers.items()}))
            if step in (len(pictures), len(order)):
                atomic_save(OUT / f"student-{step}.pt", dict(weights=weights, done=step, signature=signature,
                            params=sum(p.numel() for p in model.arms["d2"].parameters())))
    write(OUT / "training.json", dict(steps=len(order), rgb=len(pictures), log=log, signature=signature))


def synthetic():
    for scene, material, light in photographs(manifest("val")):
        image, z, valid, _ = load_render(scene, material, light)
        yield f"{scene}/{material}/{light}", scene, image, z, valid


def interval(groups, arm, metric):
    groups = list(groups.values())
    baselines = np.array([d["augmentation" + metric] for d in groups])
    treatments = np.array([d[arm + metric] for d in groups])
    ids = np.random.default_rng(SEED).integers(len(groups), size=(10000, len(groups)))
    improvement = 100 * (1 - treatments.mean() / baselines.mean())
    samples = 100 * (1 - treatments[ids].mean(1) / baselines[ids].mean(1))
    return dict(relative=float(improvement), interval=np.quantile(samples, [.025, .975]).tolist(),
                better=int((treatments < baselines).sum()))


@torch.inference_mode()
def evaluate(step, datasets):
    path = OUT / f"student-{step}.pt"
    record = torch.load(path, weights_only=True, map_location="cuda")
    if record["signature"][0] != sha(CKPT):
        raise ValueError("Frozen model changed")
    model = Model(True)
    for name, arm in ARMS.items():
        model.arms[arm].load_state_dict(record["weights"][name])
    for dataset in datasets:
        out = OUT / f"scores-{step}-{dataset}.json"
        fingerprint = sha(path)
        if out.exists() and json.loads(out.read_text(encoding="utf8"))["checkpoint"] == fingerprint:
            continue
        if dataset == "synthetic":
            frames = synthetic()
        elif dataset == "hammer":
            frames = dataframes(dataset)
        else:
            frames = images(dataset)
        rows = []
        for name, group, image, z, valid in frames:
            yield_memory()
            x, state, baseline = model.baseline(image)
            focal = model.net.fov.focal_length_px(x).float()
            z, valid = tensor(z), tensor(valid)
            valid = valid & torch.isfinite(z) & (z >= .25) & (z <= 30)
            if not valid.any():
                raise ValueError(f"No valid depths in {name}")
            scores = {}
            for key in ("frozen", *ARMS):
                inverse = baseline if key == "frozen" else model.q(state, ARMS[key])
                ratio = focal / (image.shape[1] * inverse[valid] * z[valid])
                logs = ratio.log()
                scores[key] = float((ratio - 1).abs().mean())
                scores[key + "_shape"] = float(logs.var(unbiased=False))
                scores[key + "_scale"] = float(logs.mean().square())
            if dataset == "synthetic":
                scene, material, _ = name.split("/")
                with np.load(OUT / "renders" / scene / "teacher.npz") as labels:
                    exact = torch.where(valid, z, 1.)[::STRIDE, ::STRIDE].log()
                    edges = [(labels[f"{material}_{i}_target"].astype(np.float32), labels[f"{material}_{i}_mask"])
                             for i in (0, 1)]
                for key in ("frozen", *ARMS):
                    inverse = baseline if key == "frozen" else model.q(state, ARMS[key])
                    pred, _, target = auxiliary_edges(-inverse[::STRIDE, ::STRIDE].log(), edges, exact)
                    # Some scenes have zero qualifying pairs for a particular material.
                    scores[key + "_local_n"] = target.numel()
                    scores[key + "_local_sum"] = float((pred - target).square().sum())
            if not np.isfinite(list(scores.values())).all():
                raise ValueError(name)
            rows.append(dict(key=name, group=group, scores=scores))
            if len(rows) % 100 == 0:
                print("eval", step, dataset, len(rows), flush=True)
        result = summary(rows)
        result["comparison"] = {key: {m or "absrel": interval(result["groups"], key, m)
                                       for m in ("", "_shape")} for key in ("photometric", "ground_truth")}
        write(out, dict(**result, rows=rows, checkpoint=fingerprint, official_path_error=model.check,
                        init=record["signature"][0], steps=step, params=record["params"]))
        print("eval", step, dataset, result["n"], result["means"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "eval", "both"))
    options = parser.parse_args()
    with gpu_slot():
        if options.stage in ("train", "both"):
            train()
        if options.stage in ("eval", "both"):
            evaluate(512, ("synthetic",))
            evaluate(1024, ("synthetic", "hammer", "diode", "ibims"))
