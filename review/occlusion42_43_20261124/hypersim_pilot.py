"""Fixed-backbone, synthetic-only adaptation on genuine Hypersim RGB-D."""
import argparse
import csv
from dataclasses import replace
import sys

import numpy as np

from common import ROOT, STORE, TEST, save


SYNTH = ROOT.parents[2] / "research-sources" / "hypersim42_pilot"
SCENES = tuple(f"ai_{i:03d}_001" for i in range(1, 7))
SAMPLE = SYNTH / "prepared.npz"
INFERRED = SYNTH / "inferred_s.npz"
RIDGE = .01
WIDTH = 320


def prepare(scene_names=SCENES, output=SAMPLE):
    import cv2
    import h5py
    from PIL import Image

    with (SYNTH / "camera.csv").open(newline="", encoding="utf-8") as stream:
        table = {r["scene_name"]: r for r in csv.DictReader(stream)}
    clean, changed, truth, focal, scene_ids, frames = [], [], [], [], [], []
    for scene_id, scene in enumerate(scene_names):
        r = table[scene]
        proj = np.array([[float(r[f"M_proj_{i}{j}"]) for j in range(4)] for i in range(4)])
        for path in sorted((SYNTH / scene).glob("*.tonemap.jpg")):
            with Image.open(path) as file:
                rgb = np.asarray(file.convert("RGB"))
            frame = path.name.split(".")[0]
            with h5py.File(path.with_name(frame + ".depth_meters.hdf5"), "r") as file:
                rays = file["dataset"][:]
            h, w = rays.shape
            assert rgb.shape[:2] == (h, w)
            new_h = int(round(WIDTH * h / w))
            py = ((np.arange(new_h) + .5) * h / new_h).astype(int)
            px = ((np.arange(WIDTH) + .5) * w / WIDTH).astype(int)
            y, x = np.meshgrid(py, px, indexing="ij")
            clip = np.stack(((x+.5)*2/w-1, 1-(y+.5)*2/h, np.zeros_like(x), np.ones_like(x)), -1)
            direction = clip @ np.linalg.inv(proj).T
            z = rays[y, x] * np.abs(direction[..., 2]) / np.linalg.norm(direction[..., :3], axis=-1)
            assert np.mean(np.isfinite(z) & (z > 0)) > .60
            resized = cv2.resize(rgb, (WIDTH, new_h), interpolation=cv2.INTER_AREA)
            exposure = .58 if int(frame) % 2 else 1.38
            tone = np.array([1.06, 1., .89]) if int(frame) % 2 else np.array([.93, 1., 1.08])
            edited = np.clip(np.rint(255 * (resized / 255.)**(1/exposure) * tone), 0, 255).astype(np.uint8)
            clean.append(resized)
            changed.append(edited)
            truth.append(z.astype(np.float32))
            focal.append(WIDTH * abs(proj[0, 0]) / 2)
            scene_ids.append(scene_id)
            frames.append(frame)
    assert set(scene_ids) == set(range(len(scene_names)))
    np.savez_compressed(output, clean=clean, changed=changed, truth=truth, focal=focal,
                        scenes=scene_ids, frames=frames)
    print("Prepared photorealistic frames", np.bincount(scene_ids), flush=True)


def infer():
    print("Loading torch", flush=True)
    import torch
    path = ROOT.parents[2] / "research-sources" / "ml-depth-pro-2" / "src"
    sys.path.insert(0, str(path))
    from depth_pro_2 import PRESETS, create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint
    import hashlib

    print("Loading official Depth Pro 2-S", flush=True)
    torch.set_num_threads(2)
    preset = replace(PRESETS["depthpro2-s"], fov_checkpoint_uri=None)
    with resolve_checkpoint(preset.checkpoint_uri).open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    model, transform = create_model_and_transforms(preset, device="cuda")
    with np.load(SAMPLE) as inp, torch.inference_mode():
        focal = inp["focal"]
        predictions = {}
        for mode in ("clean", "changed"):
            ys = []
            for rgb, fx in zip(inp[mode], focal):
                z = model.infer(transform(rgb), f_px=float(fx))["depth"].float().cpu().numpy()
                assert z.shape == rgb.shape[:2] and np.isfinite(fx) and fx > 0, (z.shape, rgb.shape, fx)
                valid = np.isfinite(z) & (z > 0)
                if not valid.all():
                    assert valid.any()
                    print("Masked invalid depth", mode, len(ys), int((~valid).sum()), flush=True)
                z = np.where(valid, z, np.nan)
                ys.append(z)
            predictions[mode] = np.stack(ys)
            print("Inferred", mode, len(ys), flush=True)
    np.savez_compressed(INFERRED, **predictions, checkpoint_sha256=digest)


def design(rgb, depth):
    log = np.log(np.maximum(depth, .01))
    if not np.isfinite(log).all():
        log = np.where(np.isfinite(log), log, np.nanmedian(log))
    rgb = rgb.astype(np.float64) / 255
    dy, dx = np.gradient(log)
    luma = rgb.mean(-1)
    return np.stack((np.ones_like(log), log / 3, luma, luma**2, rgb[..., 0]-rgb[..., 2],
                     np.clip(np.hypot(dy, dx), 0, .5), luma*log/3), -1)


def fit(blocks, method):
    import torch

    # RLS in information form. Unit observation covariance and a zero-mean
    # Gaussian prior are scaled so the final objective has the stated ridge.
    n = sum(len(y) for _, y in blocks)
    d = blocks[0][0].shape[1]
    prior = RIDGE * np.eye(d)
    information, natural = n * prior, np.zeros(d)
    batch_stats = [(x.T @ x, x.T @ y, len(y)) for x, y in blocks]
    for gram, cross, _ in batch_stats:
        information += gram
        natural += cross
    ridge = np.linalg.solve(information, natural)
    if method == "kalman":
        # Independent direct covariance-form block update on sufficient statistics.
        weight, covariance = np.zeros(d), np.linalg.inv(n * prior)
        for gram, cross, _ in batch_stats:
            covariance = np.linalg.solve(np.eye(d) + covariance @ gram, covariance)
            weight += covariance @ (cross - gram @ weight)
        assert np.max(np.abs(weight - ridge)) < 1e-10
        return weight
    weight = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.AdamW([weight], lr=.01, weight_decay=0)
    g = torch.from_numpy(information / n)
    c = torch.from_numpy(natural / n)
    for _ in range(1200):
        opt.zero_grad(set_to_none=True)
        (.5 * weight @ g @ weight - c @ weight).backward()
        opt.step()
    return weight.detach().numpy()


def score_predictions(predictions):
    from score import load_view, metrics
    rows = {scene: metrics(predictions[scene], load_view(scene, wide=True)) for scene in TEST}
    return {key: float(np.mean([
        np.mean([row[key] for scene, row in rows.items() if scene[:-1] == family])
        for family in sorted({s[:-1] for s in TEST})
    ])) for key in ("whole", "jump", "order")}


def evaluate():
    with np.load(SAMPLE) as f:
        synth = {key: f[key] for key in f.files}
    with np.load(INFERRED) as f:
        logs = {mode: np.log(f[mode].astype(np.float64)) for mode in ("clean", "changed")}
        digest = str(f["checkpoint_sha256"])
    with np.load(ROOT / "depthpro2-s_predictions.npz") as f:
        assert digest == str(f["checkpoint_sha256"])
        real = {scene: f[scene].astype(np.float64) for scene in TEST}
    features = {}
    for scene in TEST:
        with np.load(STORE / f"{scene}.npz") as f:
            features[scene] = design(f["rgb"], real[scene])
    truth = np.log(np.where(synth["truth"] > 0, synth["truth"], np.nan))
    finite = np.isfinite(truth) & np.isfinite(logs["clean"]) & np.isfinite(logs["changed"])
    dev = finite & (synth["scenes"][:, None, None] >= 4)
    shift = logs["clean"] - logs["changed"]
    gain = float(np.clip(np.dot(shift[dev], (truth-logs["changed"])[dev]) / np.dot(shift[dev], shift[dev]), 0, 1))
    fused = logs["changed"] + gain * shift
    rng = np.random.default_rng(42143)
    targets = {"clean": [], "gt": [], "fused": []}
    for i in np.flatnonzero(synth["scenes"] < 4):
        pool = np.flatnonzero(finite[i])
        idx = rng.choice(pool, 1024, replace=False)
        z = truth[i].ravel()[idx]
        for mode in ("clean", "changed"):
            x = design(synth[mode][i], np.exp(logs[mode][i])).reshape(-1, 7)[idx]
            prediction = logs[mode][i].ravel()[idx]
            y = z - prediction
            if mode == "clean":
                targets["clean"].extend([(x, y)] * 2)
            targets["gt"].append((x, y))
            target = z if mode == "clean" else .75 * z + .25 * fused[i].ravel()[idx]
            targets["fused"].append((x, target-prediction))
    arms = {"frozen": np.zeros(7)}
    for name in ("clean", "gt", "fused"):
        for optimizer in (("adamw",) if name == "clean" else ("adamw", "kalman")):
            arms[f"{name}_{optimizer}"] = fit(targets[name], optimizer)
    dev_images = np.flatnonzero(synth["scenes"] >= 4)
    dev_features = {mode: [design(synth[mode][i], np.exp(logs[mode][i])) for i in dev_images]
                    for mode in ("clean", "changed")}
    results = {}
    for arm, w in arms.items():
        depth = {s: real[s] * np.exp(features[s] @ w) for s in TEST}
        synth_absrel = {}
        for mode, maps in dev_features.items():
            total = []
            for i, x in zip(dev_images, maps):
                mask = finite[i]
                delta = (logs[mode][i]+x@w-truth[i])[mask]
                total.append(float(np.abs(np.expm1(delta)).mean()))
            synth_absrel[mode] = float(np.mean(total))
        results[arm] = dict(means=score_predictions(depth), weights=w.tolist(),
                            synthetic_dev_absrel=synth_absrel)
        print(arm, results[arm]["means"], flush=True)
    save("hypersim_pilot_results.json", dict(
        checkpoint_sha256=digest, scenes=np.bincount(synth["scenes"]).tolist(), gain=gain,
        invalid_prediction_pixels={mode: int((~np.isfinite(logs[mode])).sum()) for mode in logs},
        synthetic_dev_log_mse=[float(np.mean((z[dev]-truth[dev])**2))
                               for z in (logs["changed"], logs["clean"], fused)],
        adamw_kalman_max_weight_delta={name: float(np.max(np.abs(
            arms[f"{name}_adamw"] - arms[f"{name}_kalman"]))) for name in ("gt", "fused")},
        results=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    {"prepare": prepare, "infer": infer, "evaluate": evaluate}[cli.parse_args().stage]()
