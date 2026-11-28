"""Paired observation targets under independently changed lighting and viewpoint."""
import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from common import ALL, ROOT, STORE
from fetch_control_lights import TARGET


SOURCE = STORE.parent
RUN = SOURCE / "encoder42_eval" / "control43-light-view-v2"
MODES = ("light", "view", "both")


def write_json(path, payload):
    part = path.with_suffix(".partial")
    part.write_text(json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=1), encoding="utf8")
    part.replace(path)


def free_gib():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in ("total", "available", "tp", "ap", "tv", "av", "ex")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read physical memory availability")
    return status.available / 2**30


def wait_for_ram(reserve=12):
    while free_gib() < reserve:
        time.sleep(15)


def inverse_pose(pose):
    rotation = pose[:, :3].T
    return np.column_stack((rotation, -rotation @ pose[:, 3]))


def warp_geometry(first, other, pose):
    _, depth, valid, k = first
    _, other_z, other_valid, k1 = other
    y, x = np.indices(depth.shape, dtype=np.float32)[:, ::2, ::2]
    pixels = np.stack((x, y, np.ones_like(x)), -1)
    ref_z = depth[::2, ::2]
    points = (pixels @ np.linalg.inv(k).T)*ref_z[..., None]
    in_other = points @ pose[:, :3].T + pose[:, 3]
    projected = in_other @ k1.T
    xy = np.ascontiguousarray(projected[..., :2]/np.maximum(projected[..., 2:], 1e-5), dtype=np.float32)
    h, w = other_z.shape
    keep = (valid[::2, ::2] & (in_other[..., 2] > 0) & (xy[..., 0] >= 0) & (xy[..., 0] < w-1)
            & (xy[..., 1] >= 0) & (xy[..., 1] < h-1))
    # Require every contributing neighbor to be on the same visible surface.
    xx = np.clip(np.floor(xy[..., 0]).astype(int), 0, w-2)
    yy = np.clip(np.floor(xy[..., 1]).astype(int), 0, h-2)
    for dy in (0, 1):
        for dx in (0, 1):
            local_z = other_z[yy+dy, xx+dx]
            keep &= other_valid[yy+dy, xx+dx] & (np.abs(local_z-in_other[..., 2]) <= .02*local_z)
    rays = in_other / np.maximum(in_other[..., 2:], 1e-5)
    return xy, rays, keep


def remap(a, xy):
    h, w = a.shape
    x, y = xy[..., 0].clip(0, w-1), xy[..., 1].clip(0, h-1)
    ix, iy = x.astype(int), y.astype(int)
    jx, jy = np.minimum(ix+1, w-1), np.minimum(iy+1, h-1)
    u, v = x-ix, y-iy
    return (1-v)*((1-u)*a[iy, ix]+u*a[iy, jx])+v*((1-u)*a[jy, ix]+u*a[jy, jx])


def features(rgb, depth):
    dy, dx = np.gradient(np.log(depth.clip(1.e-6)))
    return depth, np.hypot(dx, dy), np.abs(rgb.mean(axis=2, dtype=np.float32)/255-.5)


def accumulate(prior, refs, candidates, truth, mask):
    x, z = np.log(prior[mask]), np.log(truth[mask])
    a = (x-z).astype(np.float64)
    out = {}
    for mode, (depth, edge, exposure) in zip(MODES, candidates):
        delta = np.log(depth[mask]).astype(np.float64)-x
        edge_bin = np.digitize(edge[mask]-refs[0][mask], [-.02, .02])
        exposure_bin = np.digitize(exposure[mask]-refs[1][mask], [-.10, .10])
        bins = 3*edge_bin + exposure_bin
        weights = [None, a*a, a*delta, delta*delta, np.minimum(a*a, (a+delta)**2)]
        out[mode] = np.stack([np.bincount(bins, weights=v, minlength=9) for v in weights], axis=1).tolist()
    return out


def examine(pair, pose, predictions):
    if pose is None:
        return []
    results = []
    for side in (0, 1):
        first, other = pair[side], pair[1-side]
        movement = pose if side == 0 else inverse_pose(pose)
        xy, rays, visible = warp_geometry(first, other, movement)
        truth = first[1][::2, ::2]
        target_features = []
        for depth, edge, exposure in predictions[1-side]:
            sample_z = remap(depth, xy)
            obs = ((rays * sample_z[..., None] - movement[:, 3]) @ movement[:, :3])[..., 2]
            target_features.append((obs, remap(edge, xy), remap(exposure, xy)))
        for light in (0, 1):
            p, edge, exposure = [v[::2, ::2] for v in predictions[side][light]]
            candidates = [[v[::2, ::2] for v in predictions[side][1-light]],
                          target_features[light], target_features[1-light]]
            keep = visible & np.isfinite(p) & np.isfinite(edge) & np.isfinite(exposure) & (p >= .25) & (p <= 30.)
            for v, e, l in candidates:
                keep &= np.isfinite(v) & (v >= .25) & (v <= 30.) & np.isfinite(e) & np.isfinite(l)
            results.append(dict(side=side, anchor_light=light, eligible=int(keep.sum()), grid=keep.size,
                                moments=accumulate(p, (edge, exposure), candidates, truth, keep)))
    return results


def real_inputs(scene):
    from depth_full_data import make_sample
    from PIL import Image

    raw = STORE / "raw" / scene
    meta = dict(s.split("=", 1) for s in (raw/"calib.txt").read_text().splitlines())
    cams, lights = [], []
    for side in (0, 1):
        k = np.fromstring(meta[f"cam{side}"].strip("[]").replace(";", " "), sep=" ").reshape(3, 3)
        with Image.open(raw/f"disp{side}.pfm") as f:
            disparity = np.asarray(f, dtype=np.float32)+float(meta["doffs"])
        good = np.isfinite(disparity) & (disparity > .25)
        good &= .25 / np.maximum(disparity-.25, .0001) <= .01
        z = float(meta["baseline"])*.001*k[0, 0]/np.where(good, disparity, 1.)
        both = []
        for light in ("L0", "L1"):
            with Image.open(TARGET/scene/"ambient"/light/f"im{side}e0.png") as f:
                rgb = np.array(f.convert("RGB"))
            data = make_sample(rgb, z, k, stride=6, valid=good)
            both.append(data[0])
        cams.append(data)
        lights.append(both)
    return cams, lights, np.column_stack((np.eye(3), [-float(meta["baseline"])/1000, 0, 0])).astype(np.float32)


def collect(phase):
    import torch
    from depth_full_data import Hypersim
    from diagnose_fusion_targets import calibration_pairs
    sys.path.insert(0, str(SOURCE/"ml-depth-pro-2"/"src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms
    from depth_pro_2.model import resolve_checkpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.hub.set_dir(str(SOURCE/"ml-depth-pro-2-cache"/"hub"))
    preset = replace(PRESETS["depthpro2-s"], fov_checkpoint_uri=None)
    wait_for_ram()
    if not ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000):
        raise OSError("Cannot set below-normal CPU priority")
    model, transform = create_model_and_transforms(preset, device="cpu")
    model.eval()
    with resolve_checkpoint(preset.checkpoint_uri).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    predictions_dir = RUN/"predictions"
    predictions_dir.mkdir(exist_ok=True)
    if phase == "real":
        source_files = json.loads((TARGET/"source_files.json").read_text())
        jobs = [(phase, scene) for scene in ALL]
    else:
        dev = Hypersim("validation_24", 4, pair=True)
        base = json.loads((SOURCE/"encoder42_eval/full-comparison-4243/hypersim_frozen_fusion_diagnostic.json").read_text())
        if digest != base["weight_sha256"]:
            raise ValueError("Changed frozen model")
        jobs = [("train", row) for row in calibration_pairs(4243)] + [
            ("dev", row["pair"]) for row in base["rows"]["native"]]
    lut = np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)
    for split, row in jobs:
        scene = row if split == "real" else row[0]
        path = RUN/f"{split}-{scene}.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf8"))
            if previous["collection_sha256"] != signature or previous["sha256"] != digest or list(previous["row"]) != list(row):
                raise ValueError("Cannot mix different collector, sample or model versions")
            continue
        wait_for_ram()
        if split == "real":
            required = [f"{scene}/ambient/{l}/im{s}e0.png" for s in (0, 1) for l in ("L0", "L1")]
            if any(source_files[f] == "missing" for f in required):
                write_json(path, dict(split=split, scene=scene, row=row, sha256=digest, collection_sha256=signature,
                                      cases=[], reason="missing light"))
                continue
            pair, lights, pose = real_inputs(scene)
        else:
            left, right, pose = dev.read_pair(row)
            pair = (left, right)
            lights = [[cam[0], lut[cam[0]]] for cam in pair]
        preds = []
        with torch.inference_mode():
            if pose is not None:
                for camera, exposures in zip(pair, lights):
                    fx = float(camera[3][0, 0])
                    values = []
                    for rgb in exposures:
                        z = model.infer(transform(rgb), f_px=fx)["depth"].cpu().numpy()
                        values.append(features(rgb, z))
                    preds.append(values)
        if pose is not None:
            cache = predictions_dir/f"{split}-{scene}.npy"
            tmp = cache.with_suffix(".partial")
            with tmp.open("wb") as stream:
                np.save(stream, np.asarray([[item[0] for item in side] for side in preds], dtype=np.float32))
            tmp.replace(cache)
        write_json(path, dict(split=split, scene=scene, row=row, sha256=digest, collection_sha256=signature,
                              cases=examine(pair, pose, preds), free_ram_gib=free_gib()))
        print(split, scene, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["synth", "real"])
    args = p.parse_args()
    RUN.mkdir(exist_ok=True, parents=True)
    collect(args.phase)
