"""Check fixed-depth relighting and real exposure changes before training."""
import argparse
from dataclasses import replace
import sys
import zipfile

import numpy as np

from common import ROOT, STORE, save


SOURCES = ROOT.parents[2] / "research-sources"
OUTPUT = SOURCES / "occlusion42_mobile" / "lighting"
PREPARED = OUTPUT / "paired_conditions.npz"
PREDICTED = OUTPUT / "depthpro2_s_conditions.npz"


def linear(rgb):
    v = rgb.astype(np.float32) / 255
    return np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055)**2.4)


def encoded(v):
    v = np.clip(v, 0, 1)
    return np.rint(255 * np.where(v <= .0031308, 12.92 * v, 1.055 * v**(1/2.4) - .055)).astype(np.uint8)


def lamp_pair(rgb, z, fx):
    """Camera-centered beam with angular falloff and inverse-square attenuation."""
    h, w = z.shape
    y, x = np.indices(z.shape, dtype=np.float32)
    ray_norm = np.sqrt(1 + ((x-(w-1)/2)/fx)**2 + ((y-(h-1)/2)/fx)**2)
    incidence_angle = np.arccos(1 / ray_norm)
    valid = np.isfinite(z) & (z > 0)
    distance = np.maximum(np.where(valid, z, np.inf) * ray_norm, .3)
    beam = .03 + .97 * np.exp(-np.log(2) * (incidence_angle / np.deg2rad(10))**2)
    incident = beam / distance**2
    v = linear(rgb)
    light = .12 + .18 * incident / max(float(incident[valid].mean()), 1e-6)
    spatial = v * light[..., None]
    spatial_rgb = encoded(spatial)
    keep = valid & (z >= .25) & (z <= 30)
    channel_weights = np.array([.2126, .7152, .0722])
    expected = float((linear(spatial_rgb[keep]) @ channel_weights).mean())
    lo, hi = 0., 8.
    for _ in range(20):
        gain = (lo+hi)/2
        actual = float((linear(encoded(v[keep]*gain)) @ channel_weights).mean())
        if actual < expected:
            lo = gain
        else:
            hi = gain
    flat = encoded(v*((lo+hi)/2))
    assert abs(float((linear(flat[keep]) @ channel_weights).mean()) - expected) < .001
    return flat, spatial_rgb


def prepare():
    import cv2
    from measured import pfm

    metadata = []
    arrays = {}

    def add(domain, group, label, rgb, z, valid, fx):
        i = str(len(metadata))
        metadata.append((domain, group, label, fx))
        arrays.update({i+"_rgb": rgb, i+"_z": z, i+"_valid": valid})

    with np.load(SOURCES/"hypersim42_pilot"/"prepared.npz") as f:
        mask = f["scenes"] >= 4
        for rgb, z, fx, scene, frame in zip(*(f[k][mask] for k in ("clean","truth","focal","scenes","frames"))):
            valid = np.isfinite(z) & (z >= .25) & (z <= 30)
            flat, spatial = lamp_pair(rgb, z, fx)
            for label, pixels in (("reference", rgb), ("global", flat), ("beam", spatial)):
                add("synthetic", f"{scene}/{frame}", label, pixels, z, valid, fx)
    with zipfile.ZipFile(OUTPUT/"ambient-skates.zip") as archive:
        for scene in ("skates1", "skates2"):
            raw = STORE/"raw"/scene
            calib = dict(line.split("=", 1) for line in (raw/"calib.txt").read_text().splitlines())
            b, off = float(calib["baseline"])/1000, float(calib["doffs"])
            for side in (0, 1):
                disp = pfm(raw/f"disp{side}.pfm")
                h,w = disp.shape
                fx = float(calib[f"cam{side}"][1:].split()[0])
                rw,rh = 320, round(320*h/w)
                ys = ((np.arange(rh)+.5)*h/rh).astype(int)
                xs = ((np.arange(rw)+.5)*w/rw).astype(int)
                d = disp[np.ix_(ys,xs)].astype(np.float64) + off
                ok = np.isfinite(d) & (d > .25)
                z = fx*b / np.where(ok, d, np.nan)
                valid = ok & (z >= .25) & (z <= 30) & (.25/(d-.25) <= .01)
                for light, exposure in (("L0",0),("L0",1),("L0",2),("L1",0),("L1",2),("T0",0),("F0",0)):
                    key = f"data/{scene}/ambient/{light}/im{side}e{exposure}.png"
                    image = cv2.imdecode(np.frombuffer(archive.read(key), np.uint8), cv2.IMREAD_COLOR)
                    assert image.shape[:2] == disp.shape
                    rgb = cv2.cvtColor(cv2.resize(image, (rw,rh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    if light == "L0" and exposure == 2 and side == 0:
                        with np.load(STORE/f"{scene}_wide.npz") as ref:
                            assert np.array_equal(rgb,ref["rgb"])
                    add("real", f"{scene}/{side}", f"{light}e{exposure}", rgb, z, valid, fx*rw/w)
    np.savez_compressed(PREPARED, metadata=metadata, **arrays)
    print("Prepared conditions", len(metadata), flush=True)


def infer(reuse=False):
    print("Loading PyTorch", flush=True)
    import torch

    sys.path.insert(0, str(SOURCES/"ml-depth-pro-2"/"src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    torch.set_num_threads(2)
    preset = replace(PRESETS["depthpro2-s"], fov_checkpoint_uri=None)
    model, transform = create_model_and_transforms(preset, device="cuda")
    records = {}
    previous = OUTPUT/"paired_conditions_initial.npz"
    if reuse:
        with np.load(PREDICTED) as old, np.load(previous) as old_data, np.load(PREPARED) as new_data:
            assert np.array_equal(old_data["metadata"], new_data["metadata"])
            for i, (_, _, label, _) in enumerate(new_data["metadata"]):
                if label != "global":
                    assert np.array_equal(old_data[f"{i}_rgb"], new_data[f"{i}_rgb"])
                    records[str(i)] = old[str(i)]
    with np.load(PREPARED) as data, torch.inference_mode():
        for i, (domain, group, label, fx) in enumerate(data["metadata"]):
            if str(i) in records:
                continue
            rgb = data[f"{i}_rgb"]
            z = model.infer(transform(rgb), f_px=float(fx))["depth"].float().cpu().numpy()
            assert z.shape == rgb.shape[:2]
            records[str(i)] = z
            if (i+1)%16 == 0:
                print("Inferred", i+1, flush=True)
    np.savez_compressed(PREDICTED, **records)


def evaluate():
    from scipy.stats import rankdata

    def auc(score, bad):
        ranks = rankdata(score, method="average")
        n = int(bad.sum())
        return float((ranks[bad].sum() - n*(n+1)/2)/(n*(len(ranks)-n)))

    rows = []
    with np.load(PREPARED) as data, np.load(PREDICTED) as preds:
        info = data["metadata"]
        refs = {group:i for i, (domain, group, label, _) in enumerate(info)
                if (domain,label) in (("real","L0e2"),("synthetic","reference"))}
        groups = {group:[i for i, entry in enumerate(info) if entry[1]==group] for group in refs}
        for i, (domain, group, label, _) in enumerate(info):
            z = data[f"{i}_z"]
            mask = data[f"{i}_valid"]
            for j in groups[group]:
                value = preds[str(j)]
                mask &= np.isfinite(value) & (value > 0)
            pred, prior = preds[str(i)][mask], preds[str(refs[group])][mask]
            assert len(pred) > 100
            rgb = linear(data[f"{i}_rgb"][mask])
            luma = rgb @ [.2126, .7152, .0722]
            error = np.abs(pred-z[mask])/z[mask]
            evidence = {}
            if i != refs[group]:
                base = linear(data[f"{refs[group]}_rgb"][mask])
                difference = np.log(rgb+.01) - np.log(base+.01)
                centered = difference - np.median(difference, axis=0)
                bad = error > np.quantile(error, .8)
                evidence = dict(raw_rgb_auc=auc(np.linalg.norm(difference, axis=1), bad),
                                centered_rgb_auc=auc(np.linalg.norm(centered, axis=1), bad))
            rows.append(dict(domain=domain,group=group,label=label,n=int(mask.sum()),
                             absrel=float(error.mean()),
                             median_change=float(np.median(np.abs(pred-prior)/prior)),
                             mean_linear_luma=float(luma.mean()), **evidence))
    summary={}
    for domain in ("real","synthetic"):
        labels=sorted({r["label"] for r in rows if r["domain"]==domain})
        summary[domain] = {}
        for label in labels:
            group_rows = [row for row in rows if row["domain"] == domain and row["label"] == label]
            summary[domain][label] = {
                key: float(np.mean([row[key] for row in group_rows]))
                for key in ("absrel", "median_change", "mean_linear_luma", "raw_rgb_auc", "centered_rgb_auc")
                if key in group_rows[0]
            }
    save("lighting_probe_results.json", dict(summary=summary,rows=rows))
    print(summary, flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","infer","refresh-global","evaluate"))
    {"prepare":prepare,"infer":infer,"refresh-global":lambda: infer(reuse=True),
     "evaluate":evaluate}[parser.parse_args().stage]()
