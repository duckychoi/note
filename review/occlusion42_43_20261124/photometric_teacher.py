"""Recover normals from calibrated image brightness and score physical depth ratios."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from depth_full_data import hypersim_intrinsics
from material_light_assets import OUT as SOURCE, save_json


OUT = SOURCE.parent / "photometric-stereo-43"
MATERIALS = ("matte", "coat")
N_LIGHTS = 8
STRIDE = 4
STEP = 8 // STRIDE


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def linear_intensity(image):
    mapped = (image.astype(np.float32) / 255) ** 2.2
    linear = mapped / np.maximum(1 - mapped, 1e-5)
    return linear @ np.array([.2126, .7152, .0722], np.float32)


def recover(y, lights, nonsaturated):
    angles = np.zeros((len(y), 3), np.float32)
    reliable = np.zeros(len(y), bool)
    for start in range(0, len(y), 8192):
        stop = min(start + 8192, len(y))
        values = y[start:stop]
        active = (values > .008) & nonsaturated[start:stop]
        scale = np.maximum(np.max(np.where(active, values, 0), axis=1), .008)
        weight = active.astype(np.float32)
        for iteration in range(5):
            matrix = np.einsum("ni,ij,ik->njk", weight, lights, lights, optimize=True)
            rhs = (weight * values) @ lights
            g = np.linalg.solve(matrix + 1e-6 * np.eye(3, dtype=np.float32), rhs[..., None])[..., 0]
            residual = g @ lights.T - values
            if iteration < 4:
                weight = active * np.minimum(1., .035 * scale[:, None] / np.maximum(abs(residual), 1e-6))
        # Score residual over the actual positively lit observations, including downweighted ones.
        rms = np.sqrt(((residual**2) * active).sum(1) / np.maximum(active.sum(1), 1))
        norm = np.linalg.norm(g, axis=1)
        angles[start:stop] = g / np.maximum(norm[:, None], 1e-8)
        reliable[start:stop] = ((active.sum(1) >= 5) & (np.linalg.eigvalsh(matrix)[:, 0] >= .35**2)
                                & (rms <= .12 * scale) & (norm > 1e-7))
    return angles, reliable


def endpoints(values, axis):
    if axis == 0:
        return values[:, :-STEP], values[:, STEP:]
    return values[:-STEP, :], values[STEP:, :]


def constraints(normal, trusted, z, valid, entity, rays):
    logz = np.log(np.where(valid, z, 1.))
    for axis in (0, 1):
        zp, zq = endpoints(logz, axis)
        vp, vq = endpoints(valid, axis)
        ip, iq = endpoints(entity, axis)
        tp, tq = endpoints(trusted, axis)
        np_, nq = endpoints(normal, axis)
        rp, rq = endpoints(rays, axis)
        exact = zq - zp
        on_surface = vp & vq & (ip == iq) & (abs(exact) <= np.log(1.05))
        predictions = []
        usable = on_surface & tp & tq
        for n in (np_, nq):
            a = np.sum(n * rp, axis=-1)
            b = np.sum(n * rq, axis=-1)
            usable &= (a < -.2) & (b < -.2)
            predictions.append(np.log(np.maximum(-a, .2)) - np.log(np.maximum(-b, .2)))
        yield .5 * (predictions[0] + predictions[1]), exact, usable, on_surface


def evaluate(folder):
    outpath = folder / "teacher.npz"
    report = folder / "teacher.json"
    if outpath.is_file() and report.is_file():
        return
    while not (folder / "render.json").is_file():
        time.sleep(2)
    meta = json.loads((folder / "render.json").read_text())
    lights = np.array(meta["light_vectors"], np.float32)
    with np.load(folder / "geometry.npz") as data:
        h, w = data["z"].shape
        z = data["z"][::STRIDE, ::STRIDE]
        valid = data["valid"][::STRIDE, ::STRIDE] & (z >= .25) & (z <= 30)
        entity = data["entity"][::STRIDE, ::STRIDE]
        truth = data["normal"][::STRIDE, ::STRIDE].astype(np.float32)
    yy, xx = np.indices(z.shape, dtype=np.float32) * STRIDE
    k = hypersim_intrinsics(meta["scene"], w, h)
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(k).astype(np.float32).T
    ambient_rgb = rgb(folder / "ambient.png")[::STRIDE, ::STRIDE]
    ambient = linear_intensity(ambient_rgb)
    arrays, scores = {}, {}
    for material in MATERIALS:
        inputs = [rgb(folder / f"{material}-{i}.png")[::STRIDE, ::STRIDE] for i in range(N_LIGHTS)]
        nonsaturated = np.stack([(x < 254).all(2) & (ambient_rgb < 254).all(2) for x in inputs], axis=-1)
        values = np.stack([linear_intensity(x) - ambient for x in inputs], axis=-1)
        normal, trust = recover(values.reshape(-1, N_LIGHTS), lights, nonsaturated.reshape(-1, N_LIGHTS))
        normal, trust = normal.reshape(*z.shape, 3), trust.reshape(z.shape)
        trust &= np.sum(normal * rays, axis=-1) < -.2
        compare = valid & trust
        dot = np.sum(truth * normal, axis=-1) / np.maximum(np.linalg.norm(truth, axis=-1), 1e-7)
        angle = np.rad2deg(np.arccos(np.clip(dot[compare], -1, 1)))
        stats = dict(accepted=int(compare.sum()), valid=int(valid.sum()), angular_sum=float(angle.sum(dtype=np.float64)),
                     covered_edges=0, surface_edges=0, log_ratio_squared=0., flat_squared=0.)
        arrays[f"{material}_normal"] = normal.astype(np.float16)
        arrays[f"{material}_trust"] = trust
        for axis, (target, exact, usable, surface) in enumerate(constraints(normal, trust, z, valid, entity, rays)):
            stats["covered_edges"] += int(usable.sum())
            stats["surface_edges"] += int(surface.sum())
            stats["log_ratio_squared"] += float(((target[usable] - exact[usable])**2).sum(dtype=np.float64))
            stats["flat_squared"] += float((exact[usable]**2).sum(dtype=np.float64))
            arrays[f"{material}_{axis}_target"] = target.astype(np.float16)
            arrays[f"{material}_{axis}_mask"] = usable
        scores[material] = stats
    np.savez_compressed(folder / "teacher.part.npz", **arrays)
    (folder / "teacher.part.npz").replace(outpath)
    save_json(report, scores)
    print("teacher", meta["split"], meta["scene"], {k: v["accepted"] for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    rows = json.loads((SOURCE / "scenes.json").read_text())
    for row in rows:
        evaluate(OUT / "renders" / row["scene"])
