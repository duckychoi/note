"""Render a fixed Hypersim view with calibrated distant lights and ambient alone."""
import json
import time

import numpy as np
import open3d as o3d
import psutil
from PIL import Image

from material_light_render import OUT as SOURCE, dot, h5_array, points, unit, visibility_scene
from material_light_assets import save_json


OUT = SOURCE.parent / "photometric-stereo-43"
MATERIALS = ("matte", "coat")
N_LIGHTS = 8


def encode(path, radiance, valid):
    image = np.rint(255 * (radiance / (1 + radiance)) ** (1 / 2.2)).astype(np.uint8)
    image[~valid] = 100
    Image.fromarray(image).save(path, compress_level=2)


def shadows(scene, xyz, light, usable):
    xyz = xyz[usable]
    if not xyz.size:
        return np.zeros(usable.shape, bool)
    visible = []
    for chunk in np.array_split(xyz, (len(xyz) + 99999) // 100000):
        direction = np.broadcast_to(light, chunk.shape)
        offset = np.maximum(.004 * chunk[:, 2], .002)
        rays = np.column_stack((chunk + offset[:, None] * direction, direction)).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)["t_hit"].numpy()
        visible.append(np.isfinite(hit))
    blocked = np.zeros(usable.shape, bool)
    blocked[usable] = np.concatenate(visible)
    return blocked


def render(row):
    folder = OUT / "renders" / row["scene"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "render.json").is_file():
        return
    while psutil.virtual_memory().available < 8 * 2**30:
        time.sleep(4)
    assets = SOURCE / "assets" / row["scene"]
    albedo = h5_array(assets / "diffuse_reflectance.hdf5").astype(np.float32)
    normal = h5_array(assets / "normal_cam.hdf5").astype(np.float32) * np.array([1, -1, -1], np.float32)
    entity = h5_array(assets / "render_entity_id.hdf5")
    xyz = points(row, row["frame"])
    valid = (np.isfinite(xyz).all(2) & np.isfinite(albedo).all(2) & np.isfinite(normal).all(2)
             & (np.linalg.norm(normal, axis=-1) > .5) & (xyz[:, :, 2] > .05) & (entity >= 0))
    xyz = np.nan_to_num(xyz, nan=0., posinf=0., neginf=0.)
    normal = unit(np.nan_to_num(normal))
    albedo = np.clip(np.nan_to_num(albedo), 0, 1)
    view = -unit(xyz)
    normal *= np.where(dot(normal, view) >= 0, 1, -1).astype(np.float32)[..., None]
    nv = np.clip(dot(normal, view), .001, 1)
    scene, count, frames = visibility_scene(row, xyz)
    ambient = .12 * albedo
    encode(folder / "ambient.png", ambient, valid)
    light_vectors = []
    for i in range(N_LIGHTS):
        az, el = np.deg2rad([45 * i, 30 if i % 2 else 55])
        light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)], np.float32)
        power = 3.
        light_vectors.append((power * light).tolist())
        nl = np.maximum(dot(normal, light), 0)
        blocked = shadows(scene, xyz, light, valid & (nl > 0))
        direct = power * nl * (~blocked)
        matte = albedo * direct[..., None] / np.pi
        halfway = unit(light + view)
        nh, vh = np.clip(dot(normal, halfway), 0, 1), np.clip(dot(view, halfway), 0, 1)
        alpha2 = .18**2
        distribution = alpha2 / (np.pi * ((alpha2 - 1) * nh**2 + 1)**2)

        def smith(cosine):
            return 2 * cosine / (cosine + np.sqrt(alpha2 + (1 - alpha2) * cosine**2))

        fresnel = .04 + .96 * (1 - vh)**5
        specular = distribution * smith(nv) * smith(nl) * fresnel / (4 * nv * np.maximum(nl, 1e-6))
        coat = matte * (1 - fresnel[..., None]) + (direct * specular)[..., None]
        for material, component in zip(MATERIALS, (matte, coat)):
            encode(folder / f"{material}-{i}.png", ambient + component, valid)
    path = folder / "geometry.npz"
    np.savez_compressed(path, z=xyz[..., 2], normal=normal.astype(np.float16), valid=valid, entity=entity)
    save_json(folder / "render.json", dict(**row, light_vectors=light_vectors, triangles=count, support_frames=frames))
    print("rendered", row["split"], row["scene"], flush=True)


if __name__ == "__main__":
    for row in json.loads((SOURCE / "scenes.json").read_text()):
        render(row)
