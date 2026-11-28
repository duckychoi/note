"""Measure calibrated photometric normals on the public DiLiGenT main set."""
from io import BytesIO
import hashlib
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.io import loadmat

from photometric_teacher import N_LIGHTS, OUT, STRIDE, recover
from material_light_assets import save_json


def decode(archive, name):
    return cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_UNCHANGED)


def choose(lights):
    # Deterministic farthest-point sampling among actual calibrated directions.
    selected = [int(np.argmax(lights[:, 2]))]
    for _ in range(N_LIGHTS - 1):
        similarity = (lights @ lights[selected].T).max(axis=1)
        similarity[selected] = np.inf
        selected.append(int(np.argmin(similarity)))
    return selected


def diagnostics(archive, folder):
    def read(name):
        return BytesIO(archive.read(folder + name))

    lights = np.loadtxt(read("light_directions.txt")).astype(np.float32)
    strength = np.loadtxt(read("light_intensities.txt")).astype(np.float32)
    names = read("filenames.txt").getvalue().decode().split()
    assert len(names) == len(lights) == len(strength) == 96
    mask = decode(archive, folder + "mask.png")[::STRIDE, ::STRIDE]
    keep = mask == 255 if mask.ndim == 2 else (mask == 255).all(2)
    truth = loadmat(read("Normal_gt.mat"))["Normal_gt"][::STRIDE, ::STRIDE][keep]
    truth /= np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-8)
    samples, nonsaturated = [], []
    for i, name in enumerate(names):
        rgb = decode(archive, folder + name)[::STRIDE, ::STRIDE][keep, ::-1]
        if rgb.dtype != np.uint16:
            raise ValueError("Expected a radiometrically linear 16-bit exposure")
        samples.append((rgb.astype(np.float32) / (65535 * strength[i])) @
                       np.array([.2989, .5870, .1140], np.float32))
        nonsaturated.append((rgb < 65534).all(1))
    samples = np.stack(samples, axis=-1)
    nonsaturated = np.stack(nonsaturated, axis=-1)
    subset = choose(lights)
    estimate = samples @ np.linalg.pinv(lights).T
    estimate /= np.maximum(np.linalg.norm(estimate, axis=1, keepdims=True), 1e-8)
    sparse, reliable = recover(samples[:, subset], lights[subset], nonsaturated[:, subset])

    def angle(est, include):
        cosine = (est[include] * truth[include]).sum(1)
        values = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                    median=float(np.median(values)) if len(values) else None)

    return dict(pixels=int(keep.sum()), lights=[i + 1 for i in subset],
                least_squares_96=angle(estimate, np.ones(len(truth), bool)),
                least_squares_96_same_pixels=angle(estimate, reliable),
                robust_8=angle(sparse, reliable))


if __name__ == "__main__":
    cv2.setNumThreads(1)
    path = OUT / "DiLiGenT.zip"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(path) as archive:
        folders = sorted(name.removesuffix("Normal_gt.mat") for name in archive.namelist()
                         if "/pmsData/" in name and name.endswith("/Normal_gt.mat"))
        assert len(folders) == 10
        results = {folder.split("/")[-2]: diagnostics(archive, folder) for folder in folders}
    save_json(OUT / "teacher-diligent.json", dict(sha256=digest, stride=STRIDE, objects=results))
    for key, value in results.items():
        print(key, value, flush=True)
