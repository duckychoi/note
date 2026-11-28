"""Streaming data for the Depth Pro 2-S training comparison."""
import csv
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import tarfile

import h5py
import numpy as np
from PIL import Image

from encoder_expand_data import selection_for
from hypersim_fetch import ROOT, frame_assets, has_asset, scene_pose_assets


def h5_array(path):
    with h5py.File(path, "r") as file:
        return file["dataset"][:]


@lru_cache(maxsize=None)
def calibration_table():
    with (ROOT / "camera.csv").open(newline="", encoding="utf8") as stream:
        return {row["scene_name"]: row for row in csv.DictReader(stream)}


@lru_cache(maxsize=1200)
def poses(scene, camera):
    assets = list(scene_pose_assets(scene, [(camera, 0)]))
    with assets[0][1].open(newline="", encoding="utf8") as stream:
        scale = float(next(row["parameter_value"] for row in csv.DictReader(stream)
                           if row["parameter_name"] == "meters_per_asset_unit"))
    ids, positions, rotations = [h5_array(path) for _, path in assets[1:]]
    rotations = rotations @ np.diag([1, -1, -1])
    if len(set(ids.ravel())) != len(ids) or not (len(ids) == len(positions) == len(rotations)):
        raise ValueError(f"Invalid camera pose indexing: {scene}/{camera}")
    finite = np.isfinite(rotations).all(axis=(1, 2)) & np.isfinite(positions).all(axis=1)
    orthogonal = np.isclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5).all(axis=(1, 2))
    valid = finite & orthogonal & np.isclose(np.linalg.det(rotations), 1., atol=1e-5)
    return {int(frame): (rotation, position * scale) for frame, rotation, position, ok in
            zip(ids.ravel(), rotations, positions, valid) if ok}


@lru_cache(maxsize=1024)
def hypersim_intrinsics(scene, width, height):
    row = calibration_table()[scene]
    m = np.array([[float(row[f"M_proj_{a}{b}"]) for b in range(4)] for a in range(4)])
    return np.array([[width*m[0,0]/2, -width*m[0,1]/2, width*(1-m[0,2])/2-.5],
                     [-height*m[1,0]/2, height*m[1,1]/2, height*(1+m[1,2])/2-.5], [0, 0, 1]])


@lru_cache(maxsize=64)
def camera_ray_lengths(shape, intrinsic_values):
    height, width = shape
    y, x = np.indices((height, width), dtype=np.float32)
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    inverse = np.linalg.inv(np.array(intrinsic_values, dtype=np.float32).reshape(3, 3))
    return np.linalg.norm(pixels @ inverse.T,axis=-1)


def make_sample(rgb, depth, intrinsics, stride, distance=False, valid=None):
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB and depth shapes disagree: {rgb.shape}, {depth.shape}")
    rgb = np.ascontiguousarray(rgb[::stride, ::stride])
    depth = depth[::stride, ::stride].astype(np.float32)
    intrinsics = intrinsics.copy()
    intrinsics[:2] /= stride
    if distance:
        depth /= camera_ray_lengths(depth.shape,tuple(intrinsics.ravel()))
    valid_depth = np.isfinite(depth) & (depth >= 0.25) & (depth <= 30.)
    if valid is not None:
        valid_depth &= valid[::stride, ::stride].astype(bool)
    return rgb, np.where(valid_depth, depth, 1).astype(np.float32), valid_depth, intrinsics.astype(np.float32)


class Hypersim:
    def __init__(self, stage, stride, pair, smoke=False):
        self.stride, self.pair = stride, pair
        self.groups, self.rows = {}, []
        for scene, frames in selection_for(stage).items():
            has_poses = not pair or all(has_asset(path) for _, path in scene_pose_assets(scene, frames))
            if smoke and not has_poses:
                continue
            for camera, frame in frames:
                available = all(has_asset(path) for _, path in frame_assets(scene, camera, frame))
                if smoke and not available:
                    continue
                if not available or not has_poses:
                    raise FileNotFoundError(f"Incomplete {stage}: {scene}/{camera}/{frame}")
                self.groups.setdefault((scene, camera), []).append(int(frame))
            if smoke and len(self.groups) >= 3:
                break
        for (scene, camera), frames in self.groups.items():
            for i, frame in enumerate(frames):
                # Match an adjacent position clip without changing the anchor sampling distribution.
                neighbor = frames[i + 3] if i + 3 < len(frames) else frames[max(0, i - 3)]
                self.rows.append((scene, camera, frame, neighbor))

    def __len__(self):
        return len(self.rows)

    def read_frame(self, scene, camera, frame):
        (_, rgb_path), (_, depth_path) = frame_assets(scene, camera, frame)
        with Image.open(rgb_path) as file:
            rgb = np.array(file.convert("RGB"))
        height, width = rgb.shape[:2]
        return make_sample(rgb, h5_array(depth_path), hypersim_intrinsics(scene, width, height),
                           self.stride, distance=True)

    def __getitem__(self, index):
        return self.read_pair(self.rows[index])

    def read_pair(self, row):
        scene, camera, frame, neighbor = row
        anchor = self.read_frame(scene, camera, frame)
        if not self.pair:
            return anchor, None, None
        other = self.read_frame(scene, camera, neighbor)
        trajectory = poses(scene, camera)
        if frame not in trajectory or neighbor not in trajectory:
            # Retain both metric RGB-D targets. Only the geometric correspondence
            # requires a rigid transformation for these particular frames.
            return anchor, other, None
        rotation, center = trajectory[frame]
        other_rotation, other_center = trajectory[neighbor]
        matrix = np.column_stack((other_rotation.T @ rotation, other_rotation.T @ (center-other_center)))
        return anchor, other, matrix.astype(np.float32)


class NyuDepth:
    """The archived aligned NYU RGB-D repack, indexed without extracting duplicate files."""

    def __init__(self, folder, split, stride, smoke=False):
        self.stride, self.rows = stride, []
        prefixes = ("train",) if split == "train" else ("val", "validation", "test")
        files = sorted(path for prefix in prefixes for path in Path(folder).glob(f"nyu-depth-{prefix}-*.tar"))
        if not files:
            raise FileNotFoundError(f"NYU {split} shards in {folder}")
        if split == "train" and not smoke and len(files) != 48:
            raise ValueError(f"Incomplete NYU training set: {len(files)} shards")
        for path in files[:1] if smoke else files:
            with tarfile.open(path, "r:") as archive:
                entries = {member.name: member for member in archive if member.isfile()}
            for name, rgb in entries.items():
                if not name.endswith(".jpg"):
                    continue
                depth = entries[name.removesuffix(".jpg") + ".depth.npy"]
                self.rows.append((path, rgb.offset_data, rgb.size, depth.offset_data, depth.size))
        expected = 47584 if split == "train" else 654
        if not smoke and len(self.rows) != expected:
            raise ValueError(f"NYU {split}: expected {expected} rows, found {len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, rgb_offset, rgb_size, depth_offset, depth_size = self.rows[index]
        with path.open("rb") as stream:
            stream.seek(rgb_offset)
            with Image.open(BytesIO(stream.read(rgb_size))) as file:
                rgb = np.array(file.convert("RGB"))
            stream.seek(depth_offset)
            depth = np.load(BytesIO(stream.read(depth_size)), allow_pickle=False).squeeze()
        # The repack center-crops 16 px without resizing. NYUv2 RGB intrinsics stay in pixel units.
        intrinsics = np.array([[518.8579, 0, 325.5824-16], [0, 519.4696, 253.7362-16], [0, 0, 1]])
        return make_sample(rgb, depth, intrinsics, self.stride), None, None
