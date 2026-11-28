"""Fetch only the official Hypersim files used in the Depth Pro 2 pilot."""
from concurrent.futures import ThreadPoolExecutor
import csv
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[3] / "research-sources" / "hypersim42_pilot"
ARCHIVE = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
METADATA = "https://raw.githubusercontent.com/apple/ml-hypersim/main"
SCENES = tuple(f"ai_{i:03d}_001" for i in range(1, 7))
FRAMES = (0, 11, 22, 33, 44, 55, 66, 77)


def fetch(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


class RemoteZip(io.RawIOBase):
    def __init__(self, url, block_size=131072):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
        )))
        self.url = url
        self.block_size = block_size
        self.pos = 0
        self.block = None
        with self.session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30) as r:
            if r.status_code != 206:
                raise OSError(f"Archive did not honor byte ranges: {r.status_code}")
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, length=-1):
        stop = self.size if length < 0 else min(self.size, self.pos + length)
        if stop <= self.pos:
            return b""
        if self.block is not None:
            start, data = self.block
            if start <= self.pos and stop <= start + len(data):
                out = data[self.pos - start:stop - start]
                self.pos = stop
                return out
        end = min(self.size, max(stop, self.pos + self.block_size))
        # HTTPAdapter retries before the body is consumed. Resume the same byte
        # range when the connection drops partway through a streamed response.
        for attempt in range(5):
            try:
                with self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end-1}"}, stream=True, timeout=45) as r:
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {self.pos}-{end-1}/{self.size}":
                        raise ValueError(f"Invalid archive range: {r.status_code} {r.headers.get('Content-Range')}")
                    data = r.content
                    if len(data) != end - self.pos:
                        raise EOFError("Incomplete archive range")
                break
            except (requests.exceptions.ChunkedEncodingError, requests.ConnectionError, requests.Timeout, EOFError):
                if attempt == 4:
                    raise
                sleep(2**attempt)
        self.block = (self.pos, data)
        out = data[:stop - self.pos]
        self.pos = stop
        return out

    def close(self):
        self.session.close()
        super().close()


def scene_files(scene, frames):
    scene_camera_files(scene, [("cam_00", frame) for frame in frames])


def frame_assets(scene, camera, frame):
    folder = ROOT / scene
    if camera != "cam_00":
        folder = folder / camera
    pairs = (("final_preview", "tonemap.jpg"), ("geometry_hdf5", "depth_meters.hdf5"))
    return [(f"scene_{camera}_{part}/frame.{frame:04d}.{kind}", folder / f"{frame:04d}.{kind}")
            for part, kind in pairs]


def has_asset(path):
    return path.is_file() and path.stat().st_size > 0


def scene_camera_files(scene, camera_frames, include_poses=False):
    missing = [(suffix, path) for camera, frame in camera_frames
               for suffix, path in frame_assets(scene, camera, frame) if not has_asset(path)]
    if include_poses:
        missing.extend((suffix, path) for suffix, path in scene_pose_assets(scene, camera_frames)
                       if not has_asset(path))
    if missing:
        block_size = 4 << 20 if len(camera_frames) >= 32 else 131072
        with RemoteZip(f"{ARCHIVE}/{scene}.zip", block_size) as remote, zipfile.ZipFile(remote) as archive:
            members = {info.filename.split("/", 1)[-1]: info for info in archive.infolist()}
            entries = [(members[suffix if suffix.startswith("_detail/") else f"images/{suffix}"], path)
                       for suffix, path in missing]
            for info, path in sorted(entries, key=lambda item: item[0].header_offset):
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = archive.read(info)
                stream = NamedTemporaryFile(dir=path.parent, delete=False)
                temp = Path(stream.name)
                try:
                    with stream:
                        stream.write(payload)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
    print(scene, len(camera_frames), flush=True)


def scene_pose_assets(scene, camera_frames):
    scene_dir = ROOT / scene
    yield "_detail/metadata_scene.csv", scene_dir / "metadata_scene.csv"
    for camera in sorted({camera for camera, _ in camera_frames}):
        folder = scene_dir if camera == "cam_00" else scene_dir / camera
        for part in ("frame_indices", "positions", "orientations"):
            filename = f"camera_keyframe_{part}.hdf5"
            yield f"_detail/{camera}/{filename}", folder / filename


def main():
    ROOT.mkdir(exist_ok=True)
    metadata = {
        "splits.csv": "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv",
        "camera.csv": "contrib/mikeroberts3000/metadata_camera_parameters.csv",
    }
    for name, path in metadata.items():
        target = ROOT / name
        if not target.exists():
            target.write_bytes(fetch(f"{METADATA}/{path}"))
    with (ROOT / "splits.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    jobs = [(scene, [int(r["frame_id"]) for r in rows if r["scene_name"] == scene
                    and r["camera_name"] == "cam_00" and r["included_in_public_release"] == "True"
                    and int(r["frame_id"]) in FRAMES]) for scene in SCENES]
    assert all(frames for _, frames in jobs)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda job: scene_files(*job), jobs))


if __name__ == "__main__":
    main()
