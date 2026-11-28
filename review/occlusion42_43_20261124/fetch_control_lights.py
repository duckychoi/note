"""Fetch only the four official Middlebury exposures used by the crossed test."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from time import sleep
from zipfile import ZipFile

from PIL import Image
import requests

from common import ALL, STORE

BASE = "https://vision.middlebury.edu/stereo/data/scenes2021/data/"
TARGET = STORE / "lighting" / "crossed"


def get(name):
    scene, _, light, filename = name.split("/")
    path = TARGET / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        if scene.startswith("skates"):
            with ZipFile(TARGET.parent / "ambient-skates.zip") as archive:
                part.write_bytes(archive.read("data/"+name))
        else:
            for attempt in range(4):
                try:
                    with requests.get(BASE+name, stream=True, timeout=(12, 35)) as response:
                        if response.status_code == 404:
                            return name, "missing"
                        response.raise_for_status()
                        with part.open("wb") as f:
                            for chunk in response.iter_content(262144):
                                f.write(chunk)
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    sleep(2**attempt)
        with Image.open(part) as image:
            if image.format != "PNG":
                raise ValueError(f"Invalid PNG for {name}")
            image.verify()
        part.replace(path)
    with path.open("rb") as f:
        return name, hashlib.file_digest(f, "sha256").hexdigest()


if __name__ == "__main__":
    files = [f"{scene}/ambient/{light}/im{side}e0.png" for scene in ALL
             for light in ("L0", "L1") for side in (0, 1)]
    with ThreadPoolExecutor(2) as pool:
        result = dict(pool.map(get, files))
    manifest = TARGET / "source_files.json"
    tmp = manifest.with_suffix(".partial")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf8")
    tmp.replace(manifest)
    print("Available", sum(value != "missing" for value in result.values()), "of", len(files))
