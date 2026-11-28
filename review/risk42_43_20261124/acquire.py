"""Fetch authors' full-resolution left/right photographs and structured-light uncertainty."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2]/"research-sources"/"risk42_middlebury"
DEV = ("Adirondack","Backpack")
TEST = ("Bicycle1","Cable","Classroom1","Couch","Flowers","Jadeplant","Mask","Motorcycle","Piano",
        "Pipes","Playroom","Playtable","Recycle","Shelves","Shopvac","Sticks","Storage","Sword1","Sword2","Umbrella","Vintage")
FILES = ("calib.txt","im0.png","im1.png","disp0.pfm","disp0-n.pgm","disp0-sd.pfm")
BASE = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets/"


def fetch(job):
    scene,name=job
    dst=CACHE/scene/name
    dst.parent.mkdir(exist_ok=True,parents=True)
    if not dst.exists():
        for attempt in range(4):
            try:
                with requests.get(BASE+scene+"-perfect/"+name,stream=True,timeout=(15,90)) as response:
                    response.raise_for_status()
                    tmp=dst.with_suffix(dst.suffix+".partial")
                    with tmp.open("wb") as f:
                        for chunk in response.iter_content(1<<19):
                            f.write(chunk)
                    if name.endswith((".pgm",".pfm")):
                        with tmp.open("rb") as f:
                            magic=f.readline().strip()
                            width,height=map(int,f.readline().split())
                            f.readline()
                            assert magic in (b"Pf",b"P5")
                            assert tmp.stat().st_size == f.tell()+width*height*(4 if magic==b"Pf" else 1)
                    tmp.replace(dst)
                break
            except (requests.RequestException,AssertionError,ValueError,OSError):
                if attempt==3:
                    raise
                time.sleep(1+attempt)
    with dst.open("rb") as f:
        return scene+"/"+name,hashlib.file_digest(f,"sha256").hexdigest()


if __name__ == "__main__":
    # The first pass also makes color-only annotation possible during acquisition.
    jobs=[(s,n) for n in FILES for s in DEV+TEST]
    with ThreadPoolExecutor(8) as pool:
        manifest=dict(pool.map(fetch,jobs))
    (HERE/"sources.json").write_text(json.dumps(manifest,indent=1),encoding="utf8")
    print("Acquired files",len(manifest),flush=True)
