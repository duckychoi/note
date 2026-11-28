"""Acquire only the official material/normal/ID passes for one fixed camera per Hypersim scene."""
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

from hypersim_fetch import ARCHIVE, ROOT, RemoteZip, frame_assets, has_asset, scene_pose_assets


OUT = ROOT.parent/"encoder42_eval/material-light-43"


def digest(path):
    with path.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_json(path,data):
    tmp = path.with_suffix(".part.json")
    tmp.write_text(json.dumps(data,indent=1,allow_nan=False),encoding="utf8")
    tmp.replace(path)


def select():
    groups = defaultdict(lambda:defaultdict(list))
    with (ROOT/"splits.csv").open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["included_in_public_release"]=="True":
                groups[r["split_partition_name"]][r["scene_name"],r["camera_name"]].append(int(r["frame_id"]))
    items,used = [],set()
    for split,n in (("val",16),("train",32)):
        chosen = 0
        for (scene,cam), all_ids in sorted(groups[split].items(),
                key=lambda x:hashlib.sha256(("43-material-1"+str(x[0])).encode()).digest()):
            family = scene.rsplit("_",1)[0]
            if family in used:
                continue
            ids = sorted(i for i in all_ids if has_asset(frame_assets(scene,cam,i)[1][1]))
            if len(ids)<3:
                continue
            center = ids[len(ids)//2]
            neighbors = sorted({min(all_ids,key=lambda k:abs(k-(center+dt))) for dt in (-14,-7,0,7,14)})
            items.append(dict(split=split,scene=scene,family=family,cam=cam,frame=center,neighbors=neighbors))
            used.add(family)
            chosen += 1
            if chosen==n:
                break
    if not items:
        raise RuntimeError("Empty scene manifest")
    return items


def source_files(row):
    s,c,f = row["scene"],row["cam"],row["frame"]
    folder = OUT/"assets"/s
    source = [(f"images/scene_{c}_{group}/frame.{f:04d}.{kind}.hdf5",folder/(kind+".hdf5")) for group,kind in
              (("final_hdf5","diffuse_reflectance"),("geometry_hdf5","normal_cam"),("geometry_hdf5","render_entity_id"))]
    source += list(scene_pose_assets(s,[(c,f)]))
    for i in row["neighbors"]:
        suffix,path = frame_assets(s,c,i)[1]
        source.append(("images/"+suffix,path))
    return source


def fetch_one(row):
    scene = row["scene"]
    sources = source_files(row)
    missing = [(key,path) for key,path in sources if not has_asset(path)]
    if missing:
        with RemoteZip(f"{ARCHIVE}/{scene}.zip") as remote, ZipFile(remote) as arc:
            for key,path in missing:
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp = path.with_suffix(".download")
                data = arc.read(f"{scene}/{key}")
                if not data:
                    raise ValueError(key)
                tmp.write_bytes(data)
                tmp.replace(path)
    print("assets",scene,flush=True)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    items = select()
    save_json(OUT/"scenes.json",items)
    print("selected",[(s,sum(row["split"]==s for row in items)) for s in ("train","val")],flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch_one,items))
