"""Select a second view from the same trajectory using only training-simulator geometry."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import time

import h5py
import numpy as np

from controlled_observations import RUN, wait_for_ram, write_json
from depth_full_data import hypersim_intrinsics, poses
from encoder_expand_data import selection_for
from hypersim_fetch import frame_assets, has_asset, scene_pose_assets
from stereo_controlled_observations import rectified_cameras


OUT = RUN.parent/"control43-geometry-pairs-v1"
PITCH = 32


@lru_cache(maxsize=256)
def coarse(scene, camera, frame):
    path = frame_assets(scene, camera, frame)[1][1]
    with h5py.File(path) as f:
        data = f["dataset"]
        h, w = data.shape
        distance = data[::PITCH, ::PITCH].astype(float)
    k = hypersim_intrinsics(scene, w, h).copy()
    k[:2] /= 4
    y,x = np.indices(distance.shape)*(PITCH/4)
    rays = np.stack((x,y,np.ones_like(x)), axis=-1) @ np.linalg.inv(k).T
    z = distance/np.linalg.norm(rays,axis=-1)
    valid = np.isfinite(z) & (z >= .25) & (z <= 30)
    return k, ((w+3)//4, (h+3)//4), np.where(valid,z,0), rays, valid


def relative(first, second):
    r, c = first
    s, d = second
    return np.column_stack((s.T @ r, s.T @ (c-d))).astype(np.float32)


def visibility(first, second, pose, rect):
    k,size,z,rays,valid = first
    k1,_,z1,_,valid1 = second
    n = max(1,valid.sum())
    points = rays*z[...,None]
    moved = points @ pose[:,:3].T+pose[:,3]
    xy = moved @ k1.T
    xy = xy[...,:2]/np.maximum(xy[...,2:],1.e-7)
    u,v = np.moveaxis(xy,-1,0)
    x = np.clip(np.rint(u/(PITCH/4)),0,z1.shape[1]-1).astype(int)
    y = np.clip(np.rint(v/(PITCH/4)),0,z1.shape[0]-1).astype(int)
    same = (valid & (moved[...,2]>0) & (u>=0) & (v>=0) & (u<size[0]-1) & (v<size[1]-1)
            & valid1[y,x] & (abs(moved[...,2]-z1[y,x]) <= .05*z1[y,x]))
    if rect is None:
        return float(same.sum()/n), 0.
    ra, rb, pa, pb, axis = rect
    search_width = size[axis]
    count = min(96,search_width//2//16*16)
    # OpenCV discards a strip on each side of a bidirectionally matched pair.
    for cloud,rot,proj,left in ((points,ra,pa,True),(moved,rb,pb,False)):
        coord = cloud @ rot.T @ proj[:,:3].T
        pix = coord[...,:2]/np.maximum(coord[...,2:],1.e-7)
        for j in (0,1):
            same_rect = (pix[...,j]>=1) & (pix[...,j]<size[j]-2)
            if j == 0:
                keep = same_rect
            else:
                keep &= same_rect
        oriented = pix[...,axis] if pb[axis,3] < 0 else search_width-1-pix[...,axis]
        keep &= oriented >= count if left else oriented < search_width-count
        if left:
            feasible = same & keep
            origin = oriented
        else:
            feasible &= keep & (origin-oriented>.25) & (origin-oriented<count-1)
    return float(same.sum()/n), float(feasible.sum()/n)


def shortlist(scene, camera, anchor, frames):
    traj = poses(scene,camera)
    if anchor not in traj:
        return []
    first = coarse(scene,camera,anchor)
    k,size = first[:2]
    ranked = []
    median_z=max(.25,float(np.median(first[2][first[4]])) if first[4].any() else 1.)
    for b in frames:
        if b == anchor or b not in traj:
            continue
        p = relative(traj[anchor],traj[b])
        lateral = min(np.linalg.norm(p[:2,3]),np.linalg.norm((p[:,:3].T@p[:,3])[:2]))
        if lateral < .001:
            continue
        rect,why = rectified_cameras([k,k], size, p)
        if rect is None:
            continue
        # Prefer lateral translation, similar headings and a moderate image disparity.
        angle = np.arccos(np.clip((np.trace(p[:,:3])-1)/2,-1,1))
        disparity = k[0,0]*lateral/median_z
        prior = lateral/np.linalg.norm(p[:,3])*np.exp(-angle-abs(np.log(max(.01,disparity)/12))*.3)
        ranked.append((float(prior),b,p,rect))
    return sorted(ranked,key=lambda t:(-t[0],t[1]))[:8]


def choose(scene, camera, anchor, old, frames):
    if anchor not in poses(scene,camera):
        return dict(row=[scene,camera,anchor,old], reason="missing_anchor_pose")
    first=coarse(scene,camera,anchor)
    best = None
    for prior,b,p,rect in shortlist(scene,camera,anchor,frames):
        overlap, searchable = visibility(first,coarse(scene,camera,b),p,rect)
        key = (searchable,overlap,prior,-b)
        if best is None or key > best[0]:
            best = (key,b)
    if best is None or best[0][0] <= 0:
        return dict(row=[scene,camera,anchor,old], reason="fallback_no_searchable_pair")
    return dict(row=[scene,camera,anchor,best[1]], reason="geometry_selected",overlap=best[0][1],
                searchable=best[0][0])


def groups(stage):
    out = defaultdict(list)
    for scene, entries in selection_for(stage).items():
        for camera, frame in entries:
            out[scene,camera].append(frame)
    return out


def run(full):
    OUT.mkdir(exist_ok=True)
    for split,stage in ((("full","train_full"),) if full else (("dev","validation_24"),("train","train_full"))):
        catalog = groups(stage)
        if full:
            source = [(s,c,n,fs[i+3] if i+3<len(fs) else fs[max(0,i-3)])
                      for (s,c),fs in catalog.items() for i,n in enumerate(fs)]
        else:
            source = [json.loads(p.read_text())["row"] for p in sorted(RUN.glob(f"{split}-*.json"))]
        by_scene = defaultdict(list)
        for row in source:
            by_scene[row[0]].append(row)
        rows = []
        for scene, jobs in by_scene.items():
            wait_for_ram()
            file = OUT/f"pairs-{split}-{scene}.json"
            if not file.exists():
                if full:
                    entries=[(row[1],row[2]) for row in jobs]
                    while not (all(has_asset(p) for _,p in scene_pose_assets(scene,entries))
                               and all(has_asset(p) for c,f in entries for _,p in frame_assets(scene,c,f))):
                        time.sleep(30)
                selected = [choose(*row,catalog[row[0],row[1]]) for row in jobs]
                write_json(file,selected)
            current = json.loads(file.read_text())
            if [r["row"][:3] for r in current] != [list(j[:3]) for j in jobs]:
                raise ValueError(f"Anchor mismatch: {file}")
            rows.extend(r["row"] for r in current)
        write_json(OUT/f"pairs-{split}.json",rows)
        print(split,len(rows),hashlib.sha256(repr(rows).encode()).hexdigest(),flush=True)


if __name__ == "__main__":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x4000)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true")
    run(parser.parse_args().full)
