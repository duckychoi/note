"""Identify reference grid points within 1 mm of a posed object CAD surface."""
import hashlib
import io
import json
import zipfile

import cv2
import numpy as np
import requests
from scipy.spatial import cKDTree

from robi_diagnostic import calibration, conservative_occlusion, grid, pose, project, read_depth, render
from robi_files import CACHE, OBJECT, cached, output_file, selection


CAD_URL="https://drive.usercontent.google.com/download?id=1WkL4DHY-HpwsOIJ3f0Ef4EZg2V4fLpT2&export=download&confirm=t"


def surface():
    # Model dimensions and poses use millimetres. Sample each triangle proportional to area.
    path = CACHE/"cad_models.zip"
    if not path.exists():
        response = requests.get(CAD_URL,timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        stl = archive.read(f"_CAD_models/{OBJECT}.stl")
    n = int.from_bytes(stl[80:84], "little")
    assert len(stl)==84+n*50
    tris = np.ndarray((n,3,3), "<f4",buffer=stl,offset=96,strides=(50,12,4)).astype(float)
    edge1,edge2 = tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]
    areas = np.linalg.norm(np.cross(edge1,edge2),axis=1)
    rng = np.random.default_rng(43163)
    idx = rng.choice(n,size=100000,p=areas/areas.sum())
    u,v = rng.random((2,len(idx)))
    flip = u+v>1
    u[flip],v[flip] = 1-u[flip],1-v[flip]
    samples = tris[idx,0]+edge1[idx]*u[:,None]+edge2[idx]*v[:,None]
    return cKDTree(samples),hashlib.sha256(data).hexdigest()


def near_object(x,y,z,k,camera,world_poses,tree):
    local = np.stack(((x-k["cx"])*z/k["fx"],(y-k["cy"])*z/k["fy"],z))
    world = camera[:3,:3]@local+camera[:3,3:4]
    near = np.zeros(len(z),bool)
    for transform in world_poses:
        transform = np.array(transform).reshape(4,4)
        points = np.linalg.solve(transform[:3,:3], world-transform[:3,3:4]).T
        inside = np.all((points>=tree.mins-1)&(points<=tree.maxes+1),axis=1)
        if inside.any():
            dist,_ = tree.query(points[inside],distance_upper_bound=1)
            near[inside] |= np.isfinite(dist)
    return near


def inspect(row,tree,world_poses):
    sensor = row["sensor"]
    first = row["raw"][0]
    k = calibration(sensor)
    a,b = [render(ref,"Ensenso",target,sensor) for ref,target in zip(row["gt"],row["raw"])]
    depth = read_depth(first,sensor)
    x,y = grid(a)
    x,y = x[a[y,x]>0],y[a[y,x]>0]
    near = near_object(x,y,a[y,x],k,pose(first),world_poses,tree)
    x,y = x[near],y[near]
    ref,obs = a[y,x],depth[y,x]
    valid = obs>0
    gross = valid & (np.abs(ref-obs)>10)
    stencil = np.ones((3,3),np.uint8)
    lo,hi = cv2.erode(a,stencil)[y,x],cv2.dilate(a,stencil)[y,x]
    smooth = (lo>0)&(hi-lo<=3)
    indices = np.flatnonzero(smooth&gross)
    example = None
    if indices.size:
        i = indices[0]
        example = dict(x=int(x[i]),y=int(y[i]),reference_mm=float(ref[i]),native_mm=float(obs[i]))
    third = read_depth(row["gt"][2],"Ensenso")
    xt,yt = grid(third)
    known = third[yt,xt]>0
    xt,yt = xt[known],yt[known]
    zt = third[yt,xt]
    kt = calibration("Ensenso")
    pt = pose(row["gt"][2])
    mask = near_object(xt,yt,zt,kt,pt,world_poses,tree)
    full = mask.copy()
    hidden = mask.copy()
    for target,gt in zip(row["raw"],(a,b)):
        coords = project(xt,yt,zt,pt,pose(target),kt,k)
        eligible,occluded = conservative_occlusion(gt,*coords)
        full &= eligible
        hidden &= occluded
    return dict(sensor=sensor,scene=row["scene"],initial=row["ids"][0],cad_points=len(x),
                valid=int(valid.sum()),gross=int(gross.sum()),smooth_valid=int((valid&smooth).sum()),
                smooth_gross=int((smooth&gross).sum()), persistent_eligible=int(full.sum()),
                persistent_hidden=int(hidden.sum()),example=example)


if __name__=="__main__":
    tree,sha = surface()
    rows = []
    pose_hashes = {}
    for row in selection():
        dirname = row["raw"][0].split("/"+row["sensor"]+"/")[0]
        path = cached(dirname+"/GT_world2obj.json")
        pose_hashes[dirname] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(inspect(row,tree,json.loads(path.read_text()).values()))
    output_file("surface.json").write_text(json.dumps(dict(cad_url=CAD_URL,sha256=sha,pose_sha256=pose_hashes,
                                                          rows=rows),indent=2),encoding="utf8")
