"""Train-only, fixed-grid tracking with the official TAPNext++ checkpoint."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil
import torch

from depth_full_data import hypersim_intrinsics,poses
from native_direction_data import OUT,cache_path,read_rgb,resize_rgb,sha,write
from stereo_controlled_observations import rectified_cameras

TAP=OUT/"tap-probe"
DIGEST="cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"

def observe(model,row,expected_A):
    _,scene,cam,frame,neighbor=row
    raw=read_rgb(scene,cam,frame)
    im=resize_rgb(raw)
    h,w=im.shape[:2]
    y,x=np.mgrid[8:h-1:16,8:w-1:16]
    points=np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)
    iy,ix=points.astype(int).T[::-1]//2
    blank=np.full(len(points),np.nan,np.float32)
    trajectory=poses(scene,cam)
    if frame==neighbor or frame not in trajectory or neighbor not in trajectory:
        return iy,ix,blank,blank
    intr=hypersim_intrinsics(scene,raw.shape[1],raw.shape[0]).copy()
    scale=np.array([w/raw.shape[1],h/raw.shape[0]])
    intr[:2]*=scale[:,None];intr[:2,2]+=(scale-1)/2
    r0,c0=trajectory[frame];r1,c1=trajectory[neighbor]
    geom,_=rectified_cameras([intr,intr],(w,h),np.column_stack([r1.T@r0,r1.T@(c0-c1)]))
    if geom is None:
        return iy,ix,blank,blank
    r0,r1,p0,p1,axis=geom
    rectify=lambda pts,r,p:cv2.undistortPoints(pts.reshape(-1,1,2),intr,None,R=r,P=p[:,:3]).reshape(-1,2)
    origin=rectify(points,r0,p0)
    rays=np.column_stack((points,np.ones(len(points))))@np.linalg.inv(intr).T
    zscale=(rays@r0.T)[:,2]
    A=abs(p1[axis,3])/np.maximum(zscale,1e-8)
    if not np.allclose(A,expected_A[iy,ix],rtol=3e-5):
        raise ValueError("Track/metric geometry disagree")
    other=resize_rgb(read_rgb(scene,cam,neighbor))
    gy,gx=np.indices((h,w))
    factor=.55+.65*np.exp(-((gx/w-.32)**2+(gy/h-.56)**2)/.12)
    changed=np.clip(other.astype(float)*factor[:,:,None],0,255).astype(np.uint8)
    disparities=[]
    for target in (other,changed):
        _,_,state=model.track_frame(np.ascontiguousarray(im[:,:,::-1]),points,autocast=False)
        pts,vis,_=model.track_frame(np.ascontiguousarray(target[:,:,::-1]),state=state,autocast=False)
        dest=rectify(pts,r1,p1)
        disparity=(origin[:,axis]-dest[:,axis])*(-np.sign(p1[axis,3]))
        ok=vis & (zscale>0)&(disparity>.25)&np.isfinite(dest).all(axis=1)
        ok &= (pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
        ok &= abs(origin[:,1-axis]-dest[:,1-axis])<=2
        disparities.append(np.where(ok,disparity,np.nan).astype(np.float32))
    return iy,ix,*disparities

def main():
    inv=json.loads((OUT/"inventory.json").read_text())
    checkpoint=TAP/"tapnextpp_ckpt.pt"
    if sha(checkpoint)!=DIGEST:
        raise ValueError("Unexpected TAPNext++ checkpoint")
    sys.path[:0]=[str(TAP/"deps"),str(TAP/"tapnet-source")]
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=TAPNextPP.from_checkpoint(checkpoint,device=device)
    signature=hashlib.sha256((inv["signature"]+DIGEST+sha(Path(__file__))).encode()).hexdigest()
    folder=OUT/("tap-tracks-"+signature[:16])
    folder.mkdir(exist_ok=True)
    for i,row in enumerate(inv["rows"]):
        if row[0]=="val":
            continue
        path=cache_path(folder,row)
        if not path.exists():
            while psutil.virtual_memory().available<9*2**30 or (device=="cuda" and torch.cuda.mem_get_info()[0]<4*2**30):
                time.sleep(5)
            with np.load(cache_path(OUT/inv["folder"],row)) as rec:
                iy,ix,d1,d2=observe(model,row,rec["A"])
            tmp=path.with_suffix(".part.npz")
            np.savez_compressed(tmp,iy=iy,ix=ix,tap1=d1,tap2=d2)
            tmp.replace(path)
        if i%200==0:
            print(i,row[1],flush=True)
    write(OUT/"tap-inventory.json",dict(folder=folder.name,signature=signature,checkpoint=DIGEST))

if __name__=="__main__":
    main()
