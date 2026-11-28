"""Cache synthetic native-resolution depth features and paired stereo observations."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import psutil
import torch
import torch.nn.functional as F

from depth_full_data import camera_ray_lengths,h5_array,hypersim_intrinsics,poses
from encoder_expand_data import selection_for
from hypersim_fetch import frame_assets
from scale_surface_rgb import CKPT,OUT as PREV,SRC,nearest,regions,resize_rgb,sha,write
from stereo_controlled_observations import rectification,stereo_pair

OUT=PREV.parent/"native-surface-direction"
PAIRS=PREV.parent/"control43-geometry-pairs-v1/pairs-full.json"
SEED=261126

def read_rgb(scene,cam,frame):
    (_,path),_=frame_assets(scene,cam,frame)
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


class Native:
    def __init__(self):
        os.environ["TORCH_HOME"]=str(SRC/"ml-depth-pro-2-cache")
        sys.path.insert(0,str(SRC/"ml-depth-pro-2/src"))
        from depth_pro_2 import create_model_and_transforms
        from depth_pro_2.network.encoder import autocast_for
        self.cast=autocast_for
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(3)
        cv2.setNumThreads(1)
        self.model,self.transform=create_model_and_transforms("depthpro2-s",device=self.device)
        self.model.load_state_dict(torch.load(CKPT,weights_only=True,map_location="cpu"),strict=True)
        self.model.eval()
        dim=self.model.encoder.blocks[-1].norm1.normalized_shape[0]
        proj=np.linalg.qr(np.random.default_rng(SEED).normal(size=(dim,16)))[0]
        self.proj=torch.from_numpy(proj.astype(np.float32)).to(self.device)
        self.check=None

    @torch.inference_mode()
    def extract(self,rgb):
        while psutil.virtual_memory().available < 9*2**30 or (
                self.device.type=="cuda" and torch.cuda.mem_get_info()[0]<5*2**30):
            time.sleep(5)
        h,w=rgb.shape[:2]
        small=resize_rgb(rgb)
        sh,sw=small.shape[:2]
        t=self.transform(rgb).unsqueeze(0).to(self.device)
        padded=F.pad(t,(0,-w%32,0,-h%32),mode="replicate")
        with self.cast(self.device,self.model.precision)():
            maps,grid=self.model.encoder.encode(padded)
        q=self.model.decode_frame(maps,grid)[0,0,:h,:w].float()
        focal=self.model.fov.focal_length_px(t)
        if self.check is None:
            ref=self.model.infer(t[0])
            self.check=float((ref["inverse_depth"]-q).abs().max())
            if self.check>1e-5 or not torch.allclose(ref["focallength_px"],focal):
                raise ValueError("Native path disagrees with official single-image inference")
        base=(focal/(w*q.clamp_min(1e-5))).cpu().numpy()
        labels=regions(small)
        n=labels.max()+1
        count=np.bincount(labels.ravel(),minlength=n)
        def means(v):
            return np.bincount(labels.ravel(),weights=v.ravel(),minlength=n)/np.maximum(count,1)
        zsmall=nearest(base,(sh,sw))
        z=np.clip(np.log(zsmall),-3,6)
        y,x=np.indices((sh,sw),dtype=np.float32)
        fields=[x/(sw-1),y/(sh-1),z,z*z,*(small[:,:,i]/255 for i in range(3))]
        # Native ViT tokens are on a regular 16-pixel grid; sample their field at
        # the actual RGB coordinates, including the padding offset at the far edges.
        tx=((x+.5)*w/sw)*grid[1]/padded.shape[-1]-.5
        ty=((y+.5)*h/sh)*grid[0]/padded.shape[-2]-.5
        columns=[means(v) for v in fields]
        for idx in (1,-1):
            tokens=(maps[idx][0].float()@self.proj).reshape(*grid,-1).cpu().numpy()
            for c in range(tokens.shape[-1]):
                field=cv2.remap(tokens[:,:,c],tx,ty,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                columns.append(means(field))
        columns.extend([np.full(n,z.mean()),np.full(n,z.std()),np.full(n,np.log(float(focal)/w)),
                        np.log1p(count),means(abs(z-cv2.blur(z,(9,9))))])
        features=np.column_stack(columns).astype(np.float32)
        return base,labels,features


def manifest():
    groups=defaultdict(list)
    for s,c,f,n in json.loads(PAIRS.read_text()):
        groups[s].append((c,f,n))
    rows=[]
    for idx,(s,available) in enumerate(sorted(groups.items())):
        which="cal" if idx%5==0 else "fit"
        for j in np.linspace(0,len(available)-1,min(4,len(available)),dtype=int):
            rows.append((which,s,*available[j]))
    for s,available in sorted(selection_for("validation_24").items()):
        for j in np.linspace(0,len(available)-1,min(4,len(available)),dtype=int):
            c,f=available[j]
            # Validation selects RGB amplitudes with metric GT; no teacher used.
            rows.append(("val",s,c,f,f))
    return rows


def stereo(scene,cam,frame,neighbor,rgb,K):
    image=resize_rgb(rgb)
    sh,sw=image.shape[:2]
    dummy=np.zeros((sh,sw),np.uint8)
    blank=np.full(dummy[::2,::2].shape,np.nan,np.float32)
    trajectory=poses(scene,cam)
    if frame==neighbor or frame not in trajectory or neighbor not in trajectory:
        return blank,blank,blank
    other=resize_rgb(read_rgb(scene,cam,neighbor))
    r0,c0=trajectory[frame]
    r1,c1=trajectory[neighbor]
    kk=K.copy()
    scale=np.array([sw/rgb.shape[1],sh/rgb.shape[0]])
    kk[:2]*=scale[:,None]
    kk[:2,2]+=(scale-1)/2
    geom=rectification([(image,dummy,None,kk),(other,dummy,None,kk)],
                       np.column_stack((r1.T@r0,r1.T@(c0-c1))))
    if geom is None:
        return blank,blank,blank
    y,x=np.indices((sh,sw))
    factor=.55+.65*np.exp(-((x/sw-.32)**2+(y/sh-.56)**2)/.12)
    altered=np.clip(other.astype(np.float32)*factor[:,:,None],0,255).astype(np.uint8)
    A=np.asarray(abs(geom[1])/np.maximum(geom[2][0][-1],1e-8),np.float32)
    values=[]
    for target in (other,altered):
        z,ok,_=stereo_pair([image,target],geom)[0]
        values.append(np.where(ok,A/np.maximum(z,.001),np.nan).astype(np.float32))
    return A,*values


def cache_path(folder,row):
    split,s,c,f,n=row
    return folder/f"{split}-{s}-{c}-{f:04}-{n:04}.npz"


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    code=Path(__file__)
    deps=[code,*[code.with_name(n) for n in ("scale_surface_rgb.py","stereo_controlled_observations.py","depth_full_data.py")]]
    rows=manifest()
    digest=hashlib.sha256((sha(CKPT)+json.dumps(rows)+"".join(sha(p) for p in deps)).encode()).hexdigest()
    folder=OUT/("synthetic-"+digest[:16])
    folder.mkdir(exist_ok=True)
    reader=Native()
    shapes=set()
    for i,row in enumerate(rows):
        path=cache_path(folder,row)
        if not path.exists():
            split,s,c,f,n=row
            rgb=read_rgb(s,c,f)
            h,w=rgb.shape[:2]
            K=hypersim_intrinsics(s,w,h)
            _,(_,depth)=frame_assets(s,c,f)
            truth=h5_array(depth)/camera_ray_lengths((h,w),tuple(K.ravel()))
            base,lab,features=reader.extract(rgb)
            A,d1,d2=stereo(s,c,f,n,rgb,K) if split!="val" else (np.float32(0),)*3
            tmp=path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,base=nearest(base,lab.shape)[::2,::2],
                z=nearest(truth.astype(np.float32),lab.shape)[::2,::2],
                lab=lab[::2,::2].astype(np.int16),X=features,
                area=np.bincount(nearest(lab.astype(np.float32),(h,w)).ravel().astype(int),
                                 minlength=len(features)),A=A,d1=d1,d2=d2,shape=[h,w],
                official_check=reader.check)
            tmp.replace(path)
        if i%100==0:
            print(i+1,len(rows),row[:2],flush=True)
    with np.load(cache_path(folder,rows[0])) as record:
        check=float(record["official_check"])
    write(OUT/"inventory.json",dict(folder=folder.name,rows=rows,signature=digest,
          ckpt=sha(CKPT),device=str(reader.device),official_check=check))


if __name__=="__main__":
    prepare()
