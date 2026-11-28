"""RGB-only inference and synthetic-only fitting of a scale and a centered region readout."""
import argparse
from collections import defaultdict
import hashlib
import inspect
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

from depth_full_data import camera_ray_lengths, h5_array, hypersim_intrinsics, poses
from encoder_expand_data import selection_for
from hypersim_fetch import frame_assets
from stereo_controlled_observations import rectification, stereo_pair

SRC=Path(__file__).resolve().parents[3]/"research-sources"
OUT=SRC/"encoder42_eval/scale-surface-rgb"
CKPT=SRC/"encoder42_eval/control43-geometry-pairs-v1/full-students-4244/d2/best.pt"
WIDTH=256
RIDGE=.02
CAP=.25
ARMS=("base","scale","surface","dense_joint","stereo_joint")


def write(path, payload):
    tmp=path.with_suffix(".partial")
    tmp.write_text(json.dumps(payload,indent=1,allow_nan=False),encoding="utf8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def resize_rgb(rgb):
    im=rgb if isinstance(rgb,Image.Image) else Image.fromarray(rgb)
    return np.array(im.convert("RGB").resize((WIDTH,max(2,round(WIDTH*im.height/im.width))),Image.Resampling.BILINEAR))


def nearest(values,shape):
    return cv2.resize(values,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST_EXACT)


def regions(rgb):
    h,w=rgb.shape[:2]
    if min(h,w)<16:
        return np.zeros((h,w),dtype=np.int32)
    seeds=np.zeros((h,w),np.int32)
    idx=0
    for y in range(12,h,28):
        for x in range(12,w,28):
            idx+=1
            seeds[y-1:y+2,x-1:x+2]=idx
    labels=cv2.watershed(rgb,seeds)
    labels=np.maximum(labels,0).astype(np.uint16)
    while (labels==0).any():
        labels=np.where(labels==0,cv2.dilate(labels,np.ones((3,3),np.uint8)),labels)
    return labels.astype(np.int32)-1


def group_mean(values,labels,n):
    return np.bincount(labels.ravel(),weights=np.asarray(values).ravel(),minlength=n)


class Reader:
    def __init__(self):
        os.environ["TORCH_HOME"]=str(SRC/"ml-depth-pro-2-cache")
        sys.path.insert(0,str(SRC/"ml-depth-pro-2/src"))
        from depth_pro_2 import create_model_and_transforms
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        self.model,self.transform=create_model_and_transforms("depthpro2-s",device=torch.device("cpu"))
        self.model.load_state_dict(torch.load(CKPT,map_location="cpu",weights_only=True),strict=True)
        self.model.eval()
        dim=self.model.encoder.blocks[-1].norm1.normalized_shape[0]
        axes=np.linalg.qr(np.random.default_rng(421126).normal(size=(dim,16)))[0]
        self.proj=torch.from_numpy(axes.astype(np.float32))
        self.official_check=None

    @torch.inference_mode()
    def extract(self,rgb):
        while psutil.virtual_memory().available<9*2**30:
            time.sleep(10)
        h,w=rgb.shape[:2]
        tensor=self.transform(rgb).unsqueeze(0)
        padded=F.pad(tensor,(0,-w%32,0,-h%32),mode="replicate")
        maps,grid=self.model.encoder.encode(padded)
        q=self.model.decode_frame(maps,grid)[0,0,:h,:w]
        f=self.model.fov.focal_length_px(tensor)
        if self.official_check is None:
            check=self.model.infer(tensor[0])
            self.official_check=float((check["inverse_depth"]-q).abs().max())
            if self.official_check>1e-5 or not torch.allclose(check["focallength_px"],f):
                raise ValueError("Single-image path differs from model.infer")
        base=(f/(w*q.float().clamp_min(1e-5))).numpy().astype(float)
        if not np.isfinite(base).all() or (base<=0).any():
            raise ValueError("Invalid base depth")
        tokens=(maps[-1][0].float()@self.proj).reshape(*grid,-1).numpy()
        labels=regions(rgb)
        n=int(labels.max()+1)
        count=np.bincount(labels.ravel(),minlength=n)
        y,x=np.indices(labels.shape)
        logz=np.clip(np.log(base),-3,5)
        cols=[logz,logz**2,x/(w-1),y/(h-1),*(rgb[:,:,i]/255. for i in range(3))]
        local=[group_mean(c,labels,n)/np.maximum(count,1) for c in cols]
        for i in range(tokens.shape[-1]):
            c=cv2.resize(tokens[:,:,i],(padded.shape[-1],padded.shape[-2]),interpolation=cv2.INTER_LINEAR)[:h,:w]
            local.append(group_mean(c,labels,n)/np.maximum(count,1))
        local=np.stack(local,axis=1)
        glob=np.r_[1.,np.log(float(f)/w),np.quantile(logz,[.1,.25,.5,.75,.9]),
                   np.mean(rgb,axis=(0,1))/255,tokens.mean(axis=(0,1))]
        local-=np.average(local,weights=count,axis=0)
        return base,labels,glob,local


def read_syn(scene,camera,frame):
    (_,image),(_,depth)=frame_assets(scene,camera,frame)
    with Image.open(image) as im:
        w,h=im.size
        rgb=resize_rgb(im)
    K=hypersim_intrinsics(scene,w,h)
    ranges=h5_array(depth)
    # Convert radial distances on their original rays before nearest label resizing.
    z=ranges/camera_ray_lengths((h,w),tuple(K.ravel()))
    z=nearest(z.astype(np.float32),rgb.shape)
    K=K.copy()
    K[0,:]*=rgb.shape[1]/w
    K[1,:]*=rgb.shape[0]/h
    K[0,2]+=(rgb.shape[1]/w-1)/2
    K[1,2]+=(rgb.shape[0]/h-1)/2
    return rgb,z,K


def stereo_measure(rgb,K,scene,cam,frame,neighbor):
    if frame==neighbor:
        return None
    trajectory=poses(scene,cam)
    if frame not in trajectory or neighbor not in trajectory:
        return None
    (_,image),_=frame_assets(scene,cam,neighbor)
    with Image.open(image) as im:
        other=resize_rgb(im)
    r0,c0=trajectory[frame]
    r1,c1=trajectory[neighbor]
    rt=np.column_stack((r1.T@r0,r1.T@(c0-c1)))
    size_only=np.empty(rgb.shape[:2],np.uint8)
    pair=[(im,size_only,None,K) for im in (rgb,other)]
    geom=rectification(pair,rt)
    if geom is None:
        return None
    z,valid,_=stereo_pair([rgb,other],geom)[0]
    return z,valid


def centered_stats(X,labels,r,mask):
    n=len(X)
    weights=np.bincount(labels[mask],minlength=n)
    n_valid=int(weights.sum())
    if n_valid==0:
        return np.zeros((X.shape[1],X.shape[1]+1)),0
    means=group_mean(r[mask],labels[mask],n)/np.maximum(weights,1)
    XC=X-np.average(X,weights=weights,axis=0)
    return (XC.T*(weights/n_valid))@np.column_stack([XC,means]),int((weights>0).sum())


def moments(rep,z,stereo):
    base,labels,g,X=rep
    mask=np.isfinite(z)&(z>=.25)&(z<=30)
    if not mask.any():
        return None
    r=np.log(np.where(mask,z,1)/base)
    G=np.outer(g,np.r_[g,r[mask].mean()])
    S,_=centered_stats(X,labels,r,mask)
    T=np.zeros_like(S)
    used=0
    if stereo is not None:
        zz,ok=stereo
        lab=labels[::2,::2]
        res=np.log(np.maximum(zz,.001)/base[::2,::2])
        T,surfaces=centered_stats(X,lab,res,ok)
        if surfaces>=2:
            used=int(ok.sum())
            T*=used/ok.size
        else:
            T*=0
    return G,S,T,np.array([used,labels[::2,::2].size,int(used>0),1])


def fit_linear(stat,intercept=False):
    A,rhs=stat[:,:-1],stat[:,-1]
    variance=np.maximum(1e-10,np.diag(A).copy())
    if intercept:
        variance=np.maximum(1e-10,variance-A[:,0]**2/A[0,0])
        variance[0]=0.
    return np.linalg.solve(A+RIDGE*np.diag(variance),rhs)


def infer_all(rep,heads):
    base,lab,g,X=rep
    scale=np.clip(g@heads["global"],-.5,.5)
    def centered(coeff):
        h=np.clip(X@coeff,-.5,.5)
        h=h[lab]
        h=h-h.mean()
        return h/max(1.,2*float(np.max(np.abs(h))))
    loc=centered(heads["surface"])
    dense=centered(heads["dense"])
    fields=dict(base=0.,scale=scale,surface=loc,dense_joint=scale+dense,stereo_joint=scale+loc)
    return {k:base+np.clip(base*np.expm1(v),-CAP,CAP) for k,v in fields.items()}


def read_heads():
    weights=json.loads((OUT/"weights.json").read_text())
    if weights["checkpoint_sha256"]!=sha(CKPT):
        raise ValueError("Depth Pro checkpoint does not match readout training")
    return {"global":np.array(weights["global_head"]),"surface":np.array(weights["surface"]),
            "dense":np.array(weights["dense"])}


def predict_rgb(reader, image, heads):
    """One RGB photo in, metric camera-axis depth at the evaluation resolution out."""
    return infer_all(reader.extract(resize_rgb(image)),heads)["stereo_joint"]


def fit_signature():
    functions=(resize_rgb,nearest,regions,group_mean,Reader,read_syn,stereo_measure,
               centered_stats,moments,rectification,stereo_pair)
    sources=[inspect.getsource(func) for func in functions]
    deps=("depth_full_data.py","encoder_expand_data.py","hypersim_fetch.py","stereo_controlled_observations.py")
    sources.extend(sha(Path(__file__).with_name(name)) for name in deps)
    sources.extend([str(WIDTH),sha(CKPT)])
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


def train():
    OUT.mkdir(exist_ok=True)
    signature=fit_signature()
    folder=OUT/("fit-"+signature[:16])
    folder.mkdir(exist_ok=True)
    model=Reader()
    pairs=json.loads((OUT.parent/"control43-geometry-pairs-v1/pairs-full.json").read_text())
    groups=defaultdict(list)
    for s,c,f,k in pairs:
        groups[s].append((c,f,k))
    manifest={}
    sums=[]
    omitted=[]
    checks=[]
    counts=np.zeros(4,dtype=np.int64)
    for i,(scene,rows) in enumerate(sorted(groups.items())):
        chosen=[rows[j] for j in np.linspace(0,len(rows)-1,min(8,len(rows)),dtype=int)]
        manifest[scene]=chosen
        chosen_sha=hashlib.sha256(json.dumps(chosen).encode()).hexdigest()
        target=folder/(scene+"-"+chosen_sha[:16]+".npz")
        if not target.exists():
            measured=[]
            for c,f,n in chosen:
                rgb,z,K=read_syn(scene,c,f)
                rep=model.extract(rgb)
                one=moments(rep,z,stereo_measure(rgb,K,scene,c,f,n))
                if one is not None:
                    measured.append(one)
            tmp=target.with_suffix(".tmp.npz")
            if measured:
                G,S,T,N=zip(*measured)
                np.savez(tmp,G=np.mean(G,axis=0),S=np.mean(S,axis=0),T=np.mean(T,axis=0),
                         N=np.sum(N,axis=0),official=model.official_check)
            else:
                np.savez(tmp,N=np.zeros(4,dtype=np.int64),official=model.official_check)
            tmp.replace(target)
        with np.load(target) as data:
            if data["N"][3]:
                sums.append([data[k] for k in ("G","S","T")])
            else:
                omitted.append(scene)
            counts+=data["N"]
            checks.append(float(data["official"]))
        if i%40==0:
            print(scene,i+1,flush=True)
    G,S,T=[np.mean(v,axis=0) for v in zip(*sums)]
    payload=dict(global_head=fit_linear(G,True).tolist(),dense=fit_linear(S).tolist(),
                 surface=fit_linear(S+T).tolist(),stereo_counts=counts.tolist(),
                 checkpoint_sha256=sha(CKPT),scenes=len(sums),selected_scenes=len(groups),
                 no_valid_target=omitted,training_manifest=manifest,fit_signature=signature,
                 source_sha256=sha(Path(__file__)),native_check_max=max(checks))
    write(OUT/"weights.json",payload)
    print("Fitted and frozen",len(sums),"eligible scenes",counts,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.image is None and args.output is None:
        train()
    elif args.image is not None and args.output is not None:
        with Image.open(args.image) as im:
            np.save(args.output,predict_rgb(Reader(),im,read_heads()))
    else:
        parser.error("--image and --output must be supplied together")
