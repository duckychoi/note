"""Procedural pinhole rendering and common inputs for the depth-transfer gate."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from data import CACHE, HERE, load_image

sys.path.insert(0,str(HERE.parents[2]/"research-sources/ml-depth-pro-2/src"))
FAMILIES = ("diffuse","silhouette","shadow","highlight")


def categories(image):
    luma = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(float)/255
    gradient = np.hypot(cv2.Sobel(luma,-1,1,0),cv2.Sobel(luma,-1,0,1))/4
    labels = np.zeros(luma.shape,np.uint8)
    labels[gradient>.10] = 1
    labels[luma<.22] = 2
    labels[luma>.78] = 3
    return labels


def render(seed,index,phase,family):
    rng = np.random.default_rng([seed,index,phase])
    h,w = 192,256
    fx = rng.uniform(.72,1.32)*w
    y,x = np.indices((h,w),dtype=np.float32)
    a,b = (x-(w-1)/2)/fx,(y-(h-1)/2)/fx
    wall = rng.uniform(3.4,7.)
    z = wall/(1+.16*a+.04*b)
    color = np.empty((h,w,3),np.float32)
    base = rng.uniform(.25,.85,3)
    walltex = 1+.12*np.sin(8*a*z+1.3)*np.cos(7*b*z)
    color[:] = base*walltex[:,:,None]
    floor = rng.uniform(.65,1.35)/np.maximum(b,.0001)
    is_floor = floor<z
    z = np.minimum(z,floor)
    checker = (np.floor(a*z*2)+np.floor(z*2))%2
    color[is_floor] = ((.32+.17*checker[...,None])*rng.uniform(.6,1.3,3))[is_floor]
    light = rng.normal(size=3)
    light[2] = -abs(light[2])-.6
    light /= np.linalg.norm(light)
    metallic = family==3
    # The same random object draws are consumed for every family.
    for n in range(9):
        cz = rng.uniform(.75,wall*.9)
        cx,cy = rng.uniform(-.51,.51)*cz*w/fx,rng.uniform(-.46,.46)*cz*h/fx
        rad = rng.uniform(.055,.20,3)*cz
        rectangular = rng.random()<.35
        albedo = rng.uniform(.10,.94,3)
        phase_tex = rng.uniform(0,6)
        if family==1 and n%2==0:
            rad[0]*=.3
            rad[1]*=1.7
        center=np.array((cx,cy,cz))
        if rectangular:
            dx = np.where(np.abs(a)<1e-6,1e-6,a)
            dy = np.where(np.abs(b)<1e-6,1e-6,b)
            x0,x1 = (cx-rad[0])/dx,(cx+rad[0])/dx
            y0,y1 = (cy-rad[1])/dy,(cy+rad[1])/dy
            entry = np.maximum(np.maximum(np.minimum(x0,x1),np.minimum(y0,y1)),cz-rad[2])
            leave = np.minimum(np.minimum(np.maximum(x0,x1),np.maximum(y0,y1)),cz+rad[2])
            hit = (entry<=leave)&(entry>0)&(entry<z)
            depth = entry[hit]
        else:
            u,v,t = 1/rad**2
            aa = a*a*u+b*b*v+t
            ab = a*cx*u+b*cy*v+cz*t
            ac = cx*cx*u+cy*cy*v+cz*cz*t-1
            disc = ab*ab-aa*ac
            entry = (ab-np.sqrt(np.maximum(0,disc)))/aa
            hit = (disc>=0)&(entry>0)&(entry<z)
            depth = entry[hit]
        if not hit.any():
            continue
        xyz = np.column_stack((a[hit]*depth,b[hit]*depth,depth))
        normal=(xyz-center)/rad**2
        if rectangular:
            surfaces=(xyz-center)/rad
            face=np.argmax(np.abs(surfaces),axis=1)
            normal=np.zeros_like(surfaces)
            normal[np.arange(len(face)),face]=np.sign(surfaces[np.arange(len(face)),face])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
        lambert = np.maximum(normal@light,0)
        view = -xyz/np.linalg.norm(xyz,axis=1,keepdims=True)
        halfway = view+light
        halfway /= np.linalg.norm(halfway,axis=1,keepdims=True)
        shine = np.clip((normal*halfway).sum(1),0,1)**(20 if metallic else 12)
        tex = 1+.12*np.sin(xyz[:,0]*32+phase_tex)*np.cos(xyz[:,1]*24)
        tint = albedo if not metallic else .35*albedo+.42
        color[hit] = tint*(tex*(.30+.68*lambert))[:,None]+(1.5 if metallic else .09)*shine[:,None]
        z[hit]=depth
    if family==2:
        color *= (.14+.46/(1+np.exp(-(a*8+b*5))))[:,:,None]
    image=np.uint8(np.clip(color+rng.normal(0,.012,color.shape),0,1)*255)
    return image,z.astype(np.float32),float(fx)


@lru_cache(maxsize=1)
def foundation():
    from depth_pro_2 import PRESETS,create_model_and_transforms
    return create_model_and_transforms(replace(PRESETS["depthpro2-h"],fov_checkpoint_uri=None),device="cuda")


def base_depth(key,image,fx):
    directory=CACHE/"foundation"
    directory.mkdir(exist_ok=True,parents=True)
    path=directory/(key+".npy")
    if not path.exists():
        model,transform=foundation()
        with torch.inference_mode():
            z=model.infer(transform(image),f_px=fx)["depth"].float().cpu().numpy()
        z=np.clip(np.nan_to_num(z,nan=8,posinf=10,neginf=.1),.1,12)
        np.save(path,z.astype(np.float32))
    return np.load(path)


def real_frame(row,with_depth=True):
    original=load_image(row["image"])
    h,w=original.shape[:2]
    ratio=320/max(h,w)
    size=(round(w*ratio),round(h*ratio))
    im=cv2.resize(original,size,interpolation=cv2.INTER_AREA)
    if row["kind"]=="optical":
        # Canon EOS R6 II full-frame. Long sensor side is 36 mm; handle portrait rotation.
        fx=row["lens_mm"]*max(size)/36
    else:
        fx=row["fx"]*size[0]/w
    gt=None
    if with_depth and row["kind"]=="eth":
        gt=cv2.resize(load_image(row["depth"],cv2.IMREAD_UNCHANGED),size,interpolation=cv2.INTER_NEAREST).astype(np.float32)/5000
    return im,gt,fx
