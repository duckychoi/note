"""Render fixed-camera Hypersim surfaces with geometric visibility and dielectric direct lighting."""
import json
import time

import numpy as np
import open3d as o3d
from PIL import Image
import psutil

from depth_full_data import h5_array, hypersim_intrinsics, poses
from hypersim_fetch import frame_assets
from material_light_assets import OUT, digest, save_json


def unit(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-9)


def dot(a,b):
    return np.sum(a*b,axis=-1)


def points(row,i):
    _,path = frame_assets(row["scene"],row["cam"],i)[1]
    dist = h5_array(path).astype(np.float32)
    h,w = dist.shape
    k = hypersim_intrinsics(row["scene"],w,h)
    y,x = np.indices((h,w),dtype=np.float32)
    rays = np.stack((x,y,np.ones_like(x)),-1)@np.linalg.inv(k).astype(np.float32).T
    return unit(rays)*dist[...,None]


def triangles(p,stride):
    a = p[::stride,::stride]
    rr,cc = a.shape[:2]
    ids = np.arange(rr*cc,dtype=np.int32).reshape(rr,cc)
    def tri(i,j,k):
        return np.stack((i.ravel(),j.ravel(),k.ravel()),axis=-1)
    f = np.concatenate((tri(ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1]),
                        tri(ids[1:,1:],ids[1:,:-1],ids[:-1,1:])))
    verts = a.reshape(-1,3).copy()
    zz = verts[f,2]
    valid = np.isfinite(zz).all(1)&(zz.min(1)>.05)&(zz.max(1)-zz.min(1)<.015+.035*zz.min(1))
    return np.nan_to_num(verts,nan=0.,posinf=0.,neginf=0.), f[valid]


def visibility_scene(row,p):
    scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    trajectory = poses(row["scene"],row["cam"])
    ref_rot,ref_center = trajectory[row["frame"]]
    count = 0
    used = []
    for i in row["neighbors"]:
        if i not in trajectory:
            continue
        same = i==row["frame"]
        verts,faces = triangles(p if same else points(row,i),2 if same else 4)
        rot,center = trajectory[i]
        coords = verts@(ref_rot.T@rot).T.astype(np.float32) + ((center-ref_center)@ref_rot).astype(np.float32)
        if len(faces):
            scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(coords)),o3d.core.Tensor(faces.astype(np.uint32)))
            count += len(faces)
            used.append(i)
    if len(used)<2:
        raise ValueError(f"Insufficient support views for {row}")
    return scene,count,used


def cast(scene,p,lamp,usable):
    result = np.zeros(p.shape[:2],bool)
    pts = p[usable]
    shadows = []
    for chunk in np.array_split(pts,max(1,int(np.ceil(len(pts)/100000)))):
        dist = np.linalg.norm(lamp-chunk,axis=1)
        direction = unit(lamp-chunk)
        # Move toward the light before testing to avoid intersecting the same surface.
        eps = np.maximum(.004*chunk[:,2],.002)
        rays = np.ascontiguousarray(np.column_stack((chunk+eps[:,None]*direction,direction)),dtype=np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays),nthreads=3)["t_hit"].numpy()
        shadows.append(hits<dist-eps)
    result[usable] = np.concatenate(shadows)
    return result


def render(row):
    folder = OUT/"renders"/row["scene"]
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"render.json").is_file():
        return
    while psutil.virtual_memory().available<8*2**30:
        time.sleep(4)
    assets = OUT/"assets"/row["scene"]
    albedo = h5_array(assets/"diffuse_reflectance.hdf5").astype(np.float32)
    normal = h5_array(assets/"normal_cam.hdf5").astype(np.float32)*np.array([1,-1,-1],np.float32)
    entity = h5_array(assets/"render_entity_id.hdf5")
    p = points(row,row["frame"])
    valid = (np.isfinite(p).all(2)&np.isfinite(albedo).all(2)&np.isfinite(normal).all(2)&
             (np.linalg.norm(normal,axis=-1)>.5)&(p[:,:,2]>.05)&(entity>=0))
    p = np.nan_to_num(p,nan=0.,posinf=0.,neginf=0.)
    normal = unit(np.nan_to_num(normal))
    albedo = np.clip(np.nan_to_num(albedo),0,1)
    view = -unit(p)
    normal = normal*np.where(dot(view,normal)>=0,1.,-1.).astype(np.float32)[...,None]
    nv = np.clip(dot(normal,view),.001,1)
    scale = float(np.median(p[:,:,2][valid]))
    scene,count,used = visibility_scene(row,p)
    lamps = []
    masks = {}
    for i,angle in enumerate((-150,-30,90)):
        rad = np.deg2rad(angle)
        lamp = scale*np.array([.34*np.cos(rad),.34*np.sin(rad),.10],dtype=np.float32)
        lamps.append(lamp.tolist())
        diff = lamp-p
        light = unit(diff)
        nl = np.clip(dot(normal,light),0,1)
        blocked = cast(scene,p,lamp,valid&(nl>0))
        masks[f"shadow{i}"] = blocked
        direct = ((~blocked)*nl*(2.4*scale**2)/np.maximum(dot(diff,diff),(.10*scale)**2)).astype(np.float32)
        halfway = unit(light+view)
        nh = np.clip(dot(normal,halfway),0,1)
        vh = np.clip(dot(view,halfway),0,1)
        # Isotropic GGX NDF and separable Smith masking, dielectric F0=0.04.
        alpha2 = .18**2
        distribution = alpha2/(np.pi*((alpha2-1)*nh**2+1)**2)
        def smith(c):
            return 2*c/(c+np.sqrt(alpha2+(1-alpha2)*c*c))
        fresnel = .04+.96*(1-vh)**5
        spec = distribution*smith(nv)*smith(nl)*fresnel/(4*nv*np.maximum(nl,1e-6))
        ambient = .12*albedo
        matte = albedo*direct[...,None]/np.pi
        mirror = direct*spec
        coat = matte*(1-fresnel[...,None])+mirror[...,None]
        masks[f"glint{i}"] = valid&(mirror>.2*(ambient+coat).mean(2))&(mirror>.015)
        for mat,component in (("matte",matte),("coat",coat)):
            radiance = ambient+component
            rgb = np.rint(255*(radiance/(1+radiance))**(1/2.2)).clip(0,255).astype(np.uint8)
            rgb[~valid]=[100,100,100]
            Image.fromarray(rgb).save(folder/f"{mat}-{i}.png",compress_level=2)
    np.savez_compressed(folder/"passes.npz",z=p[:,:,2],valid=valid,entity=entity,**masks)
    save_json(folder/"render.json",dict(scene=row["scene"],cam=row["cam"],frame=row["frame"],triangles=count,
              support_frames=used,lamps=lamps,median_depth=scale,valid_fraction=float(valid.mean()),
              z_sha256=__import__("hashlib").sha256(p[:,:,2].tobytes()).hexdigest(),
              assets={a.name:digest(a) for a in assets.glob("*.hdf5")}))
    print("render",row["scene"],count,flush=True)


if __name__=="__main__":
    for row in json.loads((OUT/"scenes.json").read_text()):
        render(row)
