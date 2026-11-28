"""Camera-aligned ETH3D candidate curves and independent source-depth evidence."""
import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval"
OUT = ROOT / "pf42-trials/crossview-b"
CAL = ("delivery_area", "relief")
FIT = ("terrains", "electro", "facade")
HELD = ("pipes", "office", "courtyard", "kicker")
WIDTH = 640
DEPTH = np.geomspace(.35, 25, 224).astype(np.float32)
PATCH = np.stack(np.meshgrid([-2,0,2],[-2,0,2]),axis=-1).reshape(9,2).astype(np.float32)


def text_rows(path):
    return [line.split() for line in path.read_text(encoding="utf8").splitlines()
            if line.strip() and not line.startswith("#")]


def fisheye(x, y, coeff):
    k1,k2,p1,p2,k3,k4,sx,sy = coeff
    r = np.hypot(x,y)
    th = np.arctan(r)
    ratio = np.divide(th,r,out=np.ones_like(r),where=r>0)
    u,v,s = x*ratio,y*ratio,th**2
    tr = 1+s*(k1+s*(k2+s*(k3+s*k4)))
    return u*tr+2*p1*u*v+p2*(s+2*u*u)+sx*s, v*tr+2*p2*u*v+p1*(s+2*v*v)+sy*s


class Camera:
    def __init__(self, path, truth, sensor, pose):
        self.path,self.truth = path,truth
        self.kind,self.ow,self.oh,p = sensor
        self.intr = np.array(p[:4],dtype=np.float64)
        self.coeff = np.array(p[4:],dtype=np.float64)
        self.width = WIDTH
        self.height = round(self.oh*WIDTH/self.ow)
        self.k = self.intr*np.array([self.width/self.ow,self.height/self.oh]*2)
        q = [float(v) for v in pose[1:5]]
        self.R = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix().astype(np.float32)
        self.t = np.array(pose[5:8],dtype=np.float32)
        self.center = -self.R.T @ self.t

    def rays(self, xy):
        fx,fy,cx,cy = self.k
        xy1 = (xy+np.array([.5-cx,.5-cy],dtype=np.float32))/np.array([fx,fy],dtype=np.float32)
        return np.concatenate([xy1,np.ones_like(xy1[...,:1])],axis=-1)

    def project(self, points):
        fx,fy,cx,cy = self.k
        z = points[...,2]
        den = np.where(z>1e-6,z,1.)
        u = (points[...,0]/den)*fx+(cx-.5)
        v = (points[...,1]/den)*fy+(cy-.5)
        inside = (z>1e-6)&(u>=1)&(v>=1)&(u<self.width-2)&(v<self.height-2)
        return np.stack([u,v],axis=-1).astype(np.float32), z, inside

    def raw_pixels(self, xy):
        ray = self.rays(xy)
        u,v = ray[...,0],ray[...,1]
        if self.kind == "THIN_PRISM_FISHEYE":
            u,v = fisheye(u,v,self.coeff)
        elif self.kind != "PINHOLE":
            raise ValueError(self.kind)
        fx,fy,cx,cy = self.intr
        return np.stack((fx*u+cx-.5,fy*v+cy-.5),axis=-1).astype(np.float32)

    def depth(self, xy):
        if self.truth is None:
            return np.full(np.asarray(xy).shape[:-1],np.nan,dtype=np.float32)
        original = labels(self.truth,self.ow,self.oh)
        raw = self.raw_pixels(xy)
        x,y = np.rint(raw).astype(int).T
        shape = np.asarray(xy).shape[:-1]
        # A hole is unknown. Permit up to a one-pixel displacement in the native LiDAR raster.
        ans = np.full(len(x),np.nan,dtype=np.float32)
        for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            xx,yy = x+dx,y+dy
            inside = (xx>=0)&(yy>=0)&(xx<self.ow)&(yy<self.oh)
            got = original[yy.clip(0,self.oh-1),xx.clip(0,self.ow-1)]
            take = np.isnan(ans)&inside&np.isfinite(got)&(got>0)
            ans[take] = got[take]
        return ans.reshape(shape)


@lru_cache(maxsize=12)
def labels(path,width,height):
    if path.suffix == ".pfm":
        raise ValueError("The low-resolution loader must convert disparity to depth")
    if path.stat().st_size != 4*width*height:
        raise ValueError(f"Unexpected ETH3D LiDAR raster size {path}")
    return np.memmap(path,dtype="<f4",mode="r",shape=(height,width))


def sensors(path):
    return {int(row[0]):(row[1],int(row[2]),int(row[3]),np.array(row[4:],dtype=float)) for row in text_rows(path)}


def scene(name, with_truth=True):
    root = OUT/"eth3d"/name
    cal = root/"dslr_calibration_jpg"
    cameras = sensors(cal/"cameras.txt")
    rows = [r for r in text_rows(cal/"images.txt") if len(r)==10]
    return [Camera(root/"images"/r[-1],root/"ground_truth_depth"/r[-1] if with_truth else None,
                   cameras[int(r[-2])],r) for r in sorted(rows,key=lambda r:r[-1])]


@lru_cache(maxsize=22)
def rgb(cam):
    reduced = cv2.IMREAD_REDUCED_COLOR_4 if cam.ow>=2000 else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(cam.path,np.uint8),reduced)
    if image is None:
        raise ValueError(cam.path)
    grid = all_xy(cam)
    pixels = cam.raw_pixels(grid+.0)
    height,width=image.shape[:2]
    pixels = ((pixels+.5)*np.array([width/cam.ow,height/cam.oh],dtype=np.float32)-.5).astype(np.float32)
    return cv2.remap(image,pixels[...,0],pixels[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[...,::-1].copy()


@lru_cache(maxsize=24)
def gray(cam):
    return cv2.cvtColor(rgb(cam),cv2.COLOR_RGB2GRAY).astype(np.float32)/255.


def all_xy(cam):
    return np.stack(np.meshgrid(np.arange(cam.width),np.arange(cam.height)),axis=-1).astype(np.float32)


def grid(cam,nx,ny,margin=8):
    x = np.rint(np.linspace(margin,cam.width-1-margin,nx))
    y = np.rint(np.linspace(margin,cam.height-1-margin,ny))
    return np.stack(np.meshgrid(x,y),axis=-1).astype(np.float32)


def move(ref,src,ray,z):
    rotation = src.R @ ref.R.T
    translation = src.t-rotation @ ref.t
    return (ray @ rotation.T)*z+translation


def select(ref, pool, maximum, spread):
    xy = grid(ref,7,5,10).reshape(-1,2)
    ray = ref.rays(xy)
    probe_z = np.array([1.,2.5,5.,9.],dtype=np.float32)[None,:,None]
    scores = []
    for j,src in enumerate(pool):
        if src is ref:
            continue
        vec = ref.R @ (src.center-ref.center)
        length = float(np.linalg.norm(vec))
        lateral = float(np.linalg.norm(vec[:2]))
        if lateral < .017 or length > 12 or ref.R[2] @ src.R[2] < .36:
            continue
        _,_,inside = src.project(move(ref,src,ray[:,None],probe_z))
        overlap = inside.mean()
        if overlap<.18:
            continue
        score = overlap-.06*min(length,5)
        angle = np.arctan2(vec[1],vec[0])
        sector = int(np.floor((angle+np.pi/4)/(np.pi/2)))%4
        scores.append((score,j,sector,float(angle),length))
    top = sorted(scores,reverse=True)
    if spread:
        selected = []
        for sector in (0,1,2,3):
            members = [k for k in top if k[2]==sector]
            selected.extend(members[:2])
        included = {v[1] for v in selected}
        selected.extend(v for v in top if v[1] not in included)
        top = selected
    return [(pool[j],sector,a,b) for _,j,sector,a,b in top[:maximum]]


def curves(ref,src,xy,depth=DEPTH):
    left, right = gray(ref),gray(src)
    lxy = xy[:,None,:]+PATCH[None,:,:]
    l = left[lxy[:,:,1].astype(int),lxy[:,:,0].astype(int)]
    l = l-l.mean(axis=1,keepdims=True)
    lnorm = np.maximum(np.sqrt((l*l).sum(1)),.025)
    result = np.empty((len(xy),len(depth)),dtype=np.float32)
    for i in range(0,len(xy),84):
        part = lxy[i:i+84]
        ray = ref.rays(part)[:,None,:,:]
        candidate = move(ref,src,ray,depth[None,:,None,None])
        pix,_,inside = src.project(candidate)
        n = len(part)
        shape = (n,len(depth)*len(PATCH))
        value = cv2.remap(right,pix[...,0].reshape(shape),pix[...,1].reshape(shape),cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT).reshape(pix.shape[:-1])
        value -= value.mean(axis=2,keepdims=True)
        rnorm = np.maximum(np.sqrt(np.sum(value*value,axis=2)),.025)
        score = np.sum(l[i:i+n,None,:]*value,axis=2)/(lnorm[i:i+n,None]*rnorm)
        result[i:i+n] = np.where(inside.all(2)&(lnorm[i:i+n,None]>.025),score,np.nan)
    return result


def peak(cost,depth=DEPTH):
    ok = np.isfinite(cost)
    active = ok.any(axis=1)
    c = np.where(ok,cost,-2.)
    i = c.argmax(axis=1)
    row = np.arange(len(i))
    l,h = c[row,(i-1).clip(0)],c[row,(i+1).clip(0,len(depth)-1)]
    y = c[row,i]
    with np.errstate(divide="ignore",invalid="ignore"):
        sub = .5*(l-h)/(l+h-2*y)
    accepted = (i>0)&(i<len(depth)-1)&(l>-2)&(h>-2)&np.isfinite(sub)
    correction = np.where(accepted,sub.clip(-.5,.5),0.)
    z = np.exp(np.log(depth[i])+correction*np.log(depth[1]/depth[0]))
    return np.where(active,z,np.nan).astype(np.float32),np.where(active,y,np.nan)


def atlas(src,cameras,excluded,*,with_reason=False):
    # The visibility proxy never uses the designated target reference camera as a partner.
    contenders = [k for k in select(src,[c for c in cameras if c not in excluded],3,False)]
    xy = grid(src,30,21,5)
    flat = xy.reshape(-1,2)
    result = np.full(xy.shape[:2],np.nan,dtype=np.float32)
    if len(contenders)<2:
        if with_reason:
            return xy,result,np.ones(xy.shape[:2],dtype=np.int8)
        return xy,result
    depths = np.geomspace(.35,25,144).astype(np.float32)
    estimate = np.column_stack([peak(curves(src,c,flat,depths),depths)[0] for c,*_ in contenders])
    finite = np.isfinite(estimate)
    lo = np.min(np.where(finite,estimate,np.inf),axis=1)
    hi = np.max(np.where(finite,estimate,-np.inf),axis=1)
    agree = (finite.sum(1)>=2)&(hi<1.16*lo)
    total = np.sum(np.where(finite,estimate,0.),axis=1)
    result.flat[agree] = total[agree]/finite.sum(1)[agree]
    if with_reason:
        reason = np.where(agree,0,np.where(finite.sum(1)<2,2,3)).reshape(result.shape)
        return xy,result,reason
    return xy,result


def visibility(ref,src,xy,depth_map):
    pix,z,inside = src.project(move(ref,src,ref.rays(xy)[:,None],DEPTH[None,:,None]))
    grid_xy,values = depth_map
    h,w=values.shape
    xx = np.rint((pix[...,0]-grid_xy[0,0,0])/(grid_xy[0,-1,0]-grid_xy[0,0,0])*(w-1)).astype(int)
    yy = np.rint((pix[...,1]-grid_xy[0,0,1])/(grid_xy[-1,0,1]-grid_xy[0,0,1])*(h-1)).astype(int)
    known = inside&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
    measured = values[yy.clip(0,h-1),xx.clip(0,w-1)]
    # Keep unknown distinct from an observed foreground obstruction.
    occluded = known&np.isfinite(measured)&(measured+np.maximum(.08,.08*measured)<z)
    return np.where(inside,np.where(occluded,.08,.85),0.).astype(np.float32)


def truth_labels(ref,sources,xy):
    z = ref.depth(xy)
    label = np.full((len(xy),len(sources)),-1,dtype=np.int8)
    for i,src in enumerate(sources):
        xyz = move(ref,src,ref.rays(xy),np.nan_to_num(z,nan=1.)[:,None])
        tu,expected,ins = src.project(xyz)
        seen = src.depth(tu)
        use = ins&np.isfinite(z)&np.isfinite(seen)
        tol = np.maximum(.07,.045*expected)
        label[use & (seen+tol < expected),i] = 1
        label[use & (abs(seen-expected)<=tol),i] = 0
    return z,label


def cache(partition):
    return OUT / f"{partition}-directional"


def build(name,partition):
    folder = cache(partition)
    folder.mkdir(exist_ok=True)
    with_truth = partition!="fit"
    cams = scene(name,with_truth)
    eligible = []
    for ref in cams:
        if with_truth and not ref.truth.is_file():
            continue
        found = select(ref,cams,8,True)
        max_sin = max((abs(np.sin(a[2]-b[2])) for a in found for b in found),default=0.)
        if len(found)>=3 and max_sin>=.5:
            eligible.append((ref,found,max_sin))
    if not eligible:
        raise ValueError(f"No non-collinear reference views: {name}")
    selected = [eligible[i] for i in np.linspace(0,len(eligible)-1,min(5,len(eligible)),dtype=int)]
    anchors = [ref for ref,_,_ in selected]
    depths = {}
    for i,(ref,found,max_sin) in enumerate(selected):
        output = folder/f"{name}-{i}.npz"
        truth_output = output.with_suffix(".truth.npz")
        if output.is_file() and (not with_truth or truth_output.is_file()):
            continue
        xy = grid(ref,38,26).reshape(-1,2)
        sources = [s[0] for s in found]
        costs,visibility_weights = [],[]
        for cam in sources:
            if cam not in depths:
                depths[cam] = atlas(cam,cams,anchors)
            costs.append(curves(ref,cam,xy))
            visibility_weights.append(visibility(ref,cam,xy,depths[cam]))
        cost = np.stack(costs,axis=1)
        v = np.stack(visibility_weights,axis=1)
        peaks = np.column_stack([peak(c)[0] for c in costs])
        # Save features and candidate data before opening the LiDAR maps.
        temp = output.with_suffix(".partial.npz")
        np.savez_compressed(temp,tally=xy,cost=cost.astype(np.float16),visibility=v.astype(np.float16),peaks=peaks,
                            groups=[s[1] for s in found], angles=[s[2] for s in found],
                            offsets=[s[3] for s in found],directions=max_sin,
                            rgb=rgb(ref),ref=str(ref.path),sources=[str(c.path) for c in sources],scene=name)
        temp.replace(output)
        if with_truth:
            z,labels = truth_labels(ref,sources,xy)
            np.savez_compressed(temp,truth=z,occlusion=labels)
            temp.replace(truth_output)
        print(partition,name,i,len(found),round(max_sin,3),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("partition",choices=("cal","held","fit"))
    parser.add_argument("--scene")
    opts=parser.parse_args()
    cv2.setNumThreads(1)
    for name in {"cal":CAL,"held":HELD,"fit":FIT}[opts.partition]:
        if opts.scene is None or name==opts.scene:
            build(name,opts.partition)
