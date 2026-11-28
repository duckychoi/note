"""Read native structured-light depth and validate both sides of thin foregrounds."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import cv2
import numpy as np
import requests

from common import ALL,DEV,TEST,ROOT,STORE,save
from measured import pfm,image,endpoint_indices

URL="https://vision.middlebury.edu/stereo/data/scenes2021/data/"
FILES=("im0.png","im1.png","disp0.pfm","disp1.pfm","calib.txt")


def fetch(job):
    scene,name=job
    file=STORE/"raw"/scene/name
    file.parent.mkdir(exist_ok=True,parents=True)
    for attempt in range(4):
        try:
            if not file.exists():
                tmp=file.with_suffix(".partial"+file.suffix)
                with requests.get(URL+scene+"/"+name,stream=True,timeout=(15,75)) as resp:
                    resp.raise_for_status()
                    with tmp.open("wb") as dst:
                        for chunk in resp.iter_content(524288):
                            dst.write(chunk)
                if name.endswith(".pfm"):
                    assert pfm(tmp).size==1920*1080
                elif name.endswith(".png"):
                    assert image(tmp,cv2.IMREAD_COLOR) is not None
                else:
                    assert "baseline=" in tmp.read_text()
                tmp.replace(file)
            with file.open("rb") as f:
                return scene+"/"+name,hashlib.file_digest(f,"sha256").hexdigest()
        except (requests.RequestException,OSError,AssertionError,ValueError):
            if attempt==3:
                raise
            time.sleep(1+attempt)


def prepare(s,wide=False):
    base=STORE/"raw"/s
    conf=dict(line.split("=",1) for line in (base/"calib.txt").read_text().splitlines())
    d=pfm(base/"disp0.pfm")
    H,W=d.shape
    f=float(conf["cam0"][1:].split()[0])
    B=float(conf["baseline"])/1000
    off=float(conf["doffs"])
    assert sorted((int(conf["width"]),int(conf["height"])))==sorted((W,H))
    w=320
    h=round(H*w/W)
    rgb=cv2.resize(cv2.cvtColor(image(base/"im0.png",cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB),(w,h),interpolation=cv2.INTER_AREA)
    out=dict(rgb=rgb,fx=f*w/W)
    if s in DEV:
        gy,gx=np.indices((4,8))
        y=((gy.ravel()+.5)*h/4).astype(int)
        x=((gx.ravel()+.5)*w/8).astype(int)
    else:
        y,x=np.indices((h,w))
    Y=((y+.5)*H/h).astype(int)
    X=((x+.5)*W/w).astype(int)
    def sample(dy,dx):
        disp=d[Y+dy,X+dx]
        denom=disp.astype(float)+off
        denom=np.where(np.isfinite(denom)&(denom>.25),denom,1e-8)
        z=f*B/denom
        sens=f*B*.25/(denom*np.maximum(denom-.25,1e-8))
        valid=(z>=.25)&(z<=(30 if wide else 8))&(sens<=(.01*z if wide else .005))
        return z.astype(np.float32),valid
    z,v=sample(0,0)
    stat=dict(valid=int(v.sum()),total=int(v.size))
    if s in DEV:
        out.update(x=x[v],y=y[v],z=z[v])
    else:
        stable=v.copy()
        lo,hi=z.copy(),z.copy()
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                other,good=sample(dy,dx)
                stable&=good
                lo=np.minimum(lo,other)
                hi=np.maximum(hi,other)
        raw=np.where(v,d[Y,X],0)
        xr=np.rint(X-raw).astype(int)
        right=pfm(base/"disp1.pfm")
        stable&=(xr>=0)&(xr<W)&(np.abs(right[Y,xr.clip(0,W-1)]-raw)<=1)&(hi-lo<=(np.maximum(.02,.005*z) if wide else .02))
        z=np.where(v,z,0)
        tri=endpoint_indices(z,v,stable,[[0,0,1,1]])
        if wide and len(tri):
            true=z.ravel()[tri]
            den=f*B/true
            bound=true*.25/(den-.25)
            tri=tri[((true[:,1:]-true[:,0,None])>=2*(bound[:,1:]+bound[:,0,None])).all(1)]
        out.update(triples=tri,valid=v,z=z)
        stat.update(pairs=2*len(tri),stable=int(stable.sum()),eligible=len(tri)>=8)
        if len(tri):
            jump=z.ravel()[tri[:,1:]]-z.ravel()[tri[:,0,None]]
            assert jump.min()>=.05-1e-6 and stable.ravel()[tri].all()
        right_rgb=cv2.resize(cv2.cvtColor(image(base/"im1.png",cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB),(w,h),interpolation=cv2.INTER_AREA)
        dest=(x-raw*w/W).astype(np.float32)
        keep=stable&(dest>=0)&(dest<w-1)
        warped=cv2.remap(right_rgb,dest,y.astype(np.float32),cv2.INTER_LINEAR)
        stat.update(unwarped_rgb=float(abs(rgb[keep].astype(float)-right_rgb[keep]).mean()) if keep.any() else None,
                    warped_rgb=float(abs(rgb[keep].astype(float)-warped[keep]).mean()) if keep.any() else None)
    np.savez_compressed(STORE/(s+("_wide" if wide else "")+".npz"),**out)
    return stat


if __name__=="__main__":
    with ThreadPoolExecutor(10) as pool:
        save("sources.json",dict(pool.map(fetch,[(s,n) for s in ALL for n in FILES])))
    quality={s:prepare(s) for s in DEV+TEST}
    save("quality.json",quality)
    save("wide_quality.json",{s:prepare(s,True) for s in TEST})
    print("development",[(s,quality[s]["valid"]) for s in DEV],
          "test",[(s,quality[s]["pairs"]) for s in TEST],flush=True)
