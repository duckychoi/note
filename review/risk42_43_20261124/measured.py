"""Freeze verified real depth endpoints from the RGB-only regions and independent light scans."""
import hashlib
import json

import cv2
import numpy as np

from acquire import HERE, CACHE, DEV, TEST

PROCESSED=CACHE/"processed"


def image(path,flags):
    return cv2.imdecode(np.fromfile(path,np.uint8),flags)


def pfm(path):
    with path.open("rb") as stream:
        assert stream.readline().strip()==b"Pf"
        width,height=map(int,stream.readline().split())
        scale=float(stream.readline())
        arr=np.fromfile(stream,np.dtype("<f4" if scale<0 else ">f4"))
    # Middlebury stores absolute disparity in pixels. Header magnitude is a
    # brightness/display scale; applying 0.003922 here destroys the stereo warp.
    return arr.reshape(height,width)[::-1]


def endpoint_indices(z,valid,stable,boxes):
    h,w=z.shape
    yy,xx=np.indices((h,w))
    region=np.zeros_like(valid)
    for l,t,r,b in boxes:
        region|=((xx+.5)/w>=l)&((xx+.5)/w<=r)&((yy+.5)/h>=t)&((yy+.5)/h<=b)
    possible=stable&region
    used=np.zeros_like(valid)
    found=[]
    for radius in (3,5,8):
        for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
            def neighbor(a,y,x):
                return np.roll(a,(-y,-x),(0,1))
            def inside(y,x):
                return (yy+y>=0)&(yy+y<h)&(xx+x>=0)&(xx+x<w)
            ys,xs=dy*radius,dx*radius
            a,b=neighbor(z,ys,xs),neighbor(z,-ys,-xs)
            good=possible&~used&inside(ys,xs)&inside(-ys,-xs)
            good&=neighbor(stable,ys,xs)&neighbor(stable,-ys,-xs)
            good&=(a-z>=.05)&(b-z>=.05)&(np.abs(a-b)<=np.maximum(.06,.025*(a+b)))
            for sign in (-1,1):
                ty,tx=sign*2*dx,-sign*2*dy
                good&=inside(ty,tx)&neighbor(valid,ty,tx)
                good&=np.abs(neighbor(z,ty,tx)-z)<=np.maximum(.03,.02*z)
            y,x=np.nonzero(good)
            if len(y):
                found.append(np.array([y*w+x,(y+ys)*w+x+xs,(y-ys)*w+x-xs],np.int32).T)
                used|=good
    return np.concatenate(found) if found else np.empty((0,3),np.int32)


def prepare(scene,regions):
    root=CACHE/scene
    calib=dict(line.split("=",1) for line in (root/"calib.txt").read_text().splitlines())
    f=float(calib["cam0"][1:].split()[0])
    baseline=float(calib["baseline"])/1000
    offset=float(calib["doffs"])
    W,H=int(calib["width"]),int(calib["height"])
    w=320
    h=round(w*H/W)
    dst={}
    for name in ("im0","im1"):
        rgb=cv2.cvtColor(image(root/(name+".png"),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        assert rgb.shape[:2]==(H,W)
        dst[name]=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    dst.update(focal=f*w/W,baseline=baseline,offset=offset*w/W)
    disp,sd=pfm(root/"disp0.pfm"),pfm(root/"disp0-sd.pfm")
    n=image(root/"disp0-n.pgm",cv2.IMREAD_UNCHANGED)
    assert all(a.shape==(H,W) for a in (disp,sd,n))
    if scene in DEV:
        gy,gx=np.indices((4,8))
        y=np.floor((gy.ravel()+.5)*h/4).astype(int)
        x=np.floor((gx.ravel()+.5)*w/8).astype(int)
    else:
        y,x=np.indices((h,w))
    iy=np.floor((y+.5)*H/h).astype(int)
    ix=np.floor((x+.5)*W/w).astype(int)
    def sample(dy,dx):
        rows,cols=iy+dy,ix+dx
        assert rows.min()>=0 and rows.max()<H and cols.min()>=0 and cols.max()<W
        denom=disp[rows,cols].astype(np.float64)+offset
        denom=np.where(np.isfinite(denom)&(denom>.25),denom,1e-12)
        z=f*baseline/denom
        good=(z>=.25)&(z<=8)&(n[rows,cols]>=3)&(sd[rows,cols]>=0)&(sd[rows,cols]<=.25)
        bound=f*baseline*.25/(denom*np.maximum(1e-12,denom-.25))
        return z.astype(np.float32),good&(bound<=.005),bound
    z,valid,bound=sample(0,0)
    stat=dict(output=[h,w],valid=int(valid.sum()),sampled=int(valid.size))
    if scene in DEV:
        dst.update(y=y[valid],x=x[valid],z=z[valid])
        triples=np.empty((0,3),np.int32)
    else:
        lo,hi=z.copy(),z.copy()
        stable=valid.copy()
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy or dx:
                    adj,ok,_=sample(dy,dx)
                    stable&=ok
                    lo=np.minimum(lo,adj)
                    hi=np.maximum(hi,adj)
        stable&=(hi-lo<=.02)
        z=np.where(valid,z,0)
        triples=endpoint_indices(z,valid,stable,regions[scene]["boxes"])
        dst.update(z=z,valid=valid,triples=triples)
        idx=triples.ravel()
        stat.update(stable=int(stable.sum()),fronts=len(triples),pairs=2*len(triples),
                    thin_eligible=bool(len(triples)>=8),roi=regions[scene]["item"],
                    max_native_sigma_px=float(sd[iy,ix].ravel()[idx].max()) if len(idx) else None,
                    max_depth_sensitivity_mm=float(bound.ravel()[idx].max()*1000) if len(idx) else None)
        if len(idx):
            jumps=z.ravel()[triples[:,1:]]-z.ravel()[triples[:,0,None]]
            stat.update(min_jump_mm=float(jumps.min()*1000),mean_jump_mm=float(jumps.mean()*1000))
            assert jumps.min()>=.05-1e-6
        assert np.all(stable.ravel()[idx])
    np.savez_compressed(PROCESSED/(scene+".npz"),**dst)
    return stat,triples,dst["im0"]


def main():
    PROCESSED.mkdir(exist_ok=True,parents=True)
    region_file=HERE/"regions.json"
    regions=json.loads(region_file.read_text(encoding="utf8"))
    assert set(regions)==set(TEST)
    quality={}
    pairs={}
    cards=[]
    for scene in DEV+TEST:
        stat,triples,rgb=prepare(scene,regions)
        quality[scene]=stat
        if scene not in DEV:
            pairs[scene]=triples
            card=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            h,w=card.shape[:2]
            for front,back1,back2 in triples[np.linspace(0,len(triples)-1,min(40,len(triples)),dtype=int)]:
                for idx,col in ((back1,(255,255,10)),(back2,(255,255,10)),(front,(5,255,255))):
                    cv2.circle(card,(int(idx%w),int(idx//w)),1,col,-1)
            canvas=np.full((265,320,3),35,np.uint8)
            height=min(h,240)
            canvas[:height]=cv2.resize(card,(320,height),interpolation=cv2.INTER_AREA)
            cv2.putText(canvas,f"{scene}  {2*len(triples)} pairs",(6,258),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            cards.append(canvas)
    np.savez_compressed(HERE/"endpoint_pairs.npz",**pairs)
    cv2.imencode(".jpg",np.vstack([np.hstack(cards[i:i+3]) for i in range(0,len(cards),3)]))[1].tofile(HERE/"endpoints.jpg")
    out=dict(region_sha256=hashlib.sha256(region_file.read_bytes()).hexdigest(),scenes=quality)
    (HERE/"quality.json").write_text(json.dumps(out,indent=1),encoding="utf8")
    print("dev",[(s,quality[s]["valid"]) for s in DEV],"test",[(s,quality[s]["pairs"]) for s in TEST])


if __name__=="__main__":
    main()
