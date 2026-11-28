"""Fixed-camera boundary-motion teacher, integrable shape correction and scene-level evaluation."""
from collections import defaultdict
from functools import lru_cache
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from depth_full_data import hypersim_intrinsics
from light_encoder_score import middlebury_pairs
from light_encoder_study import CKPT, Model, sha, yield_memory
from material_light_assets import OUT as RENDERS, save_json


OUT = RENDERS.parent/"edge-motion-43"
ARMS = ("moving","unique","shift")
RHO = 1/16


def resize(x,size,nearest=False):
    return cv2.resize(x,size,interpolation=cv2.INTER_NEAREST_EXACT if nearest else cv2.INTER_LINEAR)


def grid(rgb):
    h,w = rgb.shape[:2]
    return 512, 2*round(h*256/w)


def gray(rgb,size):
    img = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
    intensity = img@np.array([.2126,.7152,.0722],np.float32)
    return cv2.GaussianBlur(np.log(.04+intensity),(5,5),.9)


def signed_peaks(x):
    diff = np.zeros_like(x)
    diff[:,:-1] = np.diff(x,axis=1)
    strength = abs(diff)
    mask = (strength>.075)&(strength>np.roll(strength,1,1))&(strength>=np.roll(strength,-1,1))
    mask[:,:2]=False
    mask[:,-2:]=False
    return np.where(mask,np.sign(diff),0).astype(np.int8)


def nearest_along(peaks):
    width=peaks.shape[1]
    x=np.arange(width,dtype=np.int32)[None,:]
    left=np.maximum.accumulate(np.where(peaks,x,-3*width),axis=1)
    right=np.minimum.accumulate(np.where(peaks,x,4*width)[:,::-1],axis=1)[:,::-1]
    return np.where(x-left<=right-x,left,right)


def halo(mask,radius=2):
    return cv2.dilate(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)


def correspondence(first,second):
    width=first.shape[1]
    x=np.arange(width)[None,:]
    moving=np.zeros_like(first,bool)
    fixed=moving.copy()
    matches=np.full(first.shape,-1,np.int32)
    for s in (-1,1):
        aa,bb=first==s,second==s
        ab=nearest_along(bb)
        ba=nearest_along(aa)
        j=np.clip(ab,0,width-1)
        dist=abs(ab-x)
        mutual=aa&(np.take_along_axis(ba,j,1)==x)
        stay=mutual&(dist<=1)
        move=mutual&(dist>=4)&(dist<=20)
        fixed |= stay
        moving |= move
        matches=np.where(move,j,matches)
    exclusive=(first!=0)&~halo(second!=0,2)
    return fixed,moving&exclusive,exclusive,matches


def orientation(a,b):
    first,second=signed_peaks(a),signed_peaks(b)
    fa,ma,ua,ja=correspondence(first,second)
    fb,mb,ub,jb=correspondence(second,first)
    return dict(fixed=fa|fb,ma=ma,mb=mb,ua=ua,ub=ub,ja=ja,jb=jb)


def bands(x,y):
    a,b=halo(x,2),halo(y,2)
    return a&~b,b&~a


@lru_cache(maxsize=16)
def frequencies(h,w):
    return (RHO+2-2*np.cos(np.pi*np.arange(h)/h)[:,None]+2-2*np.cos(np.pi*np.arange(w)/w)[None,:]).astype(np.float32)


def inverse(rhs,check=False):
    v=cv2.idct(cv2.dct(np.ascontiguousarray(rhs))/frequencies(*rhs.shape))
    if check:
        lhs=RHO*v
        for axis in (0,1):
            a,b=neighbors(axis,1,1)
            diff=v[b]-v[a]
            lhs[a]-=diff
            lhs[b]+=diff
        if np.abs(lhs-rhs).max()>2e-5:
            raise ArithmeticError("Neumann system residual exceeded tolerance")
    return v


def neighbors(axis,d,step):
    a,b=[slice(None,None,step)]*2,[slice(None,None,step)]*2
    a[axis],b[axis]=slice(None,-d,step),slice(d,None,step)
    return tuple(a),tuple(b)


def edge(field,valid,threshold):
    out=np.zeros_like(valid)
    for axis in (0,1):
        a,b=neighbors(axis,1,1)
        jump=(abs(field[a]-field[b])>threshold)&valid[a]&valid[b]
        out[a]|=jump
        out[b]|=jump
    return out


def observations(im_a,im_b,za,zb):
    sz=grid(im_a)
    a,b=gray(im_a,sz),gray(im_b,sz)
    # Keep the prior equal to the arithmetic mean of native metric depth.
    base=np.log((za.astype(np.float64)+zb)*.5)
    low=resize(base.astype(np.float32),sz)
    logs=[resize(np.log(z),sz) for z in (za,zb)]
    orient=[orientation(a,b),orientation(a.T,b.T)]
    guard=halo(orient[0]["fixed"]|orient[1]["fixed"].T,2)
    for i,obs in enumerate(orient):
        normal_guard=guard.T if i else guard
        # Both endpoints must survive exclusion, maintaining reciprocal labels.
        for key,other,mt in (("ma","mb","ja"),("mb","ma","jb")):
            j=obs[mt].clip(0,a.T.shape[1]-1 if i else a.shape[1]-1)
            obs[key] &= ~normal_guard & np.take_along_axis(~normal_guard&obs[other],j,axis=1)
    rhs={name:np.zeros_like(low) for name in ARMS}
    for i,obs in enumerate(orient):
        axis=1-i
        a1,b1=neighbors(axis,1,1)
        residuals=[np.diff(l-low,axis=axis) for l in logs]
        for arm in ARMS:
            keys=("ua","ub") if arm=="unique" else ("ma","mb")
            one,two=bands(obs[keys[0]],obs[keys[1]])
            if i:
                one,two=one.T,two.T
            if arm!="unique":
                one,two=one&~guard,two&~guard
            if arm=="shift":
                one,two=np.roll(one,low.shape[1]//4,axis=1),np.roll(two,low.shape[1]//4,axis=1)
            target=residuals[1]*one[a1]+residuals[0]*two[a1]
            rhs[arm][a1]-=target
            rhs[arm][b1]+=target
    shape={}
    for arm,field in rhs.items():
        up=resize(inverse(field,True),(base.shape[1],base.shape[0])).astype(np.float64)
        shape[arm]=up-up.mean()
    return base,shape,orient


def quality(orient,z,mask,entity,shadows,sz):
    valid=resize(mask.astype(np.uint8),sz,True)>0
    truth=resize(z,sz,True)
    log=np.log(np.where(valid,truth,1))
    depth=edge(log,valid,np.log(1.05))
    gd=halo(depth,2)
    inside=~halo(~valid,2)
    stationary=orient[0]["fixed"]|orient[1]["fixed"].T
    candidates=[orient[0][key]|orient[1][key].T for key in ("ma","mb")]
    def count(x):
        return int((x&inside).sum())
    result=dict(fixed=count(stationary),fixed_near_depth=count(stationary&gd),
                gt_depth=count(depth),depth_near_fixed=count(depth&halo(stationary,2)),
                moving=sum(count(c) for c in candidates),moving_near_depth=sum(count(c&gd) for c in candidates))
    masks=None
    if shadows is not None:
        s=[edge(resize(item.astype(np.uint8),sz,True).astype(np.float32),valid,.5) for item in shadows]
        near=[halo(v,2) for v in s]
        count_ok=count_total=0
        for i,obs in enumerate(orient):
            for j,(key,match) in enumerate((("ma","ja"),("mb","jb"))):
                cur,other,dom=near[j],near[1-j],inside
                if i:
                    cur,other,dom=cur.T,other.T,dom.T
                target=obs[match].clip(0,other.shape[1]-1)
                active=obs[key]&dom&np.take_along_axis(dom,target,axis=1)
                count_ok+=int((active&cur&~other&np.take_along_axis(other,target,axis=1)&~np.take_along_axis(cur,target,axis=1)).sum())
                count_total+=int(active.sum())
        result.update(matched=count_total,matched_shadow=count_ok,
                      overlap=sum(count(depth&t) for t in near),
                      gt_shadow=sum(count(t) for t in s),
                      shadow_near_moving=sum(count(t&halo(c,2)) for t,c in zip(s,candidates)))
        masks=(depth,s)
    return result,stationary,candidates,masks


def moments(base,corrections,z,valid,entity):
    truth=np.log(np.where(valid,z,1.).astype(float))
    result={}
    for axis in (0,1):
        a,b=neighbors(axis,16,4)
        gt=truth[b]-truth[a]
        ok=valid[a]&valid[b]
        surface=entity[a]==entity[b] if entity is not None else abs(gt)<np.log(1.05)
        r=gt-(base[b]-base[a])
        terms={arm:v[b]-v[a] for arm,v in corrections.items()}
        for key,mask in (("surface",ok&surface),("all",ok),("jump",ok&(abs(gt)>=np.log(1.05)))):
            rec=result.setdefault(key,{"n":0,"base":0.,**{arm:[0.,0.] for arm in ARMS}})
            rec["n"]+=int(mask.sum())
            rec["base"]+=float((r[mask]**2).sum())
            for arm,u in terms.items():
                rec[arm][0]+=float((r[mask]*u[mask]).sum())
                rec[arm][1]+=float((u[mask]**2).sum())
    for rec in result.values():
        n=max(1,rec["n"])
        rec["base"]/=n
        for arm in ARMS:
            rec[arm]=[v/n for v in rec[arm]]
    return result


@torch.inference_mode()
def infer(model,rgb,fx):
    yield_memory()
    native=model.net.infer(model.transform(rgb),f_px=float(fx))
    return (fx/(rgb.shape[1]*native["inverse_depth"].float().clamp_min(1e-5))).cpu().numpy()


def record(name,group,label,rgb_a,rgb_b,za,zb,z,valid,entity,shadows):
    base,h,obs=observations(rgb_a,rgb_b,za,zb)
    q,fixed,moved,masks=quality(obs,z,valid,entity,shadows,grid(rgb_a))
    if name in ("ai_046_005","Books") and label in ("coat-0-1","real"):
        save_example(name,rgb_a,rgb_b,fixed,moved,masks,z,valid)
    return dict(scene=name,group=group,pair=label,quality=q,moments=moments(base,h,z,valid,entity))


def save_example(name,a,b,fixed,moved,masks,z,valid):
    sz=grid(a)
    tiles=[]
    for idx,rgb in enumerate([a,b]):
        im=cv2.resize(rgb,sz,interpolation=cv2.INTER_AREA)
        tiles.append(im)
        layer=(im*.52).astype(np.uint8)
        layer[halo(fixed,1)]=[75,203,240]
        layer[halo(moved[idx],1)]=[255,144,52]
        tiles.append(layer)
    Image.fromarray(np.concatenate([np.concatenate(tiles[::2],1),np.concatenate(tiles[1::2],1)],0)).save(OUT/(name+".jpg"),quality=91)
    if masks:
        g,s=masks
        overlay=(tiles[0]*.48).astype(np.uint8)
        overlay[halo(g,1)]=[75,203,240]
        overlay[halo(s[0],1)]=[255,144,52]
        overlay[halo(g,1)&halo(s[0],1)]=[235,83,160]
        Image.fromarray(overlay).save(OUT/"overlap.jpg",quality=92)


def scene_rows(model,row):
    folder=RENDERS/"renders"/row["scene"]
    with np.load(folder/"passes.npz") as f:
        z,ent=f["z"],f["entity"]
        valid=f["valid"]&(z>=.25)&(z<=30)
        shadows=[f[f"shadow{i}"] for i in range(3)]
    fx=hypersim_intrinsics(row["scene"],z.shape[1],z.shape[0])[0,0]
    result=[]
    for mat in ("matte","coat"):
        rgb=[np.array(Image.open(folder/f"{mat}-{i}.png").convert("RGB")) for i in range(3)]
        depths=[infer(model,img,fx) for img in rgb]
        for i,j in itertools.combinations(range(3),2):
            result.append(record(row["scene"],row["family"],f"{mat}-{i}-{j}",rgb[i],rgb[j],depths[i],depths[j],
                                 z,valid,ent,(shadows[i],shadows[j])))
    return result


def levels(rows,key):
    groups=defaultdict(list)
    for r in rows:
        if r["moments"][key]["n"]:
            groups[r["group"]].append(r["moments"][key])
    return {g:{k:np.mean([r[k] for r in rs],axis=0) for k in ("base",*ARMS)} for g,rs in groups.items()}


def evaluate(rows,locks):
    results={}
    for key in ("surface","all","jump"):
        bygroup=levels(rows,key)
        grouped=[]
        for g,rec in bygroup.items():
            scores={"half":float(rec["base"])}
            for arm,lam in {**locks,"unit":1.}.items():
                a,c=rec["moving" if arm=="unit" else arm]
                scores[arm]=float(rec["base"]-2*lam*a+lam**2*c)
            grouped.append(dict(group=g,scores=scores))
        vals={k:np.array([r["scores"][k] for r in grouped]) for k in grouped[0]["scores"]}
        ix=np.random.default_rng(43626).integers(0,len(grouped),(10000,len(grouped)))
        bs=100*(1-vals["moving"][ix].mean(1)/vals["half"][ix].mean(1))
        results[key]=dict(mean={k:float(v.mean()) for k,v in vals.items()},interval=np.quantile(bs,[.025,.975]).tolist(),
                          wins=int((vals["moving"]<vals["half"]).sum()),grouped=grouped)
    quality={k:sum(row["quality"][k] for row in rows) for k in rows[0]["quality"]}
    return dict(**results,quality=quality)


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    model=Model()
    manifest=json.loads((RENDERS/"scenes.json").read_text())
    all_rows={}
    for split in ("train","val"):
        rows=[]
        for row in (r for r in manifest if r["split"]==split):
            rows.extend(scene_rows(model,row))
            print(split,row["scene"],len(rows),flush=True)
        save_json(OUT/f"{split}-rows.json",rows)
        if split=="train":
            group=levels(rows,"surface")
            moments_fit={k:np.mean([v[k] for v in group.values()],axis=0).tolist() for k in ARMS}
            locks={k:float(np.clip(a/c,0,1)) if c>0 else 0. for k,(a,c) in moments_fit.items()}
            save_json(OUT/"lock.json",dict(strength=locks,moments=moments_fit,model=sha(CKPT),code=sha(Path(__file__)),scenes=list(group)))
        all_rows[split]=evaluate(rows,locks)
    rows=[]
    for name,group,a,b,z,mask in middlebury_pairs():
        za,zb=infer(model,a,3740/3),infer(model,b,3740/3)
        rows.append(record(name,group,"real",a,b,za,zb,z.cpu().numpy(),mask.cpu().numpy(),None,None))
    save_json(OUT/"real-rows.json",rows)
    all_rows["real"]=evaluate(rows,locks)
    save_json(OUT/"result.json",dict(sets=all_rows,lock=sha(OUT/"lock.json")))


if __name__=="__main__":
    main()
