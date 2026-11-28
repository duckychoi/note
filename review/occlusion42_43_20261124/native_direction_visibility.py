"""Post-hoc synthetic calibration audit of visibility and the disparity search range."""
from collections import defaultdict
import json
import math
import numpy as np

from native_direction_data import OUT,cache_path,write
from native_direction_fit import valid
from depth_full_data import camera_ray_lengths,h5_array,hypersim_intrinsics,poses
from hypersim_fetch import frame_assets

def main():
    inv=json.loads((OUT/"inventory.json").read_text())
    tracks=json.loads((OUT/"tap-inventory.json").read_text())
    records=defaultdict(lambda:defaultdict(list));counts=defaultdict(int)
    for row in inv["rows"]:
        split,scene,cam,f,g=row
        if split!="cal" or f==g:
            continue
        trajectory=poses(scene,cam)
        if f not in trajectory or g not in trajectory:
            continue
        with np.load(cache_path(OUT/inv["folder"],row)) as data,np.load(cache_path(OUT/tracks["folder"],row)) as tr:
            iy,ix=tr["iy"],tr["ix"];p=(iy,ix)
            z=data["z"][p];a=data["A"][p];one=data["d1"][p];two=data["d2"][p]
            h,w=data["shape"];sh,sw=data["z"].shape
            dd=a/np.maximum(z,.001)
            K=hypersim_intrinsics(scene,w,h)
            # Low-resolution query integer centres mapped back to the native RGB camera.
            uv=np.stack([(2*ix+.5)*w/(2*sw)-.5,(2*iy+.5)*h/(2*sh)-.5,np.ones(len(ix))],axis=1)
            rf,cf=trajectory[f];rg,cg=trajectory[g]
            xyz=((uv@np.linalg.inv(K).T)*z[:,None])@(rg.T@rf).T+rg.T@(cf-cg)
            projected=xyz@K.T
            target=projected[:,:2]/np.maximum(projected[:,2:],1e-8)
            target=np.nan_to_num(target,nan=-1e8,posinf=-1e8,neginf=-1e8)
            inside=(target[:,0]>=0)&(target[:,0]<w-1)&(target[:,1]>=0)&(target[:,1]<h-1)&(xyz[:,2]>0)
            tx=np.rint(target[:,0].clip(0,w-1)).astype(int);ty=np.rint(target[:,1].clip(0,h-1)).astype(int)
            _,(_,dep)=frame_assets(scene,cam,g)
            other_range=h5_array(dep)
            measured=other_range[ty,tx]/camera_ray_lengths((h,w),tuple(K.ravel()))[ty,tx]
            visible=inside & valid(measured) & (np.abs(measured-xyz[:,2])<=np.maximum(.03,.02*xyz[:,2]))
            for teacher,(d1,d2) in {"tap":(tr["tap1"],tr["tap2"]),"sgbm":(one,two)}.items():
                ok=np.isfinite(d1)&np.isfinite(d2)&valid(z)
                res=np.maximum(0,np.maximum(np.minimum(d1,d2)-dd,dd-np.maximum(d1,d2)))
                for key,m in (("all",ok),("inside",ok&inside),("visible",ok&visible),
                              ("visible_d_le95",ok&visible&(dd<=95))):
                    if m.any():
                        records[(teacher,key)][scene].append(res[m])
                    counts[(teacher,key)]+=int(m.sum())
    out={}
    for (name,kind),scenes in records.items():
        v=sorted(float(np.quantile(np.concatenate(a),.9)) for a in scenes.values())
        idx=min(len(v),math.ceil(.9*(len(v)+1)))-1
        out.setdefault(name,{})[kind]=dict(points=counts[(name,kind)],scenes=len(v),width_px=v[idx])
    write(OUT/"visibility-diagnostic.json",out)
    print(out,flush=True)

if __name__=="__main__":
    main()
