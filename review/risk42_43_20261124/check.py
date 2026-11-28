"""Unit landmarks, independent real-image reprojection, and held-out teacher diagnostics."""
import json
import cv2
import numpy as np
import torch

from acquire import HERE,TEST
from measured import endpoint_indices,PROCESSED
from evaluate import metrics
from fit import med,estimate_gain,SEEDS,CKPT,DepthNet,predict,tensor_input,correct,observe
from paired import optimal,risks


def algebra():
    rng=np.random.default_rng(84243)
    vals=rng.uniform(.3,3,(6,71))
    grid=np.linspace(0,1,1001)
    k=optimal(vals)
    assert np.mean(risks(vals,k))<=min(np.mean(risks(vals,v)) for v in grid)+1e-10
    a,b,s,t,za,zb=vals
    k=estimate_gain(za,a,s)
    assert np.mean(abs(a+k*(s-a)-za))<=min(np.mean(abs(a+v*(s-a)-za)) for v in grid)+1e-10
    p=med(za/a,a)
    assert np.mean(abs(a*p-za))<=min(np.mean(abs(a*v-za)) for v in np.linspace(.3,3,1001))+1e-10
    assert optimal(np.array([a,b,a,b,za,zb]))==0
    z=np.full((32,44),2.,np.float32)
    z[:,20:23]=1
    mask=z>0
    trips=endpoint_indices(z,mask,mask,[[0,0,1,1]])
    scene=dict(z=z,valid=mask,triples=trips)
    assert len(trips)>16 and metrics(z,scene)["jump"]==0
    assert metrics(np.ones_like(z),scene)["jump"]==1000 and metrics(3-z,scene)["jump"]==2000


def main():
    algebra()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    photos={}
    real_audit={}
    nets={}
    ledgers={}
    for seed in SEEDS:
        net=DepthNet(False).cuda()
        net.load_state_dict(torch.load(CKPT/f"{seed}_prior_net.pt",weights_only=True))
        nets[seed]=net.eval()
        ledgers[seed]=json.loads((HERE/f"seed{seed}.json").read_text())
    for scene in TEST:
        with np.load(PROCESSED/(scene+".npz")) as d:
            left,right=d["im0"],d["im1"]
            z,valid=d["z"],d["valid"]
            f,B,off=d["focal"],d["baseline"],d["offset"]
            yy,xx=np.indices(z.shape,np.float32)
            disp=f*B/np.maximum(z,.001)-off
            keep=valid&(xx-disp>=1)&(xx-disp<z.shape[1]-1)&(disp>=0)
            ref=left[keep].astype(float)
            def photometric(mult):
                warped=cv2.remap(right,(xx-disp*mult).astype(np.float32),yy,cv2.INTER_LINEAR)
                return float(abs(ref-warped[keep]).mean())
            photos[scene]=dict(points=int(keep.sum()),unwarped=photometric(0),raw_disparity=photometric(1),
                               display_scaled=photometric(.003922))
            x=tensor_input(left,f)[None].cuda()
            data=dict(zip(("stereo","ok","grad","spread"),observe(left,right,f,B,off,128)[0]))
            q=valid&data["ok"]
            real_audit[scene]={}
            tri=d["triples"]
            if len(tri):
                comp=data["ok"].ravel()[tri]
                photos[scene]["paired_stereo_coverage"]=float((comp[:,0,None]&comp[:,1:]).mean())
            for seed in SEEDS:
                m=predict(nets[seed],x)[0]
                rec=ledgers[seed]
                new,g=correct(m,data,np.array(rec["gain"]),)
                blind=data["stereo"]
                def err(v):
                    return 1000*np.abs(v[q]-z[q])
                before,after=err(m),err(new)
                order=np.argsort(np.asarray(rec["utility_m"])[g[q]],kind="stable")
                parts=np.array_split(order,4)
                real_audit[scene][seed]=dict(before=float(before.mean()),corrected=float(after.mean()),
                    stereo=float(err(blind).mean()),n=int(q.sum()),
                    conditional_reductions=[float(np.mean((before-after)[i])) for i in parts])
    (HERE/"independent_checks.json").write_text(json.dumps(dict(reprojection=photos,teacher=real_audit),allow_nan=False),encoding="utf8")
    print("Unit and geometric checks passed; scene mean raw/unwarped/display scaled",
          [float(np.mean([v[k] for v in photos.values()])) for k in ("raw_disparity","unwarped","display_scaled")])


if __name__=="__main__":
    main()
