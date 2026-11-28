"""Score matched real brackets and independent virtual captures on measured surfaces."""
from collections import defaultdict
import argparse
import json
import math

import cv2
import numpy as np
from scipy.optimize import minimize

from repeat_capture_acquire import OUT
from repeat_capture_render import BASELINE, FOCAL, LOG_MIN, VIEWS, geometry, load_gray, sample
from repeat_encoder import DESIGN, LIMIT, family, quadrants


def observations(src,render_dir="renders"):
    x,disp,dmin,truth,xyz,norm,good=geometry(src)
    enc=np.load(OUT/"encoder"/(src.name+".npz"))
    synth=np.load(OUT/render_dir/(src.name+".npz"))
    pred=cv2.resize(enc["inverse"],(x.shape[1],x.shape[0]),interpolation=cv2.INTER_LINEAR)
    model_d=FOCAL*BASELINE*pred-dmin
    true_x=[x-v*disp for v in VIEWS]
    model_x=[x-v*model_d for v in VIEWS]
    id_maps=synth["source_x"]
    for v,tx in enumerate(true_x):
        support=synth["valid"][v].astype(np.float32)
        good &= (sample(support,tx)>.999)
        good &= np.abs(sample(id_maps[v],tx)-x)<.6
    regions=quadrants(*x.shape)
    interior=np.logical_or.reduce(regions)
    before=int((good&interior).sum())
    eps=.001
    stages=(true_x,model_x,[pos-v*FOCAL*BASELINE*eps for pos,v in zip(model_x,VIEWS)],
            [pos+v*FOCAL*BASELINE*eps for pos,v in zip(model_x,VIEWS)])
    for v in range(2):
        support=synth["valid"][v].astype(np.float32)
        for maps in stages[1:]:
            good &= sample(support,maps[v])>.999
    keep=good & np.logical_or.reduce(regions)
    if not keep.any():
        raise ValueError(f"{src.name}: no common surface")
    def cycle(images, maps):
        # c'y for y=reference-target: target difference at view 5 minus that at view 3.
        contrast=[]
        for v,pos in enumerate(maps):
            contrast.append(sample(images[v,0]-images[v,1],pos))
        return contrast[1]-contrast[0]
    def logs(a):
        return np.log(np.clip(a,LOG_MIN,1))
    photos=np.array([[[load_gray(src/f"Illum{l}/Exp{r}/view{v}.png") for l in (1,2)]
                       for v in (3,5)] for r in (1,2)])
    rows={}
    frames=synth["frames"]
    means=synth["expected"]
    for mode,raw,nominal in (
        ("physical_bracket",photos,photos[0]),
        ("diffuse",frames[0].astype(np.float32)/65535,means[0]),
        ("glossy",frames[1].astype(np.float32)/65535,means[1])
    ):
        readings=np.array([[cycle(logs(rep),maps)[keep] for maps in stages[:2]] for rep in raw])
        expected=np.array([cycle(logs(nominal),maps)[keep] for maps in stages[:2]])
        slope=-(cycle(logs(nominal),stages[2])-cycle(logs(nominal),stages[3]))/(2*eps)
        direction=sum(float(slope[keep&region].sum(dtype=float))*h
                      for region,h in zip(regions,enc["H"]))/keep.sum()
        ideal=None if mode=="physical_bracket" else synth["ideal_cycle"][int(mode=="glossy")][keep]
        rows[mode]=(readings.astype(float),expected.astype(float),direction,ideal)
    source_mask=np.logical_or.reduce(regions)
    return enc,rows,dict(gt_visible_interior=before,gt_valid_interior=int(((disp>0)&source_mask).sum()),
                          eligible=int(keep.sum()),pixels=x.size,
                          median_model_depth=float(np.median(1/np.maximum(pred[keep],1e-8))),
                          median_gt_depth=float(np.median(1/truth[keep])))


def mean(a):
    return float(np.mean(a,dtype=np.float64))


def score_scene(raw,expected,shift,physical,ideal):
    gt,model=raw[:,0],raw[:,1]
    total=model-shift
    def moment(a,b):
        if physical:
            # Raw second moments (bracket mixture); no identical-repeat claim.
            return mean((a[0]*b[0]+a[1]*b[1])/2)
        return mean((a[0]*b[1]+a[1]*b[0])/2)
    extra=total-gt
    metrics=dict(photo=moment(gt,gt),warped=moment(model,model),total=moment(total,total),
                warp_difference=moment(model-gt,model-gt),extra=moment(extra,extra),
                cross=2*moment(gt,extra),abs_95=float(np.quantile(abs(total),.95)))
    if physical:
        metrics["bracket_disagreement"]=mean((model[0]-model[1])**2/2)
    else:
        metrics.update(noise=mean((model[0]-model[1])**2/2),expected_photo=mean(expected[0]**2),
                       expected_total=mean((expected[1]-shift)**2),ideal_same_point=mean(ideal**2))
    return metrics


def certificate(M,j,Q,B,N,eta):
    g=M@j
    Mp=M+eta*np.eye(len(j))
    eig,vec=np.linalg.eigh(Mp)
    ell=(vec*np.sqrt(np.maximum(eig,0)))@vec.T
    radius=eta*np.linalg.norm(j)
    second=(math.sqrt(max(0,j@Mp@j+N))+B)**2
    if np.linalg.norm(g)<=radius or (B>0 and
            np.linalg.norm(np.linalg.lstsq(ell,g,rcond=None)[0])<=B and
            np.linalg.norm(ell@np.linalg.lstsq(ell,g,rcond=None)[0]-g)<1e-12):
        return dict(lower=0.,radius=float(radius),signal=float(np.linalg.norm(g)),D=second)
    k=len(j)
    def obj(y):
        res=g-radius*y[:k]-B*ell@y[k:]
        return float(res@Q@res)
    if radius==0 and B==0:
        dist=obj(np.zeros(2*k))
    else:
        scale=max(float(g@Q@g),1e-18)
        sol=minimize(lambda y:obj(y)/scale,np.zeros(2*k),method="SLSQP",
                     constraints=[dict(type="ineq",fun=lambda y:1-y[:k]@y[:k]),
                                  dict(type="ineq",fun=lambda y:1-y[k:]@y[k:])],
                     options=dict(ftol=1e-13,maxiter=500))
        if not sol.success:
            raise RuntimeError(sol.message)
        residual=g-radius*sol.x[:k]-B*ell@sol.x[k:]
        # Evaluate a dual feasible witness; a numerical primal minimum alone is an upper bound.
        qr=Q@residual
        dist=max(0.,2*qr@g-2*(radius*np.linalg.norm(qr)+B*np.linalg.norm(ell@qr))-residual@qr)
    return dict(lower=float(dist/second) if second else 0.,radius=float(radius),
                signal=float(np.linalg.norm(g)),D=second)


def main(smooth=False):
    cv2.setNumThreads(1)
    paths=sorted((OUT/"physical").iterdir())
    modes=("physical_bracket","diffuse","glossy")
    render_dir="renders-plane" if smooth else "renders"
    design=[observations(OUT/"physical"/name,render_dir) for name in DESIGN]
    J={m:np.mean([row[1][m][2] for row in design],axis=0) for m in modes}
    H=np.concatenate([row[0]["H"] for row in design])
    Q=H.T@H/len(H)
    rows={}
    for p in paths:
        enc,obs,counts=observations(p,render_dir)
        e=enc["e"]
        matte,gloss=obs["diffuse"][1],obs["glossy"][1]
        effects=dict(material_at_gt=mean((gloss[0]-matte[0])**2),
                     material_at_model=mean((gloss[1]-matte[1])**2),
                     geometry_matte=mean((matte[1]-matte[0])**2),
                     geometry_glossy=mean((gloss[1]-gloss[0])**2),
                     interaction=mean(((gloss[1]-gloss[0])-(matte[1]-matte[0]))**2))
        rows[p.name]=dict(family=family(p.name),design_family=family(p.name) in set(map(family,DESIGN)),
            e=e.tolist(),**counts,noiseless_effects=effects,modes={m:score_scene(raw,net,float(J[m]@e),m=="physical_bracket",ideal)
            for m,(raw,net,unused,ideal) in obs.items()})
    groups=defaultdict(list)
    for name,row in rows.items():
        if not row["design_family"]:
            groups[family(name)].append(row)
    def avg(fun):
        return np.mean([np.mean([fun(row) for row in rs],axis=0) for rs in groups.values()],axis=0)
    M=avg(lambda row:np.outer(row["e"],row["e"]))
    mu=avg(lambda row:row["e"])
    P=M-np.outer(mu,mu)
    n=len(groups)
    delta=.05/3
    r=len(mu)
    eta=r*LIMIT**2*np.sqrt(2*np.log(2*r*r/delta)/n)
    U=2*np.log(1/LOG_MIN)
    report={}
    for mode in modes:
        metrics={k:float(avg(lambda row:row["modes"][mode][k])) for k in
                 next(iter(rows.values()))["modes"][mode]}
        assert abs(metrics["total"]-metrics["photo"]-metrics["extra"]-metrics["cross"])<1e-8
        j=J[mode]
        C=U+np.linalg.norm(j)*LIMIT
        physical=mode=="physical_bracket"
        b=math.sqrt(max(0,metrics["total"]))
        # Single-shot Jensen upper bound for the bracket mixture, not a repeat estimate.
        rad=C*C*math.sqrt((.5 if physical else 2)*math.log(1/delta)/n)
        B=math.sqrt(min(C*C,max(0,metrics["total"]+rad)))
        N=U**2 if physical else min(U**2,metrics["noise"]+U**2*math.sqrt(2*math.log(1/delta)/n))
        N0=U**2 if physical else metrics["noise"]
        report[mode]=dict(metrics=metrics,j=j.tolist(),upper_B=B,point_rms=b,upper_noise=N,
            bias_threshold_at_exact_M=math.sqrt(float(j@M@j)),
            strict=certificate(M,j,Q,B,N,eta),point_moments=certificate(M,j,Q,b,N0,0),
            covariance_uncertainty_alone=certificate(M,j,Q,0,N0,eta),
            ignoring_bias_and_estimation=certificate(M,j,Q,0,N0,0))
    contrasts={}
    for k in ("photo","extra","total"):
        changes=[np.mean([r["modes"]["glossy"][k]-r["modes"]["diffuse"][k] for r in group])
                 for group in groups.values()]
        contrasts[k]=dict(glossy_minus_diffuse=float(np.mean(changes)),
                          positive_families=int(np.sum(np.array(changes)>0)))
    effect_names=("material_at_gt","material_at_model","geometry_matte","geometry_glossy","interaction")
    effects={k:float(np.sqrt(avg(lambda row:row["noiseless_effects"][k]))) for k in effect_names}
    paired=np.array([[np.mean([r["noiseless_effects"][k] for r in group])
                      for k in ("geometry_glossy","material_at_gt")] for group in groups.values()])
    selected=np.random.default_rng(261126).integers(0,n,size=(5000,n))
    boot=np.sqrt(paired[selected].mean(axis=1))
    effects["geometry_exceeds_material_families"]=int(np.sum(paired[:,0]>paired[:,1]))
    effects["rms_ratio_family_bootstrap95"]=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()
    result=dict(model=json.loads((OUT/"encoder/model.json").read_text()),scenes=len(rows),
                evaluated_scenes=sum(map(len,groups.values())),families=n,groups=list(groups),
                eligible_pixels=sum(row["eligible"] for group in groups.values() for row in group),
                M=M.tolist(),P=P.tolist(),mu=mu.tolist(),Q=Q.tolist(),eta=float(eta),
                modes=report,material_intervention=contrasts,noiseless_effects=effects,by_scene=rows)
    result["render_dir"]=render_dir
    (OUT/("results-plane.json" if smooth else "results.json")).write_text(json.dumps(result,indent=1,allow_nan=False),encoding="utf8")
    print(n,"families",result["eligible_pixels"],"pixels",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-normals",action="store_true")
    main(parser.parse_args().smooth_normals)
