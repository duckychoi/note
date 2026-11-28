"""An exact mixed-likelihood C pilot along real Depth Pro 2-S weight directions."""
import copy
from collections import defaultdict
from itertools import product
import time

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

import pf42_visibility_encoder as model_code
from pf42_crossview import entries, load
from pf42_monocular import RUN as SPLIT_RUN
from pf42_scale_confidence import CALIBRATION, supported, write
from pf42_visibility import NEW, PARTITION


RUN = model_code.OUT / "c-encoder"
SIGMA, NORMAL, OUTLIER, ETA = .25, .08, .60, .20


def directions(model):
    a = model.encoder.blocks[model.start:].state_dict()
    apple = {k: v.detach().cpu() for k,v in a.items()}
    seed = model_code.SEEDS[0]
    prior = torch.load(model_code.RUN/f"seed-{seed}.pt",weights_only=True,map_location="cpu")["weights"]
    separate = torch.load(SPLIT_RUN/f"seed-{seed}.pt",weights_only=True,map_location="cpu")["weights"]
    base = prior["direct"]
    delta = [{k: lhs[k]-rhs[k] for k in base}
             for lhs,rhs in ((base,apple),(prior["shared"],base),(separate["factor_direct"],base))]
    return base,delta


def set_arm(model, name, u, base, delta):
    layer = copy.deepcopy(model.encoder.blocks[model.start:]).requires_grad_(False)
    layer.load_state_dict({k: p + sum(float(c)*d[k] for c,d in zip(u,delta,strict=True)) for k,p in base.items()})
    model.arms[name]=layer


@torch.no_grad()
def get_predictions(model, obs, keep, names):
    rgb,xy=model_code.rescale(obs["rgb"],obs["tally"][keep])
    x,y=xy.T
    st=model.encode(rgb)
    return {a:model.depth(st,a)[y,x].cpu().numpy().astype(np.float64) for a in names}


def sufficient(j,y):
    n=len(y)
    jc,yc=j-j.mean(0),y-y.mean()
    precision=1/(n*SIGMA**2)
    return j.mean(0),float(y.mean()),jc.T@jc*precision,jc.T@yc*precision,float(yc@yc*precision)


def fit_component(stats, variances):
    a=np.eye(len(stats[0][0]))
    h=np.zeros(len(a))
    c=0.
    for (jm,ym,acenter,hcenter,ccenter),v in zip(stats,variances,strict=True):
        a+=acenter+np.outer(jm,jm)/v
        h+=hcenter+jm*ym/v
        c+=ccenter+ym**2/v+np.log(v)
    mu=np.linalg.solve(a,h)
    log_evidence=-.5*(c-h@mu+np.linalg.slogdet(a)[1])
    return mu,a,log_evidence


def exact_update(stats):
    count=len(stats)
    iid, iid_info, _ = fit_component(stats,[SIGMA**2]*count)
    correlated,_,_=fit_component(stats,[SIGMA**2+NORMAL**2]*count)
    states=np.array(list(product((0,1),repeat=count)))
    components=[fit_component(stats,SIGMA**2+np.where(k,OUTLIER**2,NORMAL**2))
                for k in states]
    means=np.array([r[0] for r in components]); precisions=np.array([r[1] for r in components])
    logp=np.array([r[2] for r in components])+states.sum(1)*np.log(ETA)+(count-states.sum(1))*np.log1p(-ETA)
    mass=np.exp(logp-logsumexp(logp))
    mean=mass@means
    log_norm = np.linalg.slogdet(precisions)[1]/2 + np.log(mass)

    def objective(u):
        ds=u[None]-means
        slopes=np.einsum("bij,bj->bi",precisions,ds)
        v=log_norm-.5*np.sum(ds*slopes,axis=1)
        ld=logsumexp(v)
        return -ld, np.exp(v-ld)@slopes

    candidates=[minimize(objective,x,method="BFGS",jac=True) for x in [mean,*means]]
    mode=min(candidates,key=lambda r:r.fun).x
    covariance = np.einsum("i,ijk->jk",mass,np.linalg.inv(precisions))
    centered=means-mean
    covariance+=(centered.T*mass)@centered
    return dict(iid=iid,normal_bias=correlated,mixture_map=mode,mixture_mean=mean),dict(
        states=len(states),outlier_probability=(mass@states).tolist(),
        information_eigenvalues=np.linalg.eigvalsh(iid_info-np.eye(len(iid))).tolist(),
        posterior_sd=np.sqrt(np.diag(covariance)).tolist())


def run():
    RUN.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    with model_code.gpu_slot():
        while not model_code.capacity():
            time.sleep(5)
        model=model_code.Trainer()
        model.arms={}
        base,delta=directions(model)
        eye=np.eye(len(delta))
        set_arm(model,"base",eye[0]*0,base,delta)
        for i,e in enumerate(eye):
            set_arm(model,f"plus{i}",e,base,delta)
            set_arm(model,f"minus{i}",-e,base,delta)
        collection=defaultdict(list)
        for part,names in CALIBRATION:
            for path in entries(part,names):
                obs=load(path)
                keep=supported(obs)
                if not keep.any():
                    continue
                pred=get_predictions(model,obs,keep,tuple(model.arms))
                y=np.log(np.nanmedian(obs["peaks"][keep],axis=1))-pred["base"]
                jac=np.column_stack([(pred[f"plus{i}"]-pred[f"minus{i}"])*.5 for i in range(len(eye))])
                collection[str(obs["scene"])].append(np.column_stack((jac,y)))
        sample={n:np.concatenate(rows) for n,rows in collection.items()}
        coeff,info=exact_update([sufficient(p[:,:-1],p[:,-1]) for p in sample.values()])
        info["outlier_probability"]=dict(zip(sample,info["outlier_probability"],strict=True))
        print("encoder parameter posterior",{k:v.tolist() for k,v in coeff.items()},info,flush=True)
        for name,u in coeff.items():
            set_arm(model,name,u,base,delta)
        rows=[]
        for path in entries(PARTITION,NEW):
            obs,truth=load(path),load(path.with_suffix(".truth.npz"))["truth"]
            keep=supported(obs)&np.isfinite(truth)&(truth>=.35)&(truth<=25)
            if not keep.any():
                continue
            pred=get_predictions(model,obs,keep,(None,*model.arms))
            target=np.log(truth[keep])
            arms={k:float(abs(np.exp(pred[k]-target)-1).mean()) for k in (None,"base",*coeff)}
            arms["apple"]=arms.pop(None)
            jac=np.column_stack([(pred[f"plus{i}"]-pred[f"minus{i}"])*.5 for i in range(len(eye))])
            approx={k:float(abs(pred[k]-pred["base"]-jac@u).mean()) for k,u in coeff.items()}
            rows.append(dict(image=path.stem,scene=str(obs["scene"]),scores=arms,log_approximation=approx,n=int(keep.sum())))
        grouped={n:{k:float(np.mean([r["scores"][k] for r in rows if r["scene"]==n])) for k in rows[0]["scores"]} for n in NEW}
        means={k:float(np.mean([a[k] for a in grouped.values()])) for k in rows[0]["scores"]}
        write(RUN/"results.json",dict(fit={n:len(v) for n,v in sample.items()},coeff={k:u.tolist() for k,u in coeff.items()},
                                     info=info,means=means,grouped=grouped,rows=rows))
        print("C exact realized weights",means,flush=True)


if __name__=="__main__":
    run()
