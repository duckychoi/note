"""Paired scalar sanity checks for #42, plus a separate source-correlation model."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def recursion(seed,shock):
    # Common draws for all methods. theta's true conditional mean is available,
    # or EVERY predictor can take the same finite synthetic mean perturbation.
    rng=np.random.default_rng(seed)
    n=2048
    theta=np.zeros((120,1))+2
    if shock:
        theta[50:]+=3
    real=theta+rng.normal(size=(120,n))/np.sqrt(5)
    synth=rng.normal(size=(120,n))/np.sqrt(50)
    init=2+rng.normal(size=n)/np.sqrt(5)
    scores={}
    for shared in (False,True):
        for method in ("count_mix","steady_gain","KF","decaying_gain","adaptive_KF"):
            estimate=init.copy()
            cov=np.full(n,.2)
            errors=[]
            for t in range(120):
                prior=estimate+(synth[t] if shared else 0)
                innovation=real[t]-prior
                predicted=cov+.02
                if method=="adaptive_KF":
                    predicted*=1+8*np.maximum(0,innovation**2/(predicted+.2)-1)
                if method in ("KF","adaptive_KF"):
                    gain=predicted/(predicted+.2)
                elif method=="steady_gain":
                    p=(.02+np.sqrt(.02**2+4*.02*.2))/2
                    gain=p/(p+.2)
                elif method=="decaying_gain":
                    gain=1/(t+2)
                else:
                    gain=5/55
                estimate=prior+gain*innovation
                cov=(1-gain)*predicted
                errors.append(np.abs(estimate-theta[t]))
            e=np.mean(errors,axis=1)
            recovered=np.flatnonzero(e[50:]<.3)
            scores[f"{'sampled' if shared else 'exact'}_{method}"]=dict(
                final=float(e[-1]),after_shift=float(e[50:].sum()),
                recovery=int(recovered[0]) if len(recovered) else None)
    return scores


def clustered(seed):
    rng=np.random.default_rng(seed)
    n=4096
    counts=np.array([2,3,5,8,12,20,30,48])
    out={}
    for tau in (0.,.6):
        for bias in (0.,.7):
            parent=rng.normal(0,tau,(n,len(counts)))+bias
            groups=[parent[:,i,None]+rng.normal(0,1,(n,k)) for i,k in enumerate(counts)]
            means=np.stack([g.mean(1) for g in groups],axis=1)
            noise=sum(((g-g.mean(1,keepdims=True))**2).sum(1) for g in groups)/(counts.sum()-len(counts))
            between=np.maximum(0,means.var(1,ddof=1)-noise*(1/counts).mean())
            for anchors in (0,2,8):
                y=rng.normal(0,1,(n,anchors)).sum(1)
                # Fixed mixing per origin and count weighting receive the same anchors.
                raw=counts[None,:]/(1+tau*tau)
                equal=np.ones_like(raw)*len(counts)/(len(counts)*(tau*tau+(1/counts).mean()))
                estimated=1/(between[:,None]+noise[:,None]/counts)
                oracle=1/(tau*tau+1/counts[None,:])
                metrics={}
                for name,w in (("count",raw),("equal_origin_oracle_variance",equal),
                               ("estimated_cov",estimated),("oracle_cov",oracle)):
                    t=((w*means).sum(1)+y)/(w.sum(1)+anchors)
                    metrics[name]=float(np.sqrt(np.mean(t*t)))
                out[f"tau{tau}_bias{bias}_real{anchors}"]=metrics
    return out


if __name__=="__main__":
    output=dict(stationary=recursion(42001,False),shift=recursion(42002,True),origins=clustered(42003))
    (HERE/"revisit42.json").write_text(json.dumps(output,indent=2),encoding="utf8")
