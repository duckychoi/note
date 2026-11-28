"""Exact and sampled partition functions for real multi-view depth candidate grids."""
import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import product
import json
from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import logsumexp

from pf42_crossview import entries, load, mix_evidence, soft_curve
from pf42_crossview_data import DEPTH, HELD, OUT, move, scene
from pf42_visibility import NEW, PARTITION


RUN = OUT / "a-complexity"
SHAPE = (26, 38)
LABELS = 4
WIDTHS = (3, 5, 7, 8)
PARTICLES = (128, 512, 2048, 8192)
SEEDS = (24101, 24119, 24137)


def write(path, payload):
    tmp=path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf8")
    tmp.replace(path)


def alternatives(obs):
    logl,support=soft_curve(obs["cost"],obs["visibility"],.20)
    score=sum(mix_evidence(logl[:,obs["groups"]==g].sum(1),.05) for g in np.unique(obs["groups"]))
    allowed=support>=2
    active=allowed.any(1)
    ceiling=np.where(allowed,score,-np.inf).max(1)
    ceiling=np.where(active,ceiling,0.)
    score=np.where(allowed,score-ceiling[:,None],-12.)
    score[~active]=0.
    ranked=score.copy()
    picks=[]
    axis=np.arange(len(DEPTH))
    for _ in range(LABELS):
        k=ranked.argmax(1)
        picks.append(k)
        ranked[abs(k[:,None]-axis[None,:])<12]=-np.inf
    picks=np.sort(np.column_stack(picks),axis=1)
    picks[~active]=np.linspace(0,len(DEPTH)-1,LABELS,dtype=int)
    return picks, np.take_along_axis(score,picks,axis=1)-np.log(LABELS), active


def compatibility(refs, dst, log_depths, rgb, pix, visible, zcam):
    d=np.abs(log_depths[refs,:,None]-log_depths[dst,None,:])
    color=np.linalg.norm(rgb[refs]-rgb[dst],axis=-1)
    near=np.exp(-color/.15)[:,None,None]
    logp=-.65*near*np.minimum(d/.20,1.)
    near_ray=np.linalg.norm(pix[refs,:,None]-pix[dst,None,:],axis=-1)<=1.5
    in_view=visible[refs,:,None]&visible[dst,None,:]
    za,zb=zcam[refs,:,None],zcam[dst,None,:]
    close=abs(za-zb)>.12*np.maximum(np.minimum(za,zb),.1)
    conflict=(near_ray&in_view&close).sum(-1)
    return logp-.8*np.minimum(conflict,2), conflict>0


def blocks(obs, reference, cameras):
    labels,unary,active=alternatives(obs)
    depth=DEPTH[labels]
    n=len(labels)
    pix=np.empty((n,LABELS,len(obs["sources"]),2),dtype=np.float32)
    zcam=np.empty(pix.shape[:-1],dtype=np.float32)
    visible=np.empty(zcam.shape,dtype=bool)
    xy=obs["tally"]
    ray=reference.rays(xy)
    for j,source in enumerate(obs["sources"]):
        src=cameras[str(source)]
        uv,z,inside=src.project(move(reference,src,ray[:,None],depth[:,:,None]))
        pix[:,:,j]=uv
        zcam[:,:,j]=z
        vi=np.take_along_axis(obs["visibility"][:,j],labels,axis=1)
        visible[:,:,j]=inside&(vi>=.5)
    x,y=xy.astype(int).T
    rgb=obs["rgb"][y,x].astype(np.float32)/255
    lattice=np.arange(n).reshape(SHAPE)

    def edges(a,b):
        rows,conflict=compatibility(a.ravel(),b.ravel(),np.log(depth),rgb,pix,visible,zcam)
        s=(*a.shape,LABELS,LABELS)
        return rows.reshape(s),conflict.reshape(s)

    h,hc=edges(lattice[:,:-1],lattice[:,1:])
    v,vc=edges(lattice[:-1,:],lattice[1:,:])
    return unary.reshape(*SHAPE,LABELS),depth.reshape(*SHAPE,LABELS),active.reshape(SHAPE),h,v,hc,vc


def windows(unary,active):
    w=max(WIDTHS)
    valid=sliding_window_view(active,(w,w)).mean(axis=(-1,-2))
    p=np.exp(unary-logsumexp(unary,axis=-1,keepdims=True))
    entropy=-(p*np.log(p)).sum(-1)
    quality=sliding_window_view(entropy,(w,w)).mean(axis=(-1,-2))
    rows=np.argwhere(valid>=.70)
    if not len(rows):
        return []
    ranked=sorted((tuple(map(int,r)) for r in rows),key=lambda r:quality[r])
    picked=[]
    for q in (.85,.50,.15):
        target=round(q*(len(ranked)-1))
        for i in sorted(range(len(ranked)),key=lambda j:abs(j-target)):
            candidate=ranked[i]
            if all(max(abs(u-v) for u,v in zip(candidate,c))>=w//2 for c in picked):
                picked.append(candidate)
                break
    return picked


def prepare():
    scenes=[("held",HELD),(PARTITION,NEW)]
    assembled=[]
    prep=0.
    for part,names in scenes:
        for name in names:
            paths=list(entries(part,[name]))
            chosen=[paths[i] for i in sorted({0,len(paths)-1})]
            cams={str(c.path):c for c in scene(name,with_truth=False)}
            for path in chosen:
                t=perf_counter()
                obs=load(path)
                unary,depth,active,h,v,hc,vc=blocks(obs,cams[str(obs["ref"])],cams)
                chosen_windows=windows(unary,active)
                prep+=perf_counter()-t
                labels=load(path.with_suffix(".truth.npz"))
                truth=labels["truth"].reshape(SHAPE)
                occ=(labels["occlusion"]==1).any(1).reshape(SHAPE)
                for i,(y,x) in enumerate(chosen_windows):
                    assembled.append(dict(scene=name,frame=path.stem,index=i,xy=(x,y),
                                          truth=truth,occlusion=occ,active=active,
                                          unary=unary,depth=depth,horiz=h,vert=v,hc=hc,vc=vc))
    return assembled,prep


@lru_cache(maxsize=8)
def configurations(width,k):
    return np.indices((k,)*width,dtype=np.int8).reshape(width,-1)


def across(frontier, factors, k, width):
    msg=frontier.reshape((k,)*width)
    for col in range(width):
        moved=np.moveaxis(msg,col,-1)
        msg=np.moveaxis(moved@factors[col],-1,col)
    return msg.ravel()


def exact(unary, horiz, vert):
    height,width,k=unary.shape
    conf=configurations(width,k)
    r=np.arange(width)
    rows=[]
    logz=0.
    for u,h in zip(unary,horiz,strict=True):
        l=u[r[:,None],conf].sum(0)+h[r[:-1,None],conf[:-1],conf[1:]].sum(0)
        m=float(l.max())
        rows.append(np.exp(l-m)); logz+=m
    edges=np.exp(vert)
    front=[]
    for i,x in enumerate(rows):
        z=x if i==0 else x*across(front[-1],edges[i-1],k,width)
        den=z.sum()
        logz+=np.log(den)
        front.append(z/den)
    posterior=np.empty_like(unary)
    backward=np.ones_like(rows[-1])
    for row in reversed(range(height)):
        all_states=front[row]*backward
        all_states/=all_states.sum()
        tensor=all_states.reshape((k,)*width)
        for col in range(width):
            posterior[row,col]=tensor.sum(axis=tuple(j for j in range(width) if j!=col))
        if row:
            backward=across(rows[row]*backward,edges[row-1].swapaxes(-1,-2),k,width)
            backward/=backward.sum()
    return logz,posterior


def particle(unary,horiz,vert,count,seed):
    height,width,k=unary.shape
    u,h,v=np.exp(unary),np.exp(horiz),np.exp(vert)
    identity=np.arange(count,dtype=np.int32)
    rng=np.random.default_rng(seed)
    frontier=np.zeros((count,width),dtype=np.int8)
    w=np.full(count,1./count)
    labels=np.empty((height*width,count),dtype=np.int8)
    parent=np.empty((height*width,count),dtype=np.int32)
    logz,resampling=0.,0
    for i in range(height):
        for j in range(width):
            p=np.broadcast_to(u[i,j],(count,k)).copy()
            if j:
                p*=h[i,j-1,frontier[:,j-1]]
            if i:
                p*=v[i-1,j,frontier[:,j]]
            cdf=np.cumsum(p,axis=1)
            w*=cdf[:,-1]
            norm=w.sum()
            logz+=np.log(norm); w/=norm
            anc=identity
            if 1./(w@w)<count/2:
                cs=np.cumsum(w); cs[-1]=1.
                anc=np.searchsorted(cs,(identity+rng.random())/count).astype(np.int32)
                frontier=frontier[anc]
                cdf=cdf[anc]
                w.fill(1./count)
                resampling+=1
            y=(rng.random(count)[:,None]*cdf[:,-1,None]>cdf[:,:-1]).sum(1)
            frontier[:,j]=y
            t=i*width+j
            labels[t]=y
            parent[t]=anc
    marginal=np.empty((height*width,k))
    for t in reversed(range(height*width)):
        marginal[t]=np.bincount(labels[t],weights=w,minlength=k)
        if t:
            w=np.bincount(parent[t],weights=w,minlength=count)
    return logz,marginal.reshape(height,width,k),int((w>0).sum()),resampling


def belief_propagation(unary,horiz,vert,steps=40,damped=True):
    height,width,k=unary.shape
    shape=np.arange(height*width).reshape(height,width)
    a=np.concatenate([shape[:,:-1].ravel(),shape[:-1,:].ravel()])
    b=np.concatenate([shape[:,1:].ravel(),shape[1:,:].ravel()])
    pair=np.concatenate([horiz.reshape(-1,k,k),vert.reshape(-1,k,k)])
    p=np.concatenate([pair,pair.swapaxes(1,2)])
    src=np.concatenate([a,b]); dst=np.concatenate([b,a])
    rev=(np.arange(2*len(a))+len(a))%(2*len(a))
    logu=unary.reshape(-1,k)
    msg=np.full((len(src),k),-np.log(k))
    for step in range(steps):
        total=logu.copy(); np.add.at(total,dst,msg)
        update=logsumexp((total[src]-msg[rev])[:,:,None]+p,axis=1)
        update-=logsumexp(update,axis=1,keepdims=True)
        new=np.logaddexp(msg,update)-np.log(2.) if damped else update
        diff=np.max(np.abs(new-msg)); msg=new
        if diff<1e-7:
            break
    total=logu.copy(); np.add.at(total,dst,msg)
    logb=total-logsumexp(total,axis=1,keepdims=True)
    cavity=total[src]-msg[rev]
    joint=cavity[:len(a),:,None]+cavity[len(a):,None,:]+pair
    joint-=logsumexp(joint,axis=(1,2),keepdims=True)
    logz=(np.exp(joint)*(pair-joint)).sum()+(np.exp(logb)*(logu+(np.bincount(src,minlength=len(logu))[:,None]-1)*logb)).sum()
    return logz,np.exp(logb).reshape(unary.shape),step+1,bool(diff<1e-7)


def evaluate(model,depth,truth,valid):
    sorted_labels=depth.argsort(-1)
    values=np.take_along_axis(depth,sorted_labels,axis=-1)
    w=np.take_along_axis(model,sorted_labels,axis=-1)/values
    index=(w.cumsum(-1)<w.sum(-1,keepdims=True)*.5).sum(-1)
    selected=np.take_along_axis(values,index[...,None],axis=-1)[...,0]
    if not valid.any():
        return 0.,0
    return float(abs(selected[valid]/truth[valid]-1).sum()),int(valid.sum())


def summary(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row["width"],row["solver"])].append(row)
    output={}
    for (w,solver),items in sorted(grouped.items()):
        total=sum(i["n"] for i in items)
        report=dict(ms=float(np.median([i["ms"] for i in items])),
                    grad_mean=float(np.mean([i["grad_l1"] for i in items])),
                    grad_pass=float(np.mean([i["grad_l1"]<=.05 for i in items])),
                    logz_per_pixel=float(np.mean([i["logz_per_pixel"] for i in items])),
                    absrel=sum(i["absolute_sum"] for i in items)/total if total else None,
                    runs=len(items))
        if "ancestors" in items[0]:
            report["median_ancestors"]=float(np.median([i["ancestors"] for i in items]))
            report["median_resampled"]=float(np.median([i["resamples"] for i in items]))
        if "converged" in items[0]:
            report["converged"]=sum(i["converged"] for i in items)
        output.setdefault(str(w),{})[solver]=report
    for w,solvers in output.items():
        base=solvers["exact"]
        for solver,v in solvers.items():
            v["speedup"]=base["ms"]/v["ms"]
            v["pass"]=bool(v["speedup"]>=2 and v["grad_pass"]>=.90 and v["logz_per_pixel"]<=.01
                           and v["absrel"] is not None and v["absrel"]<=base["absrel"]*1.01)
    return output


def main(fast_bp=False):
    RUN.mkdir(exist_ok=True)
    patches,prep=prepare()
    print("patches",len(patches),"prepare seconds",prep,flush=True)
    rows,diagnostics=[],[]
    for w in (WIDTHS[-2:] if fast_bp else WIDTHS):
        configurations(w,LABELS)
        for i,patch in enumerate(patches):
            x,y=patch["xy"]
            off=(max(WIDTHS)-w)//2
            y+=off; x+=off
            c=(slice(y,y+w),slice(x,x+w))
            hc=(slice(y,y+w),slice(x,x+w-1)); vc=(slice(y,y+w-1),slice(x,x+w))
            u=patch["unary"][c].astype(np.float64)
            d,gt=patch["depth"][c],patch["truth"][c]
            h=patch["horiz"][hc].astype(np.float64); v=patch["vert"][vc].astype(np.float64)
            valid=np.isfinite(gt)&(gt>=DEPTH[0])&(gt<=DEPTH[-1])&patch["active"][c]
            recall=((abs(np.log(d/np.where(valid,gt,1.)[:,:,None]))<=.10).any(-1)&valid).sum()
            diagnostics.append(dict(width=w,scene=patch["scene"],frame=patch["frame"],xy=[x,y],
                                    n=int(valid.sum()),recall=int(recall),
                                    occluded=int((patch["occlusion"][c]&valid).sum()),
                                    conflicts=int(patch["hc"][hc].sum()+patch["vc"][vc].sum())))
            def save(solver,wall,z,p,**more):
                loss,n=evaluate(p,d,gt,valid)
                rows.append(dict(width=w,patch=i,solver=solver,ms=1000*wall,
                                 grad_l1=float(abs(p-ep).sum(-1).mean()),
                                 logz_per_pixel=float(abs(z-ez)/w**2),absolute_sum=loss,n=n,**more))

            t=perf_counter(); ez,ep=exact(u,h,v); first=perf_counter()-t
            t=perf_counter(); exact(u,h,v); second=perf_counter()-t
            save("exact",.5*(first+second),ez,ep)
            t=perf_counter(); bz=logsumexp(u,axis=-1,keepdims=True); independent=np.exp(u-bz)
            save("independent",perf_counter()-t,bz.sum(),independent)
            t=perf_counter(); bz,p,it,converged=belief_propagation(u,h,v)
            save("bp",perf_counter()-t,bz,p,iterations=it,converged=converged)
            if fast_bp:
                for steps in (4,8):
                    t=perf_counter(); bz,p,it,converged=belief_propagation(u,h,v,steps,False)
                    save(f"bp{steps}",perf_counter()-t,bz,p,iterations=it,converged=converged)
            for size in PARTICLES:
                for seed in SEEDS:
                    t=perf_counter(); z,p,n,resampling=particle(u,h,v,size,seed+997*i)
                    save(str(size),perf_counter()-t,z,p,seed=seed,ancestors=n,resamples=resampling)
        result=summary(rows)
        name="fast-bp.json" if fast_bp else "results.json"
        write(RUN/name,dict(rows=rows,diagnostics=diagnostics,summary=result,input_preparation_s=prep))
        print("width",w,result[str(w)],flush=True)


def verify():
    rng=np.random.default_rng(51)
    u=rng.uniform(-1,1,(2,3,3)); h=rng.uniform(-1,0,(2,2,3,3)); v=rng.uniform(-1,0,(1,3,3,3))
    s=list(product(range(3),repeat=6)); a=np.array(s).reshape(-1,2,3)
    scores=(u[np.arange(2)[None,:,None],np.arange(3)[None,None,:],a].sum((1,2))
            +h[np.arange(2)[None,:,None],np.arange(2)[None,None,:],a[:,:,:-1],a[:,:,1:]].sum((1,2))
            +v[0,np.arange(3)[None,:],a[:,0],a[:,1]].sum(1))
    z=logsumexp(scores); p=np.exp(scores-z)
    expected=np.stack([(p[a[:,i,j]==d]).sum() for i,j,d in product(range(2),range(3),range(3))]).reshape(2,3,3)
    got,m=exact(u,h,v)
    np.testing.assert_allclose(got,z,atol=1e-12); np.testing.assert_allclose(m,expected,atol=1e-12)
    t,b,_,_=belief_propagation(u,h,v*0)
    c,q=exact(u,h,v*0)
    np.testing.assert_allclose(t,c,atol=1e-5); np.testing.assert_allclose(b,q,atol=1e-5)
    pp=particle(u,h*0,v*0,8192,51)
    independent=exact(u,np.zeros_like(h),np.zeros_like(v))
    np.testing.assert_allclose(pp[0],independent[0],atol=1e-12)
    np.testing.assert_allclose(pp[1],independent[1],atol=.02)
    print("verified exact vs full 729 states, BP on trees, and particle independent logZ",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fast-bp",action="store_true")
    opts=parser.parse_args()
    verify()
    main(opts.fast_bp)
