"""Frozen real-image predictions, scene scores and sign-preserving amplitude controls."""
import json
import cv2
import numpy as np
import torch

from common import ROOT,STORE,SEEDS,ARMS,DEV,TEST,save
from fit import DepthNet,tensor_input,predict,med


def load_view(s,wide=False):
    with np.load(STORE/(s+("_wide" if wide else "")+".npz")) as f:
        return {k:f[k] for k in f.files}


def grid_amplitude(p):
    parts=(np.abs(p[:,4:]-p[:,:-4]),np.abs(p[4:]-p[:-4]))
    return sum(x.sum() for x in parts)/sum(x.size for x in parts)


def affine(p,ratio):
    center=np.median(p)
    shifted=center+ratio*(p-center)
    return shifted+max(0.,1e-8-shifted.min())


def metrics(p,d):
    z,v=d["z"],d["valid"]
    whole=float(abs(p[v]-z[v]).mean()*1000) if v.any() else None
    tri=d["triples"]
    if len(tri)<8:
        return dict(whole=whole,jump=None,order=None,strict=None,tie=None,amplitude=None)
    b,f=tri[:,1:],tri[:,0,None]
    h=p.ravel()[b]-p.ravel()[f]
    t=z.ravel()[b].astype(float)-z.ravel()[f]
    return dict(whole=whole,**jump_metrics(h,t))


def jump_metrics(h,t):
    return dict(jump=float(1000*abs(h-t).mean()),order=float(((h>0)+.5*(h==0)).mean()),
                strict=float((h>0).mean()),tie=float((h==0).mean()),amplitude=float(1000*abs(h).mean()))


def ratio(num,den):
    if den>0:
        return float(num/den)
    return 1. if num==0 else None


def methods(maps,d):
    old,new=maps["legacy"],maps["lines_order"]
    tri=d["triples"]
    grid_ratio=ratio(grid_amplitude(new),grid_amplitude(old))
    r=min(1,grid_ratio) if grid_ratio is not None else 1.
    grid=affine(old,r)
    summaries={k:metrics(v,d) for k,v in maps.items()}
    summaries["grid_shrink"]=metrics(grid,d)
    rec=dict(grid_ratio=grid_ratio,exact_ratio=None,eligible=len(tri)>=8,impossible=False)
    if len(tri)>=8:
        f,b=tri[:,0,None],tri[:,1:]
        orig=old.ravel()[b]-old.ravel()[f]
        cand=new.ravel()[b]-new.ravel()[f]
        target=d["z"].ravel()[b].astype(float)-d["z"].ravel()[f]
        assert np.all(target>0)
        ro=ratio(abs(cand).mean(),abs(orig).mean())
        rec["exact_ratio"]=ro
        rec["impossible"]=ro is None
        base=affine(old,ro) if ro is not None else old
        if ro is not None:
            matched=base.ravel()[b]-base.ravel()[f]
            assert abs(abs(matched).mean()-abs(cand).mean())<1e-9
            assert ro==0 or np.array_equal(np.sign(matched),np.sign(orig))
        small_old=affine(old,min(1,ro)) if ro is not None else old
        inv=ratio(abs(orig).mean(),abs(cand).mean())
        small_new=affine(new,min(1,inv)) if inv is not None else new
        assert abs(metrics(small_old,d)["amplitude"]-metrics(small_new,d)["amplitude"])<1e-6
        common=orig!=0
        rec["old_ties"]=int((~common).sum())
        rec["pair_count"]=int(orig.size)
        synthetic=abs(cand)*np.sign(orig)
        summaries["pair_exact_sign"]=dict(whole=None,**jump_metrics(synthetic,target))
        rec["pair_exact_non_ties"]=jump_metrics(synthetic[common],target[common]) if common.any() else None
    else:
        base=small_old=grid
        small_new=new
        summaries["pair_exact_sign"]=dict.fromkeys(summaries["legacy"])
    for k,p in (("exact_old",base),("small_old",small_old),("small_new",small_new)):
        summaries[k]=metrics(p,d)
    return summaries,rec


def main():
    torch.set_num_threads(2)
    real={s:load_view(s) for s in DEV+TEST}
    wide={s:load_view(s,True) for s in TEST}
    x={s:tensor_input(d["rgb"],d["fx"])[None].cuda() for s,d in real.items()}
    zcal=np.concatenate([real[s]["z"] for s in DEV])
    assert len(zcal)>0
    rows=[]
    cases=[]
    cals={}
    dest=STORE/"predictions"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        ledger=json.loads((ROOT/f"train_{seed}.json").read_text())
        pred={}
        scal={}
        for arm in ARMS:
            net=DepthNet(False).cuda()
            net.load_state_dict(torch.load(STORE/"models"/f"{seed}_{arm}.pt",weights_only=True))
            net.eval()
            pred[arm]={s:predict(net,x[s])[0] for s in DEV+TEST}
            p=np.concatenate([pred[arm][s][real[s]["y"],real[s]["x"]] for s in DEV])
            scal[arm]=med(zcal/p,p)
        cals[seed]=dict(few_scale=scal,few_constant=float(np.median(zcal)),valid=len(zcal))
        for s in TEST:
            np.savez_compressed(dest/f"{seed}_{s}.npz",**{a:pred[a][s] for a in ARMS})
            for budget in ("zero","few"):
                scales=ledger["scales"] if budget=="zero" else scal
                maps={a:pred[a][s].astype(float)*scales[a] for a in ARMS}
                maps["constant"]=np.full_like(maps["legacy"],ledger["constant"] if budget=="zero" else np.median(zcal))
                for testset,d in (("strict",real[s]),("wide",wide[s])):
                    summary,record=methods(maps,d)
                    meta=dict(seed=seed,scene=s,budget=budget,subset=testset)
                    cases.append(dict(**meta,**record))
                    rows.extend(dict(**meta,arm=arm,**values) for arm,values in summary.items())
        print("scored",seed,flush=True)
    save("calibration.json",cals)
    save("amplitude_checks.json",cases)
    save("scores.json",rows)


if __name__=="__main__":
    main()
