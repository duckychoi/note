"""Recompute every gate decision from the held-out engine outcomes and saved trajectory samples."""
import json
import sys
import numpy as np
from world import CACHE,HERE,INERTIA
from swing import BAND,G,LEVELS,L,SPEED,TIMES
from controls import trajectory
from evaluate import choices,bootstrap

SEEDS=(4821,4822,4823)
ALPHAS=(0.,.125,.25,.5,1.)
SPLITS=("val","clean","seen","strong")


def score(paths,truth,peak,goals):
    if paths.ndim==2:
        paths=paths[:,None,:]
    path_peaks=paths.max(-1)
    dist=abs(path_peaks-goals[:,None])
    p=np.stack([(dist<=BAND).mean(1),dist.mean(1)],axis=-1).reshape(-1,4,2)
    a=choices(p)
    y=(abs(peak-goals)<=BAND).reshape(-1,4)
    ix=np.arange(len(a))
    absolute=np.rad2deg(abs(paths.mean(1)-truth))
    return {"success":y[ix,a].astype(float),
            "mae":absolute.reshape(-1,4,5).mean((1,2)),
            "peak_mae":np.rad2deg(abs(path_peaks.mean(1)-peak)).reshape(-1,4).mean(1),
            "chosen":a,"reachable":y.any(1)}


def summarize(data):
    reachable=data["reachable"].astype(bool)
    return {"success":float(data["success"].mean()),"trajectory_MAE_deg":float(data["mae"].mean()),
            "peak_MAE_deg":float(data["peak_mae"].mean()),"when_reachable":float(data["success"][reachable].mean())}


def run():
    scaled={}
    rows={s:{} for s in SPLITS}
    result={"controls":{},"physical_checks":{},"validation_choices":{},"metrics":{},
            "seed_performance":{},"contrasts":{}}
    with np.load(HERE/"cases.npz") as f:
        for s in ("seen","strong"):
            for k in ("state","action","goal","level"):
                assert np.array_equal(f[f"{s}_{k}"],f[f"clean_{k}"])
            assert np.array_equal(f[f"{s}_angles"][:,0],f["clean_angles"][:,0])
        for s in SPLITS:
            truth=f[f"{s}_angles"][:,2:]
            peak=f[f"{s}_peak"]
            goal=f[f"{s}_goal"]
            a=f[f"{s}_action"]
            strength=G*np.sin(np.deg2rad(LEVELS["source"][f[f"{s}_level"]]))*L*.2/INERTIA
            exact=trajectory(f[f"{s}_angles"][:,0],SPEED[a],strength,f[f"{s}_damping"],TIMES)
            max_err=float(np.rad2deg(abs(exact-truth)).max())
            assert max_err<.002,(s,max_err)
            result["physical_checks"][s]={"known_damped_equation_max_deg":max_err}
            y=(abs(peak-goal)<=BAND).reshape(-1,4)
            result["controls"][s]={"n_states":len(y),"oracle_success":float(y.any(1).mean()),
                                   "random_success":float(y.mean()),"reachable":int(y.any(1).sum()),
                                   "per_action_success":y.mean(0).tolist()}
            for name in ("prior","drag","ridge"):
                rows[s][name]=score(f[f"{s}_{name}"],truth,peak,goal)
            result["seed_performance"][s]={}
        for seed in SEEDS:
            with np.load(HERE/f"flow_{seed}.npz") as samples:
                for s in SPLITS:
                    base=f[f"{s}_prior"][:,None,:]
                    delta=samples[s]-base
                    assert np.isfinite(delta).all()
                    truth=f[f"{s}_angles"][:,2:]
                    def at(alpha):
                        if alpha==0:
                            return rows[s]["prior"]
                        return score(base+alpha*delta,truth,f[f"{s}_peak"],f[f"{s}_goal"])
                    if s=="val":
                        table={alpha:at(alpha) for alpha in ALPHAS}
                        best=max(ALPHAS,key=lambda alpha:(table[alpha]["success"].mean(),
                                                          -table[alpha]["mae"].mean(),-alpha))
                        result["validation_choices"][str(seed)]={"alpha":best,
                            "sweep":{str(a):summarize(row) for a,row in table.items()}}
                    scaled[s,"flow",seed]=at(best)
                    scaled[s,"full",seed]=at(1.)
                    result["seed_performance"][s][str(seed)]={n:summarize(scaled[s,n,seed]) for n in ("flow","full")}
        for s in SPLITS:
            for name in ("flow","full"):
                rows[s][name]={k:np.stack([scaled[s,name,j][k] for j in SEEDS]).mean(0)
                               for k in ("success","mae","peak_mae","reachable")}
            result["metrics"][s]={name:summarize(row) for name,row in rows[s].items()}
        # Strict real-action reversals at identical starting poses, goal and gravity.
        goal=f["strong_goal"]
        ys=[(abs(f[f"{s}_peak"]-goal)<=BAND).reshape(-1,4).astype(int) for s in ("seen","strong")]
        i,j=np.triu_indices(4,1)
        flip=(ys[0][:,i]-ys[0][:,j])*(ys[1][:,i]-ys[1][:,j])<0
        result["real_action_reversal_states"]=int(flip.any(1).sum())

    diffs={}
    for s,baselines in (("seen",("prior",)),("strong",("prior","drag","ridge"))):
        for b in baselines:
            diffs[f"{s}_flow_over_{b}_mae"]=rows[s][b]["mae"]-rows[s]["flow"]["mae"]
            diffs[f"{s}_flow_over_{b}_success"]=rows[s]["flow"]["success"]-rows[s][b]["success"]
    assert len(diffs)==8
    intervals=bootstrap(np.column_stack(list(diffs.values())),comparisons=8)
    for (k,d),ci in zip(diffs.items(),intervals):
        result["contrasts"][k]=[float(d.mean()),*ci.tolist()]
    mismatch=rows["strong"]["prior"]["mae"].mean()>rows["clean"]["prior"]["mae"].mean()
    def lower(b,metric):
        return result["contrasts"][f"strong_flow_over_{b}_{metric}"][1]
    over_prior=(result["contrasts"]["strong_flow_over_prior_success"][0]>=.03
                and lower("prior","success")>0 and lower("prior","mae")>0)
    result["gate"]={"misspecification_observed":bool(mismatch),"passes":bool(mismatch and over_prior),
                    "above_both_simple":bool(all(lower(b,m)>0 for b in ("drag","ridge") for m in ("success","mae")))}
    return result


if __name__=="__main__":
    r=run()
    (HERE/"results.json").write_text(json.dumps(r,indent=2),encoding="utf8")
    for s in ("clean","seen","strong"):
        print(s,{k:[round(x["success"],4),round(x["trajectory_MAE_deg"],2)] for k,x in r["metrics"][s].items()})
    print("alpha",[v["alpha"] for v in r["validation_choices"].values()])
    print(r["gate"],r["contrasts"])
