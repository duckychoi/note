"""Score the paired policies and fit a gravity-aware control using only camera pixels."""
import argparse
import json
import sys
from itertools import combinations

import numpy as np
from swing import BAND,CACHE,G,HERE,L,LEVELS,PROBE_SPEED,PROBE_TIME,R,SPEED,camera_angle,raster
sys.path.insert(0,str(HERE.parent/"fm38_materials_20261123"))
from analyse import bootstrap,interval

NAMES=("flow","mixture","retrieval")
MODES=("photo","probe")
DOMAINS=("source","shift")
EFFECTIVE_LENGTH=(L*L+.4*R*R)/L


def peak_angle(start,g,velocity):
    energy=np.cos(start)-EFFECTIVE_LENGTH*velocity**2/(2*g)
    return np.arccos(np.clip(energy,-1,1))


def practice_curve(start,gravity):
    """Integrate candidate responses of the known pendulum across a grid of gravity values."""
    q=np.broadcast_to(start,gravity.shape).copy()
    v=np.full(q.shape,PROBE_SPEED)
    step=.005
    def acceleration(x):
        return -(gravity/EFFECTIVE_LENGTH)*np.sin(x)
    for _ in range(round(PROBE_TIME/step)):
        k1=acceleration(q)
        k2=acceleration(q+step*v/2)
        k3=acceleration(q+step*(v+step*k1/2)/2)
        k4=acceleration(q+step*(v+step*k2/2))
        q+=step*(6*v+step*(k1+k2+k3))/6
        v+=step*(k1+2*k2+2*k3+k4)/6
    return q


def close_to_target(x,goal):
    return np.abs(x-goal[:,None]).argmin(1)


def physical_control(data,nominal_g):
    obs=data["angles"][::4,:2]
    q=np.array([camera_angle(raster(x)) for x in obs[:,0]])
    practice=np.array([camera_angle(raster(x)) for x in obs[:,1]])
    goal=data["goal"][::4]
    grid=np.linspace(.2*G,G,321)
    gravity=np.broadcast_to(grid,(len(q),len(grid)))
    curves=practice_curve(q[:,None],gravity)
    estimated=grid[np.abs(curves-practice[:,None]).argmin(1)]
    pred={}
    for mode,g in (("photo",nominal_g),("probe",estimated[:,None])):
        peaks=peak_angle(q[:,None],g,SPEED[None])
        pred[mode]=close_to_target(peaks,goal)
    return pred,estimated


def prepare():
    with np.load(CACHE/"train.npz") as tr:
        nominal=(G*np.sin(np.deg2rad(LEVELS["source"][tr["level"]]))).mean()
    with np.load(CACHE/"val.npz") as val:
        fixed=np.abs(val["peak"]-val["goal"]).reshape(-1,4)
        fixed_action=int((fixed<=BAND).mean(0).argmax())
    raw={"nominal_source_gravity":nominal,"fixed_action":fixed_action}
    for domain in DOMAINS:
        with np.load(CACHE/f"{domain}.npz") as d:
            assert np.array_equal(d["action"].reshape(-1,4),np.tile(np.arange(4),(len(d["action"])//4,1)))
            phys,estimate=physical_control(d,nominal)
            true_g=G*np.sin(np.deg2rad(LEVELS[domain][d["level"][::4]]))
            raw["g_"+domain]=true_g
            raw["estimated_g_"+domain]=estimate
            raw["angles_"+domain]=d["angles"][::4,:2]
            raw["peak_"+domain]=d["peak"].reshape(-1,4)
            raw["goal_"+domain]=d["goal"][::4]
            exact=peak_angle(d["angles"][:,0],np.repeat(true_g,4),SPEED[d["action"]])
            # A conservative, independently derived check of gravity, inertia and first-peak extraction.
            assert np.rad2deg(np.max(abs(exact-d["peak"])))<.01
            for mode in MODES:
                raw[f"physics_{mode}_{domain}"]=phys[mode]
    assert np.array_equal(raw["goal_source"],raw["goal_shift"])
    assert np.array_equal(raw["angles_source"][:,0],raw["angles_shift"][:,0])
    np.savez_compressed(HERE/"outcomes.npz",**raw)


def choices(p):
    prob,err=p[...,0],p[...,1]
    best=prob==prob.max(-1,keepdims=True)
    return np.where(best,err,np.inf).argmin(-1)


def rate(a):
    return float(np.mean(a))


def evaluate():
    with np.load(HERE/"outcomes.npz") as d,np.load(HERE/"forecasts.npz") as f:
        yes={dm:np.abs(d["peak_"+dm]-d["goal_"+dm][:,None])<=BAND for dm in DOMAINS}
        n=len(yes["source"])
        ix=np.arange(n)
        ai,aj=np.triu_indices(4,1)
        margin=[ys[:,ai].astype(int)-ys[:,aj] for ys in yes.values()]
        rev=margin[0]*margin[1]<0
        separate=~(yes["source"]&yes["shift"]).any(1)&yes["source"].any(1)&yes["shift"].any(1)
        example=485
        out={"n":n,"reversed_states":int(rev.any(1).sum()),"reversed_pairs":int(rev.sum()),
             "strict_reversal_state_CI95":interval(rev.any(1).sum(),n),"no_shared_success":int(separate.sum()),
             "example":{"id":example,"target_deg":float(np.rad2deg(d["goal_source"][example]))},
             "controls":{},"models":{},"inference":{}}
        values={}
        for dm,y in yes.items():
            p=d["peak_"+dm]
            raw=d["angles_"+dm]
            out["example"][dm]={"g":float(d["g_"+dm][example]),
                                "peak_deg":np.rad2deg(p[example]).tolist(),
                                "practice_deg":float(np.rad2deg(raw[example,1]))}
            out["controls"][dm]={"oracle":rate(y.any(1)),"random":rate(y),"fixed_action":1+int(d["fixed_action"]),
                                 "fixed_success":rate(y[:,int(d["fixed_action"])]),
                                 "success_by_action":y.mean(0).tolist(),
                                 "gravity_estimate_MAE":rate(abs(d["estimated_g_"+dm]-d["g_"+dm])),
                                 "physics":{m:rate(y[ix,d[f"physics_{m}_{dm}"]]) for m in MODES}}
        for name in NAMES:
            sel={}
            summary={}
            for mode in MODES:
                summary[mode]={}
                for dm,y in yes.items():
                    keys=sorted(k for k in f if k.startswith(name+"_") and k.endswith(f"_{mode}_{dm}"))
                    assert len(keys)==(1 if name=="retrieval" else 3)
                    a=np.stack([choices(f[k]) for k in keys])
                    assert all(np.isfinite(f[k]).all() for k in keys)
                    sel[mode,dm]=a
                    success=y[ix,a]
                    values[name,mode,dm]=success.mean(0)
                    summary[mode][dm]={"success":rate(success),"seed_rates":success.mean(1).tolist(),
                        "actual_mean_abs_error_deg":rate(np.rad2deg(abs(d["peak_"+dm][ix,a]-d["goal_"+dm]))),
                        "action_mix":[rate(a==i) for i in range(4)]}
                s,t=sel[mode,"source"],sel[mode,"shift"]
                summary[mode]["changed_action_rate"]=rate(s!=t)
                summary[mode]["harder_action_rate"]=rate(t>s)
                both= (yes["source"][ix,s]&yes["shift"][ix,t])
                summary[mode]["both_correct_on_no_shared_success"]=rate(both[:,separate])
                if mode=="photo":
                    assert np.array_equal(s,t)
            # Held-out mismatch check: use the probe observation from the paired weak-gravity twin.
            summary["wrong_gravity_probe_on_shift"]=rate(yes["shift"][ix,sel["probe","source"]])
            out["models"][name]=summary
        diff={}
        for name in NAMES:
            for dm in DOMAINS:
                diff[f"gain_{name}_{dm}"]=values[name,"probe",dm]-values[name,"photo",dm]
        for i,j in combinations(NAMES,2):
            for dm in DOMAINS:
                diff[f"{i}_minus_{j}_{dm}"]=values[i,"probe",dm]-values[j,"probe",dm]
        limits=bootstrap(np.column_stack(list(diff.values())),comparisons=12)
        for (key,row),ci in zip(diff.items(),limits):
            out["inference"][key]=[rate(row),*ci.tolist()]
        out["confirmed_method_reversals"]=[]
        for i,j in combinations(NAMES,2):
            lo,hi=[out["inference"][f"{i}_minus_{j}_{dm}"][1:] for dm in DOMAINS]
            if (lo[0]>0 and hi[1]<0) or (lo[1]<0 and hi[0]>0):
                out["confirmed_method_reversals"].append([i,j])
    (HERE/"results.json").write_text(json.dumps(out,indent=2),encoding="utf8")
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--from-archive",action="store_true")
    opts=parser.parse_args()
    if not opts.from_archive:
        prepare()
    if not opts.prepare_only:
        result=evaluate()
        for name,item in result["models"].items():
            print(name,[[round(item[m][dm]["success"],4) for dm in DOMAINS] for m in MODES])
        print(result["inference"])
