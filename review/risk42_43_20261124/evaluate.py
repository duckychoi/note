"""Real data enter fitting only through the two fixed, bounded calibration grids."""
import json
import cv2
import numpy as np
import torch

from acquire import HERE,DEV,TEST
from measured import PROCESSED
from fit import ARMS,SEEDS,CKPT,DepthNet,observe,med,stereo_arms,tensor_input,predict


def metrics(p,d):
    z=d["z"]
    v=d["valid"]
    whole=float(1000*np.mean(np.abs(z[v]-p[v]),dtype=np.float64)) if v.any() else None
    out=dict(whole=whole)
    triples=d["triples"]
    if len(triples)<8:
        return dict(out,jump=None,front=None,back=None,sign=None)
    f,b=triples[:,0,None],triples[:,1:]
    est,gt=p.ravel(),z.ravel()
    jump=est[b]-est[f]
    true_jump=gt[b]-gt[f]
    def mae(err):
        return float(1000*np.abs(err).mean(dtype=np.float64))
    out.update(jump=mae(jump-true_jump),front=mae(est[f]-gt[f]),back=mae(est[b]-gt[b]),
               sign=float(np.mean(jump>0)))
    return out


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    scenes={}
    views={}
    obs={}
    quality={}
    for name in DEV+TEST:
        with np.load(PROCESSED/(name+".npz")) as saved:
            d={k:saved[k] for k in saved.files}
        scenes[name]=d
        views[name]=tensor_input(d["im0"],d["focal"])[None].cuda()
        s=observe(d["im0"],d["im1"],d["focal"],d["baseline"],d["offset"],128)[0]
        data=dict(zip(("stereo","ok","grad","spread"),s))
        obs[name]=data
        if name in TEST:
            oracle=metrics(d["z"],d)
            assert all(oracle[k] in (0,None) for k in ("whole","jump"))
            q=data["ok"]&d["valid"]
            trips=d["triples"]
            quality[name]=dict(valid_stereo_fraction=float(q.sum()/max(1,d["valid"].sum())),
              matched_stereo_mae_mm=float(np.mean(np.abs(d["z"][q]-data["stereo"][q]))*1000) if q.any() else None,
              front_coverage=float(data["ok"].ravel()[trips[:,0]].mean()) if len(trips) else None,
              back_coverage=float(data["ok"].ravel()[trips[:,1:]].mean()) if len(trips) else None)
    zcal=np.concatenate([scenes[s]["z"] for s in DEV])
    few_constant=float(np.median(zcal))
    scores=[]
    calibrations={}
    for seed in SEEDS:
        ledger=json.loads((HERE/f"seed{seed}.json").read_text())
        assert len(set(ledger["batch_hashes"].values()))==1
        raw={name:{} for name in DEV+TEST}
        for arm in ARMS:
            net=DepthNet(False).cuda()
            net.load_state_dict(torch.load(CKPT/f"{seed}_{arm}.pt",weights_only=True))
            net.eval()
            for name in DEV+TEST:
                y=predict(net,views[name])[0]
                raw[name][arm]=y
                if arm in ("replay","utility"):
                    raw[name].update(stereo_arms(y,obs[name],np.array(ledger["gain"],np.float32),ledger["fixed_gain"],arm))
        real_scale={}
        for arm in raw[DEV[0]]:
            p=np.concatenate([raw[s][arm][scenes[s]["y"],scenes[s]["x"]] for s in DEV])
            assert np.all(np.isfinite(p)) and np.all(p>0) and len(p)==len(zcal)
            real_scale[arm]=med(zcal/p,p)
        calibrations[seed]=dict(real_scale=real_scale,real_constant=few_constant,successful=len(zcal))
        for budget in ("zero","few"):
            scales=ledger["scales"] if budget=="zero" else real_scale
            c=ledger["synth_constant"] if budget=="zero" else few_constant
            for name in TEST:
                d=scenes[name]
                pmap={k:v*scales[k] for k,v in raw[name].items()}
                pmap["constant"]=np.full_like(d["z"],c)
                pmap["replay_raw"]=raw[name]["replay"]
                for arm,p in pmap.items():
                    assert np.isfinite(p).all() and (p>0).all()
                    result=metrics(p,d)
                    if arm=="constant" and result["jump"] is not None:
                        tri=d["triples"]
                        jump=d["z"].ravel()[tri[:,1:]]-d["z"].ravel()[tri[:,0,None]]
                        assert abs(1000*jump.mean()-result["jump"])<.001
                    scores.append(dict(seed=seed,scene=name,budget=budget,arm=arm,**result))
        print("evaluated",seed,flush=True)
    for file,obj in (("scene_scores",scores),("calibration",calibrations),("stereo_quality",quality)):
        (HERE/(file+".json")).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    main()
