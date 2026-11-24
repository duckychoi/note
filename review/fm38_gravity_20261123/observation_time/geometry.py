"""Finite-grid best possible action for each distinct observed RGB, with known uniform world prior."""
import hashlib
import json
import numpy as np
from simulate import HERE,SNAP
from swing import BAND,raster


def image_id(q):
    return hashlib.sha256(raster(q).tobytes()).digest()


def bound(starts,shots,peaks):
    goals=np.deg2rad(np.arange(32,92))
    scores=np.zeros(len(SNAP),int)
    static=oracle=total=0
    for start in np.unique(starts):
        ix=starts==start
        wins=abs(peaks[ix,None,:]-goals[None,:,None])<=BAND
        static+=wins.sum(0).max(1).sum()
        oracle+=wins.any(-1).sum()
        total+=wins.shape[0]*wins.shape[1]
        for j,angles in enumerate(shots[ix].T):
            buckets={}
            for row,q in enumerate(angles):
                buckets.setdefault(image_id(q),[]).append(row)
            scores[j]+=sum(wins[rows].sum(0).max(1).sum() for rows in buckets.values())
    assert np.all(scores>=static) and np.all(scores<=oracle)
    return {"scores":scores.tolist(),"denominator":int(total),"no_photo":int(static),"oracle":int(oracle)}


if __name__=="__main__":
    parts={}
    for d in ("seen","strong"):
        with np.load(HERE/f"grid_{d}.npz") as f:
            parts[d]=[f[k] for k in ("start","probe","peak")]
    results={d:bound(*data) for d,data in parts.items()}
    results["mixed"]=bound(*(np.concatenate(pair) for pair in zip(*parts.values())))
    f=np.load(HERE/"grid_seen.npz")
    g=np.load(HERE/"grid_strong.npz")
    weak=(f["start"]==0)&(f["level"]==6)&np.isclose(f["damping"],1.4)
    stiff=(g["start"]==0)&(g["level"]==1)&np.isclose(g["damping"],2.6)
    assert weak.sum()==stiff.sum()==1
    snaps=[f["probe"][weak][0],g["probe"][stiff][0]]
    result={"angles_deg":np.rad2deg(snaps).tolist(),"same_image":[image_id(x)==image_id(y) for x,y in zip(*snaps)],
            "winning_actions":[(np.flatnonzero(abs(np.rad2deg(v)-54)<=7.5)+1).tolist()
                               for v in (f["peak"][weak][0],g["peak"][stiff][0])]}
    results["earlier_example"]=result
    (HERE/"grid_bound.json").write_text(json.dumps(results,indent=2),encoding="utf8")
