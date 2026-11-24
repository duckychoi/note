"""Calibration action diagnostic on all additional 2014 scenes, with cached quarter views."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from stereo_gate import DATA, OUT, absrel_calculator, match, read_pfm

SCENES = ("Backpack","Bicycle1","Cable","Classroom1","Couch","Flowers","Mask",
          "Shopvac","Sticks","Storage","Sword1","Sword2","Umbrella")
SCALES = (.8,.95,1.,1.05,1.2)
RULES = ("unchanged","always","fixed3","shrink","dispersion")
CACHE = DATA / "additional2014"


def retrieve(scene):
    outfile = CACHE / (scene+".npz")
    if outfile.is_file():
        return
    url = f"https://vision.middlebury.edu/stereo/data/scenes2014/datasets/{scene}-perfect/"
    hashes, fields = {}, {}
    for fname,key in (("calib.txt","calib"),("im0.png","left"),("im1.png","right"),("disp0.pfm","gt")):
        for attempt in range(4):
            try:
                response = requests.get(url+fname,timeout=(15,75))
                response.raise_for_status()
                raw = response.content
                if key == "calib":
                    fields[key] = raw.decode()
                elif key == "gt":
                    fields[key] = read_pfm(raw)[2::4,2::4].copy()/4
                else:
                    img = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
                    fields[key] = cv2.GaussianBlur(img,(7,7),1.2)[2::4,2::4].copy()
                hashes[fname] = hashlib.sha256(raw).hexdigest()
                break
            except (requests.RequestException,ValueError,AssertionError,cv2.error):
                if attempt == 3:
                    raise
    assert fields["left"].shape == fields["gt"].shape == fields["right"].shape
    np.savez_compressed(outfile, **fields, provenance=json.dumps(hashes))


def scene_result(scene, index):
    with np.load(CACHE / (scene+".npz"),allow_pickle=False) as archive:
        left,right,gt = [archive[k] for k in ("left","right","gt")]
        calib = str(archive["calib"])
        hashes = json.loads(str(archive["provenance"]))
    offset = float(re.search(r"doffs=([\d.\-]+)",calib)[1])/4
    nd = int(np.ceil(float(re.search(r"ndisp=(\d+)",calib)[1])/64)*16)
    d = match(left,right,nd)
    dr = match(np.ascontiguousarray(right[:,::-1]),np.ascontiguousarray(left[:,::-1]),nd)[:,::-1]
    y,x = np.indices(d.shape)
    height,width = d.shape
    xr = np.rint(x-d).astype(int)
    support = (d>=0)&(d+offset>0)&(xr>=0)&(xr<width)
    rev = dr[y,np.clip(xr,0,width-1)]
    consistency = np.where(rev>=0,np.abs(d-rev),np.inf)
    texture = np.hypot(cv2.Sobel(left,cv2.CV_32F,1,0),cv2.Sobel(left,cv2.CV_32F,0,1))
    lattice = (x%4==0)&(y%4==0)
    valid = np.isfinite(gt)&(gt>=0)&(gt+offset>0)
    rated = valid&~lattice&support
    # Z_est/Z_true: the shared calibration numerator cancels exactly.
    ratio = np.divide(gt.astype(float)+offset,d+offset,out=np.zeros(d.shape),where=valid&support)
    score = absrel_calculator(ratio[rated])
    pools = []
    for j in range(4):
        pool = np.flatnonzero(lattice&support&((x>=width//2)==(j%2))&((y>=height//2)==(j//2)))
        order = np.lexsort((-texture.flat[pool],consistency.flat[pool]))
        pools.append(pool[order[:max(1,int(np.ceil(len(pool)*.1)))]])
    scores = {str(scale):{r:[] for r in RULES} for scale in SCALES}
    counts = []
    for seed in range(12):
        rng = np.random.default_rng([8132,index,seed])
        base = []
        for pool in pools:
            pick,noise = rng.random(),rng.normal(0,.02)
            if len(pool):
                p = pool[int(pick*len(pool))]
                if valid.flat[p]:
                    base.append(noise-np.log(ratio.flat[p]))
        counts.append(len(base))
        for scale in SCALES:
            m,s = 0.,1.
            if base:
                residual = np.array(base)-np.log(scale)
                m = float(np.median(residual))
                s = 1.2533*max(.02,1.4826*np.median(np.abs(residual-m)))/np.sqrt(len(base))
            adjustment = (0.,m,m if abs(m)>.03 else 0.,m*.01/(.01+s*s),m if abs(m)>1.96*s else 0.)
            for rule,delta in zip(RULES,adjustment):
                scores[str(scale)][rule].append(float(score(scale*np.exp(delta))))
    return dict(scores=scores, coverage=float(rated.sum()/(valid&~lattice).sum()),valid_queries=counts,sha256=hashes)


def main():
    cv2.setNumThreads(1)
    CACHE.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(retrieve,SCENES))
    scenes = {s:scene_result(s,i) for i,s in enumerate(SCENES)}
    boot = np.random.default_rng(1333).integers(len(scenes),size=(10000,len(scenes)))
    scores = {r:np.array([[np.mean(scene["scores"][str(scale)][r]) for scale in SCALES]
                         for scene in scenes.values()]) for r in RULES}
    means = {r:dict(zip(map(str,SCALES),a.mean(0).tolist())) for r,a in scores.items()}
    contrasts = {}
    for rival in ("fixed3","shrink"):
        delta = (scores[rival]-scores["dispersion"]).mean(1)
        contrasts[rival] = dict(relative=float(delta.mean()/scores[rival].mean()),
                                ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist())
    output = dict(scenes=scenes, means=means, contrasts=contrasts,
                  mean_coverage=float(np.mean([s["coverage"] for s in scenes.values()])))
    (OUT/"stereo_followup_results.json").write_text(json.dumps(output,indent=1),encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k!="scenes"},indent=2))


if __name__ == "__main__":
    main()
