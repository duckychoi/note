"""Scale information after allowing one free log-depth offset per visible RGB region."""
import argparse
import json

import cv2
import numpy as np
from PIL import Image

from repeat_capture_acquire import OUT as CAPTURES
from repeat_capture_render import BASELINE, FOCAL, LOG_MIN, VIEWS, geometry, load_gray, sample
from repeat_encoder import DESIGN, family
from scale_surface_rgb import CKPT, OUT, Reader, infer_all, nearest, read_heads, regions, resize_rgb, sha, write


def information(labels, derivative):
    # Normalize by sample count; area weights refer to the same measured support.
    _, index, count = np.unique(labels, return_inverse=True, return_counts=True)
    weights = count / len(labels)
    energy = np.bincount(index, weights=derivative.astype(float)**2) / len(labels)
    known = float(energy.sum())
    available = float(1. / np.sum(weights**2 / energy)) if (energy > 0).all() else 0.
    # Independent numerical elimination of the nuisance columns on sufficient statistics.
    anchor = np.argmax(weights)
    others = np.arange(len(weights)) != anchor
    basis = np.eye(len(weights))[:, others]
    basis[anchor] = -weights[others] / weights[anchor]
    def projected(e):
        column = np.sqrt(e)
        nuisance = column[:, None] * basis
        fit = np.linalg.lstsq(nuisance, column, rcond=None)[0]
        return float(np.sum((column - nuisance @ fit)**2))
    check = projected(energy)
    blind = energy.copy()
    blind[anchor] = 0
    hidden = projected(blind)
    if not np.isclose(available, check, rtol=1e-7, atol=1e-12) or hidden > 1e-10*max(known,1):
        raise ValueError("Independent projection disagrees with analytic identifiability")
    return dict(scale_if_shape_known=known,scale_with_unknown_shape=available,
                retained_fraction=available/known if known else 0.,
                blind_region_projected=hidden,regions=len(weights))


def check_bound(labels,derivative,obs,target,correction):
    _,region,counts = np.unique(labels,return_inverse=True,return_counts=True)
    w = counts/len(region)
    D,y,t,c = (np.asarray(a,float) for a in (derivative,obs,target,correction))
    C = np.bincount(region,weights=D*D)/len(region)
    tr = np.bincount(region,weights=t)/counts
    cr = np.bincount(region,weights=c)/counts
    residual_rms = np.sqrt(np.mean((y-D*tr[region])**2))
    actual = np.mean(2*t*c-c*c)
    result = dict(actual=float(actual),residual_rms=float(residual_rms))
    if not (C>0).all():
        return dict(**result,lower=None,scale_lower=None)
    observed = np.bincount(region,weights=D*y)/(len(region)*C)
    info = 1/np.sum(w*w/C)
    scale = np.sum(w*observed)
    mean_c = c.mean()
    scale_lower = 2*scale*mean_c-mean_c**2-2*abs(mean_c)*residual_rms/np.sqrt(info)
    within = np.linalg.norm(t-tr[region])*np.linalg.norm(c-cr[region])/len(region)
    lower = (2*np.sum(w*observed*cr)-np.mean(c*c)-
             2*residual_rms*np.sqrt(np.sum((w*cr)**2/C))-2*within)
    if lower>actual+1e-9 or scale_lower>2*t.mean()*mean_c-mean_c**2+1e-9:
        raise ValueError("Certified decrease exceeds actual decrease")
    return dict(**result,lower=float(lower),scale_lower=float(scale_lower))


def measure(src,reader=None,heads=None):
    x, disp, dmin, truth_inverse, _, _, mask = geometry(src)
    rgb = np.array(Image.open(src/"Illum1/Exp1/view1.png").convert("RGB"))
    labels = nearest(regions(resize_rgb(rgb)).astype(np.float32), x.shape).astype(int)
    with np.load(CAPTURES/"encoder"/(src.name+".npz")) as enc:
        inverse = cv2.resize(enc["inverse"], x.shape[::-1], interpolation=cv2.INTER_LINEAR)
    correction = None
    if reader is not None:
        outputs = infer_all(reader.extract(resize_rgb(rgb)),heads)
        baseline_inv = cv2.resize(1/outputs["base"],x.shape[::-1],interpolation=cv2.INTER_LINEAR)
        if not np.allclose(baseline_inv,inverse,rtol=5e-5,atol=2e-6):
            raise ValueError(f"{src.name}: independently computed baseline differs")
        joint_inv = cv2.resize(1/outputs["stereo_joint"],x.shape[::-1],interpolation=cv2.INTER_LINEAR)
        correction = np.log(baseline_inv/joint_inv)
        inverse = baseline_inv
    with np.load(CAPTURES/"renders-plane"/(src.name+".npz")) as data:
        valid, origin, means = data["valid"], data["source_x"], data["expected"]
    row = np.arange(x.shape[0])[:,None]
    coords = []
    fractions = []
    for v, fraction in enumerate(VIEWS):
        true = x-fraction*disp
        mask &= (sample(valid[v],true)>.999) & (np.abs(sample(origin[v],true)-x)<.6)
        pos = x-fraction*(FOCAL*BASELINE*inverse-dmin)
        left = np.floor(pos).astype(int)
        mask &= (left>=0)&(left+1<x.shape[1])
        left = np.clip(left,0,x.shape[1]-2)
        mask &= valid[v,row,left].astype(bool) & valid[v,row,left+1].astype(bool)
        coords.append(left)
        fractions.append(pos-left)
    if not mask.any():
        raise ValueError(f"No scoring support for {src.name}")
    physical = np.array([[load_gray(src/f"Illum{l}/Exp1/view{v}.png") for l in (1,2)]
                         for v in (3,5)])
    result = {}
    for name, images in zip(("physical","diffuse","glossy"),(physical,*means)):
        logs = np.log(np.clip(images,LOG_MIN,1.))
        contrast = logs[:,0]-logs[:,1]
        # Exact one-sided derivative of a piecewise-linear horizontal warp at c=0.
        parts = [fraction*FOCAL*BASELINE*inverse*(im[row,left+1]-im[row,left])
                 for fraction,im,left in zip(VIEWS,contrast,coords)]
        derivative = -(parts[1]-parts[0])
        result[name] = information(labels[mask],derivative[mask])
        if correction is not None:
            readings = [im[row,left]+f*(im[row,left+1]-im[row,left])
                        for im,left,f in zip(contrast,coords,fractions)]
            result[name]["student_bound"] = check_bound(labels[mask],derivative[mask],
                (readings[1]-readings[0])[mask],np.log(inverse[mask]/truth_inverse[mask]),correction[mask])
    return dict(group=family(src.name),pixels=int(mask.sum()),modes=result)


def main(student=False):
    cv2.setNumThreads(1)
    weight_sha = sha(CKPT)
    if json.loads((CAPTURES/"encoder/model.json").read_text())["sha256"] != weight_sha:
        raise ValueError("Stored monocular prediction uses a different checkpoint")
    excluded = set(map(family,DESIGN))
    reader,heads = (Reader(),read_heads()) if student else (None,None)
    rows = {src.name:measure(src,reader,heads) for src in sorted((CAPTURES/"physical").iterdir())
            if family(src.name) not in excluded}
    groups = sorted({r["group"] for r in rows.values()})
    summary = {}
    for mode in ("physical","diffuse","glossy"):
        summary[mode] = {k:float(np.median([
            np.mean([r["modes"][mode][k] for r in rows.values() if r["group"]==g])
            for g in groups])) for k in ("scale_if_shape_known","scale_with_unknown_shape","retained_fraction")}
        summary[mode]["positive_scenes"] = sum(r["modes"][mode]["scale_with_unknown_shape"]>0
                                              for r in rows.values())
        if student:
            checks = [r["modes"][mode]["student_bound"] for r in rows.values()]
            summary[mode]["log_actual_positive_scenes"] = sum(item["actual"]>0 for item in checks)
            for key in ("lower","scale_lower"):
                values = [item[key] for item in checks if item[key] is not None]
                summary[mode][key] = dict(evaluable_scenes=len(values),positive_scenes=sum(v>0 for v in values),
                                          scene_median=float(np.median(values)) if values else None)
    OUT.mkdir(exist_ok=True)
    output = "observation-student-bound.json" if student else "observability.json"
    write(OUT/output,dict(checkpoint=weight_sha,heads_sha256=sha(OUT/"weights.json") if student else None,
          scenes=len(rows),families=len(groups),
          pixels=sum(r["pixels"] for r in rows.values()),summary=summary,rows=rows))
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student",action="store_true")
    main(parser.parse_args().student)
