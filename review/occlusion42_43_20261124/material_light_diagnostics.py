"""Mechanism checks after the locked test. No real labels choose a gain or a scale."""
import json

import numpy as np
from PIL import Image
import torch

from light_encoder_score import middlebury_pairs
from depth_full_data import hypersim_intrinsics
from light_encoder_study import Model, sha, yield_memory
from material_light_assets import OUT, save_json
from material_light_depth import depth
from material_light_test import NAMES, candidate, image_features, pairs, render_data, scenes, weighted_median


@torch.inference_mode()
def focal(model,rgb):
    yield_memory()
    return float(model.net.fov.focal_length_px(model.transform(rgb).unsqueeze(0)))


def mask_directions(p,mat,i,j):
    sa,sb = p[f"shadow{i}"],p[f"shadow{j}"]
    ha,hb = p[f"glint{i}"],p[f"glint{j}"]
    if mat=="matte":
        ha = hb = np.zeros_like(sa)
    return {"shadow":((sa^sb)&~(ha^hb),np.where(sb,-1,1)),
            "glint":((ha^hb)&~(sa^sb),np.where(hb,-1,1))}


def fit_physics():
    import hashlib
    arrays = {"shadow":[],"glint":[]}
    for row in scenes("train"):
        folder = OUT/"renders"/row["scene"]
        r = hashlib.sha256(row["scene"].encode()).digest()
        ix = slice(r[0]%4,None,4),slice(r[1]%4,None,4)
        with np.load(folder/"passes.npz") as f:
            p = {k:f[k][ix] for k in f.files}
        with np.load(folder/"predictions.npz") as f:
            z = {k:f[k][ix].astype(float) for k in f.files}
        gt = p["z"]
        valid = p["valid"]&(gt>=.25)&(gt<=30.)
        for mat,i,j,a,b in pairs():
            for key,(active,side) in mask_directions(p,mat,i,j).items():
                m = .5*(z[a]+z[b])
                u = .5*(z[b]-z[a])*side
                keep = valid&active&(u!=0)
                arrays[key].append(((gt[keep]-m[keep])/u[keep],abs(u[keep])/gt[keep]/valid.sum()))
    return {k:weighted_median(np.concatenate([s for s,w in vals]),np.concatenate([w for s,w in vals]))
            for k,vals in arrays.items()}


def scale_scores(da,db,fa,fb,feat_a,feat_b,gt,mask,coeff,calibrated_focal):
    bins,side = candidate(feat_a,feat_b)
    weight_b = .5*(1+coeff[bins[mask]]*side[mask])
    log_ratio = float(np.log(db.astype(float)).mean()-np.log(da.astype(float)).mean())
    pred_a,pred_b = da[mask].astype(float),db[mask].astype(float)
    target = gt[mask]
    result = {}
    by_region = {}
    labels = bins[mask]
    def evaluate(a,b,key):
        loss0 = abs(.5*(a+b)/target-1)
        loss = abs(((1-weight_b)*a+weight_b*b)/target-1)
        result[key] = dict(half=float(loss0.mean()),conditional=float(loss.mean()))
        by_region[key] = {}
        for index,name in enumerate(NAMES):
            inside = labels==index
            if inside.any():
                by_region[key][name] = dict(fraction=float(inside.mean()),half=float(loss0[inside].mean()),
                                            conditional=float(loss[inside].mean()))
    for key,factor in (("original",1.),("common_focal",np.sqrt(fb/fa)),("common_geomean",np.exp(.5*log_ratio))):
        evaluate(pred_a*factor,pred_b/factor,key)
    result["absolute_focal_log_ratio"] = abs(float(np.log(fb/fa)))
    result["absolute_depth_log_ratio"] = abs(log_ratio)
    evaluate(pred_a*(calibrated_focal/fa),pred_b*(calibrated_focal/fb),"calibrated_focal")
    result["regions"] = by_region
    return result


def average(rows):
    groups = {r["group"] for r in rows}
    values = {}
    for key in ("original","common_focal","common_geomean","calibrated_focal"):
        values[key] = {}
        for name in ("half","conditional"):
            values[key][name] = float(np.mean([np.mean([r[key][name] for r in rows if r["group"]==g]) for g in groups]))
        values[key]["gain"] = 100*(1-values[key]["conditional"]/values[key]["half"])
        values[key]["regions"] = {}
        for tag in NAMES:
            cells = []
            for group in groups:
                within = [r for r in rows if r["group"]==group]
                for row in within:
                    region = row["regions"][key].get(tag)
                    if region:
                        cells.append((region,region["fraction"]/len(within)/len(groups)))
            if cells:
                mass = sum(w for _,w in cells)
                values[key]["regions"][tag] = {m:sum(r[m]*w for r,w in cells)/mass for m in ("half","conditional")}
    for key in ("absolute_focal_log_ratio","absolute_depth_log_ratio"):
        values[key] = float(np.mean([np.mean([r[key] for r in rows if r["group"]==g]) for g in groups]))
    return values


def run():
    lock = json.loads((OUT/"gain-lock.json").read_text())
    coeff = np.array(lock["coefficients"])
    model = Model()
    physics = fit_physics()
    physical_results = {k:[] for k in physics}
    rows = []
    for row in scenes("val"):
        p,mask,z,feat = render_data(row)
        folder = OUT/"renders"/row["scene"]
        focals = {k:focal(model,np.array(Image.open(folder/(k+".png")))) for k in z}
        for mat,i,j,a,b in pairs():
            true_f = hypersim_intrinsics(row["scene"],mask.shape[1],mask.shape[0])[0,0]
            rec = scale_scores(z[a],z[b],focals[a],focals[b],feat[a],feat[b],p["z"],mask,coeff,true_f)
            rows.append(dict(group=row["family"],condition=[mat,i,j],**rec))
            for key,(active,side) in mask_directions(p,mat,i,j).items():
                use = mask&active
                if use.any():
                    aa,bb = z[a][use].astype(float),z[b][use].astype(float)
                    m = .5*(aa+bb)
                    u = .5*(bb-aa)*side[use]
                    t = p["z"][use]
                    physical_results[key].append(dict(group=row["family"],mass=float(use.sum()/mask.sum()),
                        half=float(abs(m/t-1).mean()),fitted=float(abs((m+physics[key]*u)/t-1).mean())))
    real = []
    for scene,group,a,b,z,mask in middlebury_pairs():
        rec = scale_scores(depth(model,a),depth(model,b),focal(model,a),focal(model,b),
                           image_features(a),image_features(b),z.cpu().numpy(),mask.cpu().numpy(),coeff,3740/3)
        real.append(dict(scene=scene,group=group,**rec))
    result = {}
    for k,v in physical_results.items():
        weights = np.array([r["mass"] for r in v])
        result[k] = {name:float(np.average([r[name] for r in v],weights=weights)) for name in ("half","fitted")}
    save_json(OUT/"mechanisms.json",dict(synthetic=average(rows),real=average(real),synthetic_rows=rows,
              real_rows=real,physics_lambda=physics,physics_scores=result,physics_rows=physical_results,
              lock=sha(OUT/"gain-lock.json")))


if __name__=="__main__":
    run()
