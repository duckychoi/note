"""Estimate second-view depth reliability, then compare three matched adapters."""
import csv

import cv2
import h5py
import numpy as np
from scipy.stats import rankdata

from common import ROOT, STORE, TEST, save
from hypersim_pilot import SYNTH, SCENES, SAMPLE, INFERRED, design, fit, score_predictions
from multiview_right import OUTPUT


def sample(array, pixels):
    xy = np.ascontiguousarray(pixels, dtype=np.float32)
    return cv2.remap(array.astype(np.float32), xy[..., 0], xy[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def appearance(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    smooth = cv2.GaussianBlur(gray, (9, 9), 2)
    variance = cv2.GaussianBlur(gray**2, (9, 9), 2) - smooth**2
    return gray, (gray-smooth)/np.sqrt(np.maximum(variance, 0)+.004)


def project(xyz, camera):
    p = xyz @ camera.T
    return p[..., :2] / np.maximum(p[..., 2:], 1e-6)


def observe(left_rgb, right_rgb, m, n, left, right, rotation, shift):
    """rotation, shift transform left camera coordinates into right camera coordinates."""
    y, x = np.indices(m.shape)
    pixel = np.stack((x, y, np.ones_like(x)), -1)
    point = (pixel @ np.linalg.inv(left).T) * m[..., None]
    other = point @ rotation.T + shift
    xy = project(other, right)
    ns = sample(n, xy)
    valid = np.isfinite(ns) & (ns > .25) & (ns < 30) & (other[...,2] > .25)
    valid &= np.isfinite(m) & (m > .25) & (m < 30)
    ray = np.concatenate((xy,np.ones_like(xy[...,:1])), -1) @ np.linalg.inv(right).T
    back = (ray * ns[...,None]-shift) @ rotation
    s = back[...,2]
    valid &= np.isfinite(s) & (s > .25) & (s < 30)
    reproj = np.linalg.norm(project(back,left)-pixel[...,:2],axis=-1)
    lum0, feature0 = appearance(left_rgb)
    lum1, feature1 = appearance(right_rgb)
    raw = np.abs(lum0-sample(lum1,xy))
    feature = np.abs(feature0-sample(feature1,xy))
    disagreement = np.abs(np.log(np.maximum(s,.01))-np.log(np.maximum(m,.01)))
    cues = np.stack((np.ones_like(s),raw,feature,np.log1p(reproj),disagreement),-1)
    valid &= np.isfinite(cues).all(-1)
    return s, cues, valid


def synth_poses(scenes, frames, shape):
    h,w = shape
    with (SYNTH/"camera.csv").open(newline="",encoding="utf8") as f:
        params = {r["scene_name"]:r for r in csv.DictReader(f)}
    out = {}
    for j, scene in enumerate(SCENES):
        folder = SYNTH/scene
        with (folder/"metadata_scene.csv").open(newline="",encoding="utf8") as f:
            scale = float(next(r["parameter_value"] for r in csv.DictReader(f)
                               if r["parameter_name"]=="meters_per_asset_unit"))
        vals = []
        for name in ("frame_indices","positions","orientations"):
            with h5py.File(folder/f"camera_keyframe_{name}.hdf5","r") as f:
                vals.append(f["dataset"][:])
        ids, pos, orient = vals
        p = params[scene]
        M = np.array([[float(p[f"M_proj_{a}{b}"]) for b in range(4)] for a in range(4)])
        k = np.array([[w*M[0,0]/2, -w*M[0,1]/2, w*(1-M[0,2])/2-.5],
                      [-h*M[1,0]/2, h*M[1,1]/2, h*(1+M[1,2])/2-.5],[0,0,1]])
        for i in np.flatnonzero(scenes==j):
            idx, = np.flatnonzero(ids.ravel()==int(frames[i]))
            R = orient[idx] @ np.diag([1,-1,-1])
            assert np.allclose(R.T@R,np.eye(3),atol=1e-5) and np.linalg.det(R)>.99
            out[i] = (k,R,pos[idx]*scale)
    return out


def check_geometry(a, poses):
    metrics = {}
    for scene in range(len(SCENES)):
        adjacent = np.flatnonzero(a["scenes"]==scene)
        values = []
        for i, j in zip(adjacent, adjacent[1:]):
            k,r,p = poses[i]
            k1,r1,p1 = poses[j]
            z, n = a["truth"][i], a["truth"][j]
            s, _, valid = observe(a["clean"][i],a["clean"][j],z,n,k,k1,r1.T@r,r1.T@(p-p1))
            if valid.any():
                error = np.abs(s[valid]-z[valid])/z[valid]
                values.append([float(valid.mean()), float(np.median(error)), float((error<.02).mean())])
        metrics[SCENES[scene]] = np.mean(values,axis=0).tolist()
    assert all(row[1] < .01 for row in metrics.values())
    return metrics


def make_synthetic():
    with np.load(SAMPLE) as f:
        a = {k:f[k] for k in f.files}
    with np.load(INFERRED) as f:
        m, n = f["changed"], f["clean"]
    poses = synth_poses(a["scenes"],a["frames"],m.shape[1:])
    geometry = check_geometry(a, poses)
    rng = np.random.default_rng(4342)
    rows = []
    for i in range(len(m)):
        k,r,p = poses[i]
        neighbors = [j for j in poses if j!=i and a["scenes"][i]==a["scenes"][j]]
        j = min(neighbors, key=lambda j: (abs(int(a["frames"][i])-int(a["frames"][j])),j))
        k1,r1,p1 = poses[j]
        obs,cues,valid = observe(a["changed"][i],a["clean"][j],m[i],n[j],k,k1,r1.T@r,r1.T@(p-p1))
        z = a["truth"][i]
        base_valid = np.isfinite(z) & (z>.25) & (z<30) & np.isfinite(m[i]) & (m[i]>0)
        pool = np.flatnonzero(base_valid)
        if not len(pool):
            print("No eligible reference pixels", SCENES[a["scenes"][i]], a["frames"][i], flush=True)
            continue
        ids = rng.choice(pool,min(len(pool),1024),replace=False)
        f = design(a["changed"][i],m[i]).reshape(-1,7)[ids]
        rows.append(dict(scene=int(a["scenes"][i]), cues=cues.reshape(-1,5)[ids],
                         valid=valid.ravel()[ids], x=f,
                         m=np.log(m[i].ravel()[ids]), s=np.log(np.maximum(obs.ravel()[ids],.01)),
                         z=np.log(z.ravel()[ids])))
    return rows, geometry


def stack(rows):
    return {k:np.concatenate([r[k][r["valid"]] for r in rows]) for k in ("cues","m","s","z")}


def model(cal):
    c = cal["cues"]
    error = cal["s"]-cal["z"]
    target = np.log(.0025+error**2)
    fit_r = np.linalg.solve(c.T@c+.01*len(c)*np.eye(c.shape[1]),c.T@target)
    risk = np.clip(c@fit_r,-9,4)
    cut = np.quantile(risk,[.25,.5,.75])
    delta = cal["s"]-cal["m"]

    def coefficient(group):
        d = delta[group]
        return np.clip((d@(cal["z"]-cal["m"])[group]) / max(d@d,1e-9),0,1)

    scalar = coefficient(slice(None))
    bins = np.digitize(risk,cut)
    gain = np.array([coefficient(bins==i) for i in range(4)])
    return fit_r,cut,gain,float(np.exp(np.median(risk))),float(scalar)


def reliability(cues, learned):
    weights,cut,gain,typical,scalar = learned
    risk = np.clip(cues@weights,-9,4)
    return risk,gain[np.digitize(risk,cut)],1/(1+np.exp(risk)/typical)


def auc(value, error):
    bad = error>np.quantile(error,.8)
    n = int(bad.sum())
    return float((rankdata(value)[bad].sum()-n*(n+1)/2)/(n*(len(bad)-n)))


def measure(cal, learned):
    error = np.abs(cal["s"]-cal["z"])
    preds = cal["cues"]
    risk, k, _ = reliability(preds,learned)
    d = cal["s"]-cal["m"]
    return dict(n=len(d), raw_rgb=auc(preds[:,1],error), normalized_feature=auc(preds[:,2],error),
                pixel_cycle=auc(preds[:,3],error), depth_gap=auc(preds[:,4],error), combined=auc(risk,error),
                logmae_primary=float(np.abs(cal["m"]-cal["z"]).mean()),
                logmae_fixed=float(np.abs(cal["m"]+learned[-1]*d-cal["z"]).mean()),
                logmae_adaptive=float(np.abs(cal["m"]+k*d-cal["z"]).mean()))


def train(rows, learned):
    dataset = {name:[] for name in ("gt","fixed","output","param","combined")}
    for r in rows:
        y = r["z"]-r["m"]
        valid = r["valid"]
        _, k, weight = reliability(r["cues"][valid],learned)
        delta = r["s"][valid]-r["m"][valid]
        dataset["gt"].append((r["x"],y))
        for name, coefficient, strength in (
            ("fixed",learned[-1],.25),("output",k,.25),("param",1.,.5*weight),
            ("combined",k,.5*weight),
        ):
            target=y.copy()
            target[valid] = (1-strength)*y[valid] + strength*coefficient*delta
            dataset[name].append((r["x"],target))
    return {"frozen":np.zeros(7),"gt_adamw":fit(dataset["gt"],"adamw"),
            "gt_kf":fit(dataset["gt"],"kalman"),"fixed_adamw":fit(dataset["fixed"],"adamw"),
            "output_adamw":fit(dataset["output"],"adamw"),"reliability_adamw":fit(dataset["param"],"adamw"),
            "reliability_kf":fit(dataset["param"],"kalman"),"combined_kf":fit(dataset["combined"],"kalman")}


def main():
    from fit import observe as stereo

    rows, geometry = make_synthetic()
    calibration = [r for r in rows if r["scene"]>=4]
    learned = model(stack(calibration))
    dev = {str(i):measure(stack([r for r in calibration if r["scene"]==i]),
                         model(stack([r for r in calibration if r["scene"]!=i]))) for i in (4,5)}
    print("Synthetic leave-one-scene-out", dev, "gains", learned[2], flush=True)
    frozen, features, real_checks, independent = {}, {}, {}, {}
    with np.load(ROOT/"depthpro2-s_predictions.npz") as left, np.load(OUTPUT/"predictions.npz") as right:
        for scene in TEST:
            with np.load(STORE/f"{scene}_wide.npz") as f, np.load(OUTPUT/f"{scene}.npz") as c:
                m = left[scene]
                z = f["z"]
                s,g,v = observe(f["rgb"],c["rgb"],m,right[scene],c["left"],c["right"],np.eye(3),
                                [-float(c["baseline"]),0,0])
                ok = v&f["valid"]
                if ok.sum()>100:
                    real_checks[scene] = measure(dict(cues=g[ok],m=np.log(m[ok]),s=np.log(s[ok]),z=np.log(z[ok])),learned)
                # Independent stereo reference, evaluated only on its measured coverage.
                k0,k1,b = c["left"],c["right"],float(c["baseline"])
                independent_z, sv, _, _ = stereo(f["rgb"],c["rgb"],k0[0,0],b,k1[0,2]-k0[0,2],128)[0]
                support = sv & f["valid"]
                if support.sum()>100:
                    logits = np.log(np.stack((m[support],independent_z[support])))
                    errors = np.abs(logits - np.log(z[support]))
                    independent[scene] = dict(n=int(support.sum()),
                        mono_logmae=float(errors[0].mean()),stereo_logmae=float(errors[1].mean()))
                frozen[scene] = m
                features[scene] = design(f["rgb"],m)
    print("Real projected observation coverage",len(real_checks),flush=True)
    arms = train([r for r in rows if r["scene"]<4],learned)
    results = {name:score_predictions({s:frozen[s]*np.exp(features[s]@w) for s in TEST})
               for name,w in arms.items()}
    mean = {k:float(np.mean([np.mean([v[k] for s,v in real_checks.items() if s[:-1]==f])
                             for f in sorted({s[:-1] for s in real_checks})]))
            for k in next(iter(real_checks.values())) if k!="n"}
    print("Real cue diagnostics",mean,flush=True)
    print("One-view performance",results,flush=True)
    save("multiview_probe_results.json",dict(synthetic=dev, real=real_checks, real_mean=mean, geometry=geometry,
         gains=learned[2].tolist(), fixed_gain=learned[-1], single_view=results,
         independent_stereo=independent))


if __name__=="__main__":
    main()
