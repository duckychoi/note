"""Measure native depth errors and two-view occlusion on frozen ROBI episodes."""
from functools import lru_cache
import json

import cv2
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from robi_files import CACHE, HERE, OBJECT, SENSORS, output_file, selection


@lru_cache(maxsize=8)
def calibration(sensor, channel="DEPTH"):
    return yaml.safe_load((CACHE/f"{sensor}_{channel}.yml").read_text())


def read_image(name, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imdecode(np.fromfile(CACHE/name, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(name)
    return image


def read_depth(name, sensor):
    k = calibration(sensor)
    image = read_image(name)
    assert image.dtype == np.uint16 and image.shape == (k["height"], k["width"])
    return image.astype(float)*k["depth_unit"]


def pose(name):
    rows = (CACHE/(name[:-3]+"json")).read_text().splitlines()
    return np.array(json.loads(rows[0])["cam_pose"]).reshape(4, 4)


def project(x, y, z, source, target, ks, kt):
    transform = np.linalg.solve(target, source)
    local = np.stack(((x-ks["cx"])*z/ks["fx"], (y-ks["cy"])*z/ks["fy"], z))
    xyz = transform[:3, :3] @ local + transform[:3, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(xyz[0]*kt["fx"]/xyz[2]+kt["cx"])
        v = np.rint(xyz[1]*kt["fy"]/xyz[2]+kt["cy"])
    inside = (xyz[2]>0)&(u>=0)&(u<kt["width"])&(v>=0)&(v<kt["height"])
    u = np.clip(np.nan_to_num(u), 0, kt["width"]-1).astype(int)
    v = np.clip(np.nan_to_num(v), 0, kt["height"]-1).astype(int)
    return u, v, xyz[2], inside


def render(name, sensor, target_name, target_sensor):
    depth = read_depth(name, sensor)
    src, dst = pose(name), pose(target_name)
    if sensor == target_sensor and np.allclose(src, dst, atol=1e-7):
        return depth
    ks, kt = calibration(sensor), calibration(target_sensor)
    y, x = np.nonzero(depth)
    u, v, z, inside = project(x, y, depth[y,x], src, dst, ks, kt)
    output = np.full((kt["height"], kt["width"]), np.inf)
    np.minimum.at(output, (v[inside], u[inside]), z[inside])
    output[~np.isfinite(output)] = 0
    return output


def grid(depth):
    h, w = depth.shape
    y, x = np.mgrid[h//4:3*h//4:4, w//4:3*w//4:4]
    return x.ravel(), y.ravel()


def quality(depth):
    def box(array):
        return cv2.boxFilter(array.astype(float), -1, (5, 5), normalize=True)
    fraction = box(depth>0)
    divisor = np.maximum(fraction, 1e-8)
    mean = box(depth)/divisor
    std = np.sqrt(np.maximum(0, box(depth**2)/divisor-mean**2))
    return fraction, std


def conservative_occlusion(reference, u, v, z, inside):
    mask = np.ones((3, 3), np.uint8)
    full = inside & (u>0)&(v>0)&(u<reference.shape[1]-1)&(v<reference.shape[0]-1)
    full &= cv2.erode(reference, mask)[v, u]>0
    hidden = full & (cv2.dilate(reference, mask)[v,u]+10 < z)
    return full, hidden


def diagnose(row):
    sensor = row["sensor"]
    raw1, raw2 = row["raw"]
    first, second = [read_depth(n, sensor) for n in row["raw"]]
    gt1, gt2 = [render(name, "Ensenso", target, sensor)
                for name, target in zip(row["gt"][:2], row["raw"])]
    k = calibration(sensor)
    x, y = grid(first)
    ref, obs = gt1[y,x], first[y,x]
    known = ref>0
    valid = known & (obs>0)
    missing = known & ~valid
    error = np.abs(ref[valid]-obs[valid])
    frac, std = quality(first)
    features = np.stack((frac[y[valid],x[valid]],std[y[valid],x[valid]]),axis=1)
    x0, y0 = x[known], y[known]
    u,v,z,inside = project(x0,y0,ref[known],pose(raw1),pose(raw2),k,k)
    eligible = inside & (gt2[v,u]>0)
    unmatched = (second[v,u]==0)|(np.abs(second[v,u]-z)>10)
    initially_missing = missing[known]
    third = read_depth(row["gt"][2], "Ensenso")
    a,b = grid(third)
    keep = third[b,a]>0
    a,b = a[keep],b[keep]
    comparable, hidden = [], []
    for target, reference in zip(row["raw"], (gt1,gt2)):
        p = project(a,b,third[b,a],pose(row["gt"][2]),pose(target),calibration("Ensenso"),k)
        full, occluded = conservative_occlusion(reference,*p)
        comparable.append(full)
        hidden.append(occluded)
    def count(mask):
        return int(np.count_nonzero(mask))
    result = dict(sensor=sensor,scene=row["scene"],views=row["ids"],
                  translation_mm=float(np.linalg.norm(pose(raw1)[:3,3]-pose(raw2)[:3,3])),
                  sampled=len(x),known=count(known),missing=count(missing),valid=count(valid),
                  gross=count(error>10),median_absolute_mm=float(np.median(error)),
                  moved_out_of_view=count(~inside),moved_gt_comparable=count(eligible),
                  occluded_after_move=count(eligible & (gt2[v,u]+10<z)),
                  missing_in_view=count(initially_missing & inside),
                  missing_unrecovered=count(initially_missing & inside & unmatched),
                  third_known=len(a),both_comparable=count(comparable[0]&comparable[1]),
                  hidden_at_both=count(hidden[0]&hidden[1]))
    return result, features, error>10


def classify(samples, frozen=None):
    train = [s for s in samples if s[0]["scene"] in (4,8)]
    test = samples if frozen else [s for s in samples if s[0]["scene"] in (5,9)]
    def arrays(items):
        return np.concatenate([s[1] for s in items]), np.concatenate([s[2] for s in items])
    x,y = arrays(train)
    result = {}
    for method, cols in (("coverage",[0]),("scatter",[1]),("both",[0,1])):
        if frozen:
            params = frozen[method]["model"]
        else:
            model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(x[:,cols],y)
            scale, logistic = model.steps[0][1], model.steps[1][1]
            coef = logistic.coef_[0]/scale.scale_
            params = dict(coef=coef.tolist(),intercept=float(logistic.intercept_[0]-coef@scale.mean_))
        metrics = {"model":params}
        groups = {f"scene{s}":[r for r in test if r[0]["scene"]==s] for s in sorted({r[0]["scene"] for r in test})}
        for name,split in {"pooled":test,**groups}.items():
            xx,yy = arrays(split)
            prob = expit(xx[:,cols]@params["coef"]+params["intercept"])
            metrics[name] = dict(n=len(yy), positives=int(yy.sum()),prevalence=float(yy.mean()),
                                 auc=float(roc_auc_score(yy,prob)),ap=float(average_precision_score(yy,prob)))
        result[method] = metrics
    return result


if __name__ == "__main__":
    cv2.setNumThreads(1)
    output = {"pairs":[],"quality":{}}
    for sensor in SENSORS:
        samples = [diagnose(r) for r in selection() if r["sensor"]==sensor]
        output["pairs"].extend(s[0] for s in samples)
        frozen = None if OBJECT=="Chrome_screw" else json.loads((HERE/"diagnostic.json").read_text())["quality"][sensor]
        output["quality"][sensor] = classify(samples,frozen)
        for field in ("known","missing","valid","gross","both_comparable","hidden_at_both"):
            print(sensor,field,sum(r[0][field] for r in samples))
    output_file("diagnostic.json").write_text(json.dumps(output,indent=2),encoding="utf8")
