"""Replay affordable actions without assigning acquisition seconds to saved images."""
import json
import math

import cv2
import numpy as np

from robi_diagnostic import calibration, grid, quality, read_depth, read_image, render
from robi_files import HERE, OBJECT, SENSORS, output_file, selection


def match_coverage(row, depth):
    sensor = row["sensor"]
    kl, kr, kd = [calibration(sensor, c) for c in ("LEFT","RIGHT","DEPTH")]
    offset = kl["cx"]-kr["cx"]-kd["disparity_shift"]
    fb = kd["fx"]*kd["baseline"]
    minimum = math.floor(fb/1200+offset)
    maximum = math.ceil(fb/250+offset)
    bins = math.ceil((maximum-minimum+1)/16)*16
    folder = row["raw"][0].split("/Depth/")[0]+"/StereoPattern/"
    left, right = [read_image(f"{folder}{side}_view_{row['ids'][0]}.bmp",cv2.IMREAD_GRAYSCALE)
                   for side in ("LEFT","RIGHT")]
    matcher = cv2.StereoSGBM_create(minDisparity=minimum,numDisparities=bins,blockSize=5,
                                   P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=8,
                                   speckleWindowSize=40,speckleRange=2)
    d = matcher.compute(left,right).astype(float)/16
    valid = (d>=minimum)&(d<=maximum)&(d>offset)
    # At either search boundary the correspondence must still lie on the right image.
    xr = np.rint(np.arange(d.shape[1])-d)
    valid &= (xr>=0)&(xr<d.shape[1])
    comparable = (depth>0)&valid
    residual = d[comparable]-(fb/depth[comparable]+offset)
    if residual.size == 0:
        raise ValueError("no native matches to check")
    search_width = left.shape[1]-max(minimum+bins,0)-max(-minimum,0)
    return dict(coverage=float(valid.mean()),range=[minimum,maximum],search_width_fraction=search_width/left.shape[1],
                median_native_disparity_residual_px=float(np.median(residual)),
                median_abs_native_disparity_residual_px=float(np.median(np.abs(residual))),
                fraction_within_2px=float((np.abs(residual)<=2).mean()))


def observe(row):
    sensor = row["sensor"]
    first, second = row["raw"]
    depth = read_depth(first,sensor)
    match = match_coverage(row,depth)
    x,y = grid(depth)
    visible = depth[y,x]>0
    fraction, scatter = quality(depth)
    # No future view, GT, or auxiliary depth is used to choose the point.
    pool = np.flatnonzero(visible)
    center_distance = (x[pool]-depth.shape[1]/2)**2+(y[pool]-depth.shape[0]/2)**2
    order = np.lexsort((center_distance,scatter[y[pool],x[pool]],-fraction[y[pool],x[pool]]))
    px,py = x[pool[order[0]]],y[pool[order[0]]]
    other = next(s for s in SENSORS if s!=sensor)
    independent = first.replace("/"+sensor+"/","/"+other+"/")
    single = render(independent,other,first,sensor)[py,px]
    scale = float(single/depth[py,px]) if single>0 else 1.
    projected = render(second,sensor,first,sensor)
    raw = depth[y,x]
    actions = {"stop":raw,"point":raw*scale,"view":np.where(raw>0,raw,projected[y,x])}
    reference = render(row["gt"][0],"Ensenso",first,sensor)[y,x]
    keep = (reference>0)&((np.abs(px-x)>2)|(np.abs(py-y)>2))
    metrics = {}
    for action,value in actions.items():
        estimate = value[keep]
        absolute = np.abs(estimate-reference[keep])
        losses = {str(p):float(np.where(estimate>0,absolute,p).mean()) for p in (25,100,250)}
        metrics[action] = dict(loss_mm=losses,coverage=float((estimate>0).mean()))
    return dict(sensor=sensor,scene=row["scene"],initial=row["ids"][0],match=match,
                raw_coverage=float(visible.mean()),baseline="view" if match["coverage"]<.55 else "point",
                evaluation_points=int(keep.sum()),point_xy=[int(px),int(py)],point_returned=bool(single>0),
                scale=scale,actions=metrics)


def score(rows,cut):
    actions = ["stop" if r["raw_coverage"]>=cut else r["baseline"] for r in rows]
    chosen = [r["actions"][a] for r,a in zip(rows,actions)]
    # All scenes have exactly two starting poses here.
    assert len(set([sum(r["scene"]==s for r in rows) for s in {r["scene"] for r in rows}]))==1
    return dict(counts={a:actions.count(a) for a in ("stop","point","view")},
                loss_mm={k:float(np.mean([m["loss_mm"][k] for m in chosen])) for k in ("25","100","250")},
                coverage=float(np.mean([m["coverage"] for m in chosen])))


def evaluate(rows):
    dev = [r for r in rows if r["scene"] in (4,8)]
    test = [r for r in rows if r["scene"] in (5,9)]
    ref = score(dev,1.01)
    feasible = []
    for cut in (0,.25,.5,.75,.90,1.01):
        result = score(dev,cut)
        if result["loss_mm"]["100"]<=ref["loss_mm"]["100"] and result["coverage"]>=ref["coverage"]:
            feasible.append((result["counts"]["stop"],cut))
    cut = max(feasible)[1]
    return dict(cut=cut,dev55=ref,dev_candidate=score(dev,cut),test55=score(test,1.01),
                test_candidate=score(test,cut),individual_scenes={
                    str(s):dict(baseline=score([r for r in test if r["scene"]==s],1.01),
                                candidate=score([r for r in test if r["scene"]==s],cut)) for s in (5,9)})


if __name__ == "__main__":
    cv2.setNumThreads(1)
    rows = [observe(r) for r in selection()]
    if OBJECT=="Chrome_screw":
        summary = {s:evaluate([r for r in rows if r["sensor"]==s]) for s in SENSORS}
    else:
        training = json.loads((HERE/"replay.json").read_text())["summary"]
        summary = {}
        for sensor in SENSORS:
            subset = [r for r in rows if r["sensor"]==sensor]
            cut = training[sensor]["cut"]
            summary[sensor] = dict(cut=cut,test55=score(subset,1.01),test_candidate=score(subset,cut),
                individual_scenes={str(s):dict(baseline=score([r for r in subset if r["scene"]==s],1.01),
                    candidate=score([r for r in subset if r["scene"]==s],cut)) for s in sorted({r["scene"] for r in subset})})
    for sensor in SENSORS:
        test = [r for r in rows if r["sensor"]==sensor and (OBJECT!="Chrome_screw" or r["scene"] in (5,9))]
        summary[sensor]["fixed_actions"] = {
            arm:score([dict(r,baseline=arm) for r in test],1.01) for arm in ("stop","point","view")
        }
    output_file("replay.json").write_text(json.dumps(dict(rows=rows,summary=summary),indent=2),encoding="utf8")
    print(json.dumps(summary,indent=1))
