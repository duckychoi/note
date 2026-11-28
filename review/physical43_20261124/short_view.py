"""Try closer recorded Ensenso poses under identical depth/coverage constraints."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np

from robi_diagnostic import grid, pose, read_depth, render
from robi_files import CACHE, HERE, OBJECT, cached, output_file, selection, views


def compare(row, original, offsets):
    first = row["raw"][0]
    depth = read_depth(first,"Ensenso")
    x,y = grid(depth)
    px,py = original["point_xy"]
    gt = render(row["gt"][0],"Ensenso",first,"Ensenso")[y,x]
    good = (gt>0)&((np.abs(x-px)>2)|(np.abs(y-py)>2))
    out = {}
    for step,name in offsets.items():
        extra = render(name,"Ensenso",first,"Ensenso")[y,x]
        fill = np.where(depth[y,x]>0,depth[y,x],extra)[good]
        residual = np.abs(fill-gt[good])
        change = np.linalg.solve(pose(first),pose(name))
        angle = np.arccos(np.clip((np.trace(change[:3,:3])-1)/2,-1,1))
        out[step] = dict(distance_mm=float(np.linalg.norm(change[:3,3])),rotation_deg=float(np.degrees(angle)),
                         coverage=float((fill>0).mean()),loss_mm=float(np.where(fill>0,residual,100).mean()))
    assert np.isclose(out[3]["loss_mm"],original["actions"]["view"]["loss_mm"]["100"])
    return dict(scene=row["scene"],initial=row["ids"][0],options=out)


def average(items):
    return {k:float(np.mean([m[k] for m in items])) for k in items[0]}


def shortest(options):
    base = options[3]
    possible = [i for i,m in options.items() if m["coverage"]>=base["coverage"]
                and m["loss_mm"]<=base["loss_mm"] and m["rotation_deg"]<=base["rotation_deg"]]
    return min(possible,key=lambda i:(options[i]["distance_mm"],-i))


if __name__=="__main__":
    inputs = [r for r in selection() if r["sensor"]=="Ensenso"]
    original = json.loads(output_file("replay.json").read_text())["rows"]
    baseline = {(r["scene"],r["initial"]):r for r in original if r["sensor"]=="Ensenso"}
    episodes = []
    names=set()
    for row in inputs:
        native = views(row["scene"],"Ensenso","Ensenso/Depth")
        ordered = sorted(native)
        i = ordered.index(row["ids"][0])
        offsets = {step:native[ordered[i+step]] for step in (1,2,3)}
        assert offsets[3]==row["raw"][1]
        for name in offsets.values():
            names.update((name,name[:-3]+"json"))
        episodes.append((row,baseline[(row["scene"],row["ids"][0])],offsets))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(cached,sorted(names)))
    rows = [compare(*ep) for ep in episodes]
    if OBJECT=="Chrome_screw":
        dev = [r for r in rows if r["scene"] in (4,8)]
        cut = shortest({s:average([r["options"][s] for r in dev]) for s in (1,2,3)})
        test = [r for r in rows if r["scene"] in (5,9)]
    else:
        cut = json.loads((HERE/"short_view.json").read_text())["cut"]
        test = rows
    oracle = [shortest(r["options"]) for r in test]
    output = dict(cut=cut,rows=rows,reference=average([r["options"][3] for r in test]),
                  candidate=average([r["options"][cut] for r in test]),oracle_choices=oracle,
                  oracle=average([r["options"][i] for i,r in zip(oracle,test)]),
                  sha256={n:hashlib.sha256((CACHE/n).read_bytes()).hexdigest() for n in sorted(names)})
    output_file("short_view.json").write_text(json.dumps(output,indent=2),encoding="utf8")
