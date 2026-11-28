"""Measure the exact removable, order-dependent teacher risk, holding all four gains fixed."""
import json

import numpy as np
import torch

from light_encoder_score import middlebury_pairs, summary
from light_encoder_study import Model, OUT, alternate, edges, load_scene, manifest, sha, write, yield_memory


def synthetic():
    for row in manifest():
        if row[0]=="val":
            rgb,z,mask,_ = load_scene(row)
            yield str(row[1:4]),row[1],*alternate(rgb,row),z,mask


@torch.no_grad()
def main():
    model = Model()
    calibration = json.loads((OUT/"calibration.json").read_text())
    gains = torch.tensor(calibration["conditional"],device="cuda",dtype=torch.float64)
    results = {}
    for name, pairs in (("synthetic",synthetic()),("physical",middlebury_pairs())):
        rows = []
        for key,group,a,b,z,mask in pairs:
            yield_memory()
            _,_,qa = model.baseline(a)
            _,_,qb = model.baseline(b)
            _,d,r,ab,keep = edges(a,b,qa,qb,z,mask)
            _,e,_,ba,keep_reverse = edges(b,a,qb,qa,z,mask)
            if not torch.equal(keep,keep_reverse) or not torch.equal(d,-e):
                raise ValueError("Two orientations disagree on mask or displacement")
            if not r.numel():
                continue
            d,r = d.double(),r.double()
            error_a = gains[ab]*d-r
            error_b = (1-gains[ba])*d-r
            def mse(x):
                return float(x.square().mean())
            scores = dict(original=.5*(mse(r)+mse(d-r)),half=mse(.5*d-r),
                          ordered=.5*(mse(error_a)+mse(error_b)),symmetric=mse(.5*(error_a+error_b)),
                          removable=.25*mse((gains[ab]+gains[ba]-1)*d))
            if not np.isclose(scores["ordered"]-scores["symmetric"],scores["removable"],atol=1e-12,rtol=1e-9):
                raise ValueError((name,key,scores))
            rows.append(dict(key=key,group=group,scores=scores))
        results[name] = dict(**summary(rows),rows=rows)
        print(name,results[name]["means"],flush=True)
    write(OUT/"order-diagnostic.json",dict(datasets=results,calibration=sha(OUT/"calibration.json")))


if __name__=="__main__":
    main()
