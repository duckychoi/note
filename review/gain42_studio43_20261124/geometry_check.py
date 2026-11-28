"""Synthetic-only visibility diagnostic on the independent audit split, by true object type."""
from collections import defaultdict
import json

import cv2
import numpy as np

from experiment import SEEDS
from scenes import HEIGHT, WIDTH, HERE, X, Y, disparities, pair


def main():
    output = {}
    rows = Y.astype(int)
    for seed in SEEDS:
        groups = defaultdict(lambda:np.zeros(5,np.float64))
        for parent in range(16):
            frames, f, base = pair(seed,2,parent,0)
            disps = disparities(*[cv2.cvtColor(frame[0],cv2.COLOR_RGB2GRAY) for frame in frames])
            for i,direction in enumerate((-1,1)):
                _, z, label = frames[i]
                true_d = f*base/z
                target = np.rint(X+direction*true_d).astype(int)
                other_labels = frames[1-i][2]
                visible = ((target>=0)&(target<WIDTH)&(other_labels[rows,target.clip(0,WIDTH-1)]==label))
                d = disps[i]
                guess = np.rint(X+direction*d).astype(int)
                reverse = disps[1-i][rows,guess.clip(0,WIDTH-1)]
                accepted = visible & (guess>=0) & (guess<WIDTH) & (d>.25) & (reverse>.25) & (abs(d-reverse)<=1)
                thin = (label%3)==1
                band = cv2.morphologyEx(thin.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8)).astype(bool)
                masks = {"thin_surface":thin,"thin_boundary":band,"away_from_boundary":~band,"all":np.ones(z.shape,bool)}
                for name, area in masks.items():
                    verified = accepted&area
                    groups[name] += [area.sum(),(visible&area).sum(),verified.sum(),
                                     (verified&(abs(d-true_d)/true_d >.1)).sum(),
                                     (abs(d-true_d)*verified).sum()]
        output[seed] = {name:{"pixels":int(a),"visible":v/a,"accepted":n/a,
                               "bad_disparity_among_accepted":bad/n,
                               "mean_pixel_disparity_error_among_accepted":err/n}
                        for name,(a,v,n,bad,err) in groups.items()}
    (HERE/"geometry_check.json").write_text(json.dumps(output,indent=1),encoding="utf8")


if __name__=="__main__":
    cv2.setNumThreads(1)
    main()
