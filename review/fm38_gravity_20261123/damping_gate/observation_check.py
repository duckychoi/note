"""Constructive ambiguity check for the single practice photo, separate from gate tests."""
import json
import hashlib
import numpy as np
import mujoco
from world import HERE,INERTIA,RATES
from swing import BAND,DT,LEVELS,PROBE_SPEED,SPEED,build,observe,raster


def same_image_key(angle):
    return hashlib.sha256(raster(angle)).hexdigest()


def early_photo(model,start):
    d=mujoco.MjData(model)
    d.qpos[0]=start
    d.qvel[0]=PROBE_SPEED
    mujoco.mj_step(model,d,nstep=round(.24/DT))
    return float(d.qpos[0])


def search():
    # The first matching example in a fixed enumeration is reported, without estimating frequency.
    for initial_deg in (0.,-4.,4.,-8.,8.):
        start=np.deg2rad(initial_deg)
        banks={"seen":{},"strong":{}}
        for domain,bank in banks.items():
            for inc in LEVELS["source"]:
                for c in RATES[domain]:
                    model=build(inc)
                    model.dof_damping[0]=INERTIA*c
                    after=observe(model,start,PROBE_SPEED,True)
                    key=same_image_key(after)
                    row={"incline_deg":float(inc),"damping":float(c),"practice_deg":float(np.rad2deg(after)),
                         "early_deg":float(np.rad2deg(early_photo(model,start))),
                         "peak_deg":[float(np.rad2deg(observe(model,start,s)[1])) for s in SPEED]}
                    bank.setdefault(key,[]).append(row)
        for key in sorted(banks["seen"].keys() & banks["strong"].keys()):
            for weak in banks["seen"][key]:
                for strong in banks["strong"][key]:
                    for target in range(32,92):
                        winners=[]
                        for case in (weak,strong):
                            error=abs(np.array(case["peak_deg"])-target)
                            winners.append(np.flatnonzero(error<=np.rad2deg(BAND))+1)
                        if all(len(w)==1 for w in winners) and winners[0][0]!=winners[1][0]:
                            return {"start_deg":initial_deg,"goal_deg":target,"identical_055s_rgb_sha256":key,
                                    "different_024s_pixels":same_image_key(np.deg2rad(weak["early_deg"]))!=
                                                               same_image_key(np.deg2rad(strong["early_deg"])),
                                    "seen":weak,"strong":strong,"unique_actions":[int(w[0]) for w in winners]}
    return {"example_found":False}


if __name__=="__main__":
    example=search()
    (HERE/"observation_example.json").write_text(json.dumps(example,indent=2),encoding="utf8")
    print(example)
