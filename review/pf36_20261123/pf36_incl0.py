"""#36 inclined-block stage 0 (no training): which actions (incline angle, downhill push speed) make the camera able to tell
sliding-friction values apart?  Same flat box, start at rest (or pushed), vary ONLY the friction coefficient.
MuJoCo geom friction = (slide, torsion, roll): ONE sliding coefficient, static = kinetic (soft contact allows tiny creep).
Claims are restricted to that single sliding coefficient.  Camera: per-frame 2-D position noise sd 15 mm, 50 frames at 50 Hz,
each frame observed with prob .6 (as in #36).  Measures per action a and friction mu (deterministic simulation):
  * slide distance after 1 s, slip onset (first frame with displacement > 15 mm),
  * pairwise distinguishability d'(a; mu_i, mu_j) = sqrt(.6 * sum_t |p_i(t) - p_j(t)|^2) / sigma  (Gaussian LLR; error ~ Phi(-d'/2)),
  * Fisher sd of log mu: 1/sqrt(.6 * sum_t |dp/dlog mu|^2 / sigma^2) (central finite difference, +-2% in mu).
See PREREG.md (경사 블록 0단계).  Usage: python pf36_incl0.py"""
import json
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIG, T, SUB, DT, POBS = .015, 50, 10, .002, .6
ANGLES = np.arange(5, 50, 5)                       # degrees
PUSH = np.array([0., .5, 1.])                      # m/s downhill along the slope
MUS = np.geomspace(.08, 1.2, 12)
HALF = (.05, .05, .02)                             # flat box: tips only beyond atan(.05/.02) = 68 deg


def model(mu, ang):
    th = np.deg2rad(ang)
    g = 9.81
    xml = f"""<mujoco><option timestep="{DT}" gravity="{g*np.sin(th)} 0 {-g*np.cos(th)}"/><worldbody>
<geom type="plane" size="100 100 .1" friction="{mu} .005 .0001"/>
<body pos="0 0 {HALF[2]}"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 0 1"/><joint type="hinge" axis="0 1 0"/>
<geom type="box" size="{HALF[0]} {HALF[1]} {HALF[2]}" mass=".5" friction="{mu} .005 .0001"/></body></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def traj(mu, ang, v0):
    m = model(mu, ang); d = mujoco.MjData(m)
    for _ in range(200):                           # settle on the plane with the slope gravity off-axis held: zero horizontal velocity
        mujoco.mj_step(m, d); d.qvel[0] = 0.; d.qpos[0] = 0.
    d.qvel[0] = v0
    out = np.empty((T, 2))
    out[0] = d.qpos[0], d.qpos[1]
    for t in range(1, T):
        for _ in range(SUB):
            mujoco.mj_step(m, d)
        out[t] = d.qpos[0], d.qpos[1]
    return out - out[0]


def main():
    t0 = time.time()
    res = []
    for ang in ANGLES:
        for v0 in PUSH:
            P = np.stack([traj(mu, ang, v0) for mu in MUS])                      # (12, T, 2)
            dp = np.stack([(traj(mu*1.02, ang, v0) - traj(mu/1.02, ang, v0))/(2*np.log(1.02)) for mu in MUS])
            fisher = POBS*np.square(dp).sum((1, 2))/SIG**2
            sd_logmu = 1/np.sqrt(np.maximum(fisher, 1e-12))
            dd = np.sqrt(POBS*np.square(P[:, None]-P[None]).sum((2, 3)))/SIG      # (12, 12)
            adj = np.array([dd[i, i+1] for i in range(len(MUS)-1)])
            disp = np.abs(P[:, :, 0])
            onset = [int(np.argmax(r > SIG)) if (r > SIG).any() else -1 for r in disp]
            res.append(dict(angle=int(ang), push=float(v0), slide_1s_mm=(P[:, -1, 0]*1000).round(1).tolist(), onset_frame=onset,
                            sd_logmu=sd_logmu.round(3).tolist(), adjacent_dprime=adj.round(2).tolist(),
                            dprime_lo_hi=float(dd[0, -1]), n_adjacent_separable=int((adj > 2).sum()),
                            mu_resolved=[float(m) for m, s in zip(MUS, sd_logmu) if s < .1]))
            print(f"[{time.time()-t0:6.1f}s] ang {ang:2d} push {v0:.1f}  sep-adjacent {res[-1]['n_adjacent_separable']:2d}/11"
                  f"  sd(log mu)<.1 at {np.round(res[-1]['mu_resolved'], 2).tolist()}", flush=True)
    crit = [dict(mu=float(m), tan_angle=float(np.rad2deg(np.arctan(m))),
                 measured_onset_angle=next((r["angle"] for r in res if r["push"] == 0 and r["onset_frame"][i] >= 0), None))
            for i, m in enumerate(MUS)]
    out = dict(mus=MUS.tolist(), angles=ANGLES.tolist(), push=PUSH.tolist(), sigma_m=SIG, frames=T, p_obs=POBS,
               rows=res, slip_onset_angle=crit, seconds=time.time()-t0)
    (HERE/"incl0.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
