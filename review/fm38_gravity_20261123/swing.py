"""Inclined-plane pendulum on Earth with a head-on board camera."""
from pathlib import Path
import os
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
CACHE=Path(os.environ.get("FM38_DATA","E:/0.Project/fm38_visual"))/"swing"
G=9.81
L=.45
R=.025
DT=.002
SPEED=np.array([2.4,3.3,4.2,5.1])
PROBE_SPEED=2.
PROBE_TIME=.55
TIMES=np.array([.18,.30,.42,.54,.66])
BAND=np.deg2rad(7.5)
LEVELS={"source":np.linspace(25,45,9),"shift":np.linspace(65,85,9)}


def build(incline):
    import mujoco
    # The board's y-axis is its hinge normal; local -z is downhill.
    xml=f"""<mujoco><compiler angle="degree"/><option timestep="{DT}" integrator="RK4" gravity="0 0 -9.81"/>
    <worldbody><body euler="{90-incline} 0 0"><body><joint type="hinge" axis="0 -1 0"/>
    <geom type="sphere" pos="0 0 -{L}" size="{R}" mass=".2" contype="0" conaffinity="0"/>
    </body></body></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def observe(model,start,speed,probe=False):
    import mujoco
    data=mujoco.MjData(model)
    data.qpos[0]=start
    data.qvel[0]=speed
    if probe:
        mujoco.mj_step(model,data,nstep=round(PROBE_TIME/DT))
        return float(data.qpos[0])
    targets=np.rint(TIMES/DT).astype(int)
    frames=np.empty(len(targets))
    pos=0
    peak=start
    turning=False
    for t in range(1,601):
        mujoco.mj_step(model,data)
        q=data.qpos[0]
        if not turning:
            peak=max(peak,q)
            turning=data.qvel[0]<=0
        if pos<len(targets) and t==targets[pos]:
            frames[pos]=q
            pos+=1
        if turning and pos==len(targets):
            break
    assert turning and pos==len(targets)
    return frames,peak


def generate(split,domain,n,seed):
    rng=np.random.default_rng(seed)
    model=[build(i) for i in LEVELS[domain]]
    starts=rng.uniform(-8,8,n)*np.pi/180
    targets=rng.uniform(32,91,n)*np.pi/180
    levels=rng.integers(0,len(model),n)
    logged=rng.integers(0,4,n)
    rows=[]
    for i,(start,goal,k) in enumerate(zip(starts,targets,levels)):
        probe=observe(model[k],start,PROBE_SPEED,True)
        for a in ([logged[i]] if split=="train" else range(4)):
            future,peak=observe(model[k],start,SPEED[a])
            rows.append((i,a,goal,levels[i],peak,[start,probe,*future]))
    streams=list(zip(*rows))
    return {name:np.array(v) for name,v in zip(("state","action","goal","level","peak","angles"),streams)}


def raster(angle):
    image=Image.new("RGB",(96,96),(240,241,236))
    pen=ImageDraw.Draw(image)
    x=int(round(28+49*np.sin(angle)))
    y=int(round(30+49*np.cos(angle)))
    pen.line([(28,30),(x,y)],fill=(62,76,92),width=3)
    pen.ellipse((24,26,32,34),fill=(66,79,92))
    pen.ellipse((x-5,y-5,x+5,y+5),fill=(194,77,58))
    return np.asarray(image)


def camera_angle(im):
    y,x=np.nonzero(np.all(im==[194,77,58],axis=-1))
    return np.arctan2(x.mean()-28,y.mean()-30)


if __name__=="__main__":
    CACHE.mkdir(parents=True,exist_ok=True)
    for split,domain,n,seed in (("train","source",2048,48101),("val","source",192,48102),
                                ("source","source",512,48103),("shift","shift",512,48103)):
        d=generate(split,domain,n,seed)
        np.savez_compressed(CACHE/f"{split}.npz",**d)
        print(split,len(d["action"]),flush=True)
