"""Two visual MuJoCo tasks with four interventions from the same observable start."""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get("FM38_DATA","E:/0.Project/fm38_visual"))
N_ACTION = 4
TASKS = ("push","press")
MATERIAL = {"push":{"source":(.32,.52),"shift":(.85,1.05)},
            "press":{"source":(42.,58.),"shift":(85.,105.)}}
BAND = {"push":(.19,.33),"press":(.035,.080)}
FUTURE = 5


def build(task,material):
    import mujoco
    if task=="push":
        nodes = f"""<geom name="ground" type="plane" size="1 1 .02" priority="1" friction="{material} .005 .0001"/>
        <body name="block" pos="0 0 .022"><freejoint/><geom type="box" pos="0 .018 0" size=".055 .019 .022" mass=".12"/>
        <geom type="box" pos="0 -.018 0" size=".019 .018 .022" mass=".06"/></body>
        <body name="tip" mocap="true"><geom name="tip" type="sphere" size=".023"/></body>"""
    else:
        nodes = f"""<body name="pad" pos=".34 0 .06"><joint name="spring" type="slide" axis="1 0 0"
          stiffness="{material}" damping="1.8" springref="0"/><geom name="pad" type="box" size=".027 .055 .05" mass=".16"/></body>
        <body name="ram" pos="0 0 .06"><joint type="slide" axis="1 0 0" damping="1.4"/>
          <geom name="tip" type="sphere" size=".030" mass=".20"/></body>"""
    xml = f"""<mujoco><option timestep=".002" integrator="implicitfast"/><default>
      <geom solref=".012 1" friction=".35 .005 .0001"/></default><worldbody>{nodes}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def init(task,m,params):
    import mujoco
    d = mujoco.MjData(m)
    x,y,angle,gap,offset = params[:5]
    if task=="push":
        d.qpos[:] = [x,y,.022, np.cos(angle/2),0,0,np.sin(angle/2)]
        d.mocap_pos[0] = [x-gap,y+offset,.025]
    else:
        d.qpos[:] = [0,x]
    mujoco.mj_forward(m,d)
    return d


def observe(task,d):
    if task=="push":
        w,x,y,z = d.qpos[3:7]
        yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        return [d.qpos[0],d.qpos[1],yaw,d.mocap_pos[0,0],d.mocap_pos[0,1],d.mocap_pos[0,2]]
    return [d.qpos[0],d.qpos[1],0.,0.,0.,0.]


def simulate(task,m,params,a):
    import mujoco
    d = init(task,m,params)
    stages = [observe(task,d)+[params[-1]]]
    contacts = [False]
    tip = m.geom("tip").id
    for frame in range(FUTURE):
        for step in range(40):
            t = .002*(40*frame+step+1)
            if task=="push":
                speed = (1.0,1.45,1.9,2.35)[a]
                d.mocap_pos[0] = [params[0]-params[3]+speed*min(.135,t),
                                  params[1]+params[4], .025+min(.24, max(0.,t-.135)*8)]
            else:
                d.qfrc_applied[1] = (1.45,2.45,3.60,5.0)[a]
            mujoco.mj_step(m,d)
        stages.append(observe(task,d)+[params[-1]])
        contacts.append(any(tip in (c.geom1,c.geom2) for c in d.contact))
    return stages,contacts


def generate(task,domain,n,seed,branch):
    rng = np.random.default_rng(seed)
    models = [build(task,v) for v in np.linspace(*MATERIAL[task][domain],9)]
    states,acts,rows,contacts,materials = [],[],[],[],[]
    for i in range(n):
        mi = rng.integers(len(models))
        if task=="push":
            p = [rng.uniform(-.055,.035),rng.uniform(-.045,.045),rng.uniform(-.35,.35),
                 rng.uniform(.077,.113),rng.uniform(-.015,.015),rng.uniform(.18,.38)]
        else:
            p = [rng.uniform(.105,.236),0.,0.,0.,0.,rng.uniform(.040,.128)]
        logged = rng.integers(N_ACTION)
        for a in range(N_ACTION) if branch else (logged,):
            path,touch = simulate(task,models[mi],p,a)
            states.append(i)
            acts.append(a)
            rows.append(path)
            contacts.append(touch)
            materials.append(mi)
    return dict(state=np.array(states),action=np.array(acts),path=np.array(rows),contact=np.array(contacts),
                material_index=np.array(materials))


def render(task,state):
    # Fixed orthographic camera. Neither material nor contact flags are drawn.
    scale = 113
    def xy(x,y):
        return (int(round(21+x*scale)),int(round(48-y*scale)))
    img = Image.new("RGB",(96,96),(237,239,236))
    pen = ImageDraw.Draw(img)
    if task=="push":
        x,y,a,tx,ty,tz,goal = state
        half=(BAND[task][1]-BAND[task][0])/2
        lo,hi=goal-half,goal+half
        pen.rectangle((xy(lo,.29),xy(hi,-.29)),fill=(211,230,210))
        poly=np.array([[-.055,.037],[.055,.037],[.055,-.001],[.019,-.001],[.019,-.036],
                       [-.019,-.036],[-.019,-.001],[-.055,-.001]])
        rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        pen.polygon([xy(u,v) for u,v in poly@rot.T+[x,y]],fill=(65,83,106))
        if tz<.095:
            pen.ellipse((xy(tx-.023,ty+.023),xy(tx+.023,ty-.023)),fill=(192,78,67))
    else:
        displacement,ram = state[:2]
        # Zoomed side camera to resolve the stroke of the spring.
        conv=lambda x: int(round((x-.055)*210))
        pen.rectangle((0,59,95,66),fill=(155,160,162))
        half=(BAND[task][1]-BAND[task][0])/2
        for x in (state[-1]-half,state[-1]+half):
            pen.line((conv(.34+x),25,conv(.34+x),58),fill=(81,141,91),width=1)
        x=conv(.34+displacement)
        pen.rectangle((x-5,34,x+5,57),fill=(61,90,132))
        x=conv(ram)
        pen.ellipse((x-6,40,x+6,52),fill=(190,79,64))
    return np.asarray(img)


def labels(task,data):
    coord = data["path"][:,:,0]
    half=(BAND[task][1]-BAND[task][0])/2
    mask=np.abs(coord-data["path"][:,:,-1])<=half
    return mask if task=="push" else mask&data["contact"]


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--calibration",action="store_true")
    ap.add_argument("--readout-check",action="store_true")
    args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        if args.readout_check:
            if task=="press":
                for domain in ("source","shift"):
                    np.savez_compressed(ROOT/f"press_check_{domain}.npz",**generate(task,domain,384,15202,True))
            continue
        if args.calibration:
            for part,n,seed in (("cal",64,14201),("fresh",256,14202)):
                for domain in ("source","shift"):
                    d=generate(task,domain,n,seed,True)
                    np.savez_compressed(ROOT/f"{task}_{part}_{domain}.npz",**d)
            continue
        if args.pilot:
            d=generate(task,"source",72,171,True)
            print(task,np.round(np.quantile(d["path"][:,-3:,0],[.1,.5,.9],axis=0),3),
                  labels(task,d)[:,-3:].all(1).reshape(-1,4).mean(0),flush=True)
            continue
        for name,domain,n,seed,branch in [("train","source",2048,13201,False),("val","source",192,13202,True),
                    ("source","source",384,13203,True),("shift","shift",384,13203,True)]:
            d=generate(task,domain,n,seed,branch)
            np.savez_compressed(ROOT/f"{task}_{name}.npz",**d)
            print(task,name,len(d["action"]),flush=True)
