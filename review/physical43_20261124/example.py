"""Make the two fixed Gear examples from registered native photographs."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robi_diagnostic import calibration, conservative_occlusion, grid, pose, project, read_depth, read_image
from robi_files import HERE, OBJECT, cached, selection
from surface_check import near_object, surface

if __name__=="__main__":
    assert OBJECT=="Gear", "set ROBI_OBJECT=Gear"
    row = next(r for r in selection() if r["scene"]==4 and r["sensor"]=="Ensenso")
    first,second = row["raw"]
    gt1,gt2,third = [read_depth(n,"Ensenso") for n in row["gt"]]
    k = calibration("Ensenso")
    x,y = grid(third)
    valid = third[y,x]>0
    x,y = x[valid],y[valid]
    z = third[y,x]
    pt = pose(row["gt"][2])
    tree,_ = surface()
    shapes = json.loads(cached("Gear/Scene_4/GT_world2obj.json").read_text()).values()
    hidden = near_object(x,y,z,k,pt,shapes,tree)
    projections = []
    for image,reference in ((first,gt1),(second,gt2)):
        u,v,d,inside = project(x,y,z,pt,pose(image),k,k)
        _,occ = conservative_occlusion(reference,u,v,d,inside)
        hidden &= occ
        projections.append((u,v,d,reference[v,u]))
    choose = np.flatnonzero(hidden)
    ix = choose[len(choose)//2]
    example = json.loads((HERE/"gear_surface.json").read_text())["rows"]
    anchor = next(r["example"] for r in example if r["sensor"]=="Ensenso" and r["scene"]==4 and r["initial"]==row["ids"][0])
    panels = [(row["ids"][0],anchor["x"],anchor["y"],"거리값이 틀린 점",
               f"기준 {anchor['reference_mm']:.1f}mm · 측정 {anchor['native_mm']:.1f}mm"),
              (row["ids"][2],x[ix],y[ix],"기준 시점에서 보인 점",f"대상 깊이 {z[ix]:.1f}mm")]
    records = []
    for i,(u,v,d,front) in enumerate(projections):
        panels.append((row["ids"][i],u[ix],v[ix],("이동 전 같은 점","이동 후 같은 점")[i],
                       f"앞 표면이 {d[ix]-front[ix]:.1f}mm 가까움"))
        records.append(dict(view=row["ids"][i],xy=[int(u[ix]),int(v[ix])],target_mm=float(d[ix]),front_mm=float(front[ix])))
    plt.rcParams["font.family"]="Malgun Gothic"
    fig,axes = plt.subplots(1,4,figsize=(13.5,3.9))
    sources = {}
    for i,(ax,(view,px,py,title,label)) in enumerate(zip(axes,panels)):
        name = f"Gear/Scene_4/Ensenso/Stereo/LEFT_view_{view}.bmp"
        sources[name] = hashlib.sha256(cached(name).read_bytes()).hexdigest()
        im = read_image(name)
        radius = 85
        xx,yy = int(np.clip(px,radius,im.shape[1]-radius)),int(np.clip(py,radius,im.shape[0]-radius))
        ax.imshow(im[yy-radius:yy+radius,xx-radius:xx+radius],cmap="gray",vmin=0,vmax=255)
        ax.plot(px-xx+radius,py-yy+radius,marker="o",markerfacecolor="none",markersize=13,
                markeredgewidth=1.9,color="#ffb000" if i==0 else "#46e6fb")
        ax.set_title(title,fontsize=13)
        ax.set_xlabel(f"시점 {view}  ·  {label}",fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=.012,right=.988,top=.91,bottom=.19,wspace=.12)
    fig.text(.015,.055,"ROBI · Yang et al. (IROS 2021), CC BY-NC-SA 4.0   |   실제 좌영상 확대 · 원은 평가점의 투영 위치",fontsize=9,color="#444444")
    fig.savefig(HERE/"gear_examples.png",dpi=145,bbox_inches="tight")
    (HERE/"gear_examples.json").write_text(json.dumps(dict(anchor=anchor,third_xy=[int(x[ix]),int(y[ix])],
                third_mm=float(z[ix]),views=records,image_sha256=sources),indent=2),encoding="utf8")
