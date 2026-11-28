"""One fixed public scene illustrating opposite directions and RGB abstention."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

def main():
    root=Path(__file__).resolve().parents[3]/"research-sources/encoder42_eval"
    out=root/"native-surface-direction"
    scene="Books"
    src=root/"repeat-capture/physical"/scene
    rgb=np.array(Image.open(src/"Illum1/Exp1/view1.png").convert("RGB"))
    A=3740*.16/3;offset=float((src/"dmin.txt").read_text())/3
    with np.load(out/"tap-probe/result.npz") as obs,np.load(out/"middlebury-native"/(scene+".npz")) as student:
        x,y=obs[scene+"/query_xy"].astype(int).T
        rho=1/(float(student["k"])*student["base"][y,x])
        prefix=f"{scene}/L1-L1/view5/"
        truth=np.sign(obs[prefix+"gt_inverse_depth_per_m"]-rho)
        eligible=obs[prefix+"gt_eligible"].astype(bool)&(truth!=0)
        answers=[]
        for light in (1,2):
            p=f"{scene}/L1-L{light}/view5/"
            diff=x-obs[p+"tap_xy"][:,0]+offset-A*rho
            answers.append(np.where((abs(diff)>1)&obs[p+"tap_visible"],np.sign(diff),0))
        tap=np.where(answers[0]==answers[1],answers[0],0)
        rgb_only=-np.sign(student["h"][y,x])
    plt.rcParams["font.family"]="Malgun Gothic"
    fig,axs=plt.subplots(1,3,figsize=(11,3.75))
    fig.subplots_adjust(left=.012,right=.988,top=.87,bottom=.13,wspace=.025)
    for ax,field,title in zip(axs,(truth,tap,rgb_only),("측정 깊이가 요구한 방향","TAPNext++ 두 조명 합의","단안 학생의 보정")):
        ax.imshow(rgb)
        for value,color,mark in ((1,"#ff9e2c","o"),(-1,"#00c8ed","^"),(0,"white","x")):
            m=eligible & (field==value)
            kwargs=dict(edgecolors="#192434",linewidths=.5) if value else dict(linewidths=.7)
            ax.scatter(x[m],y[m],c=color,s=17 if value else 10,marker=mark,**kwargs)
        ax.set_axis_off();ax.set_title(title,fontsize=12)
    handles=[Line2D([],[],linestyle="",marker=m,color=c,markeredgecolor=e,label=l)
             for m,c,e,l in (("o","#ff9e2c","#192434","더 가깝게"),("^","#00c8ed","#192434","더 멀게"),("x","#787878","#787878","보정 보류"))]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False)
    fig.savefig(out/"Books-native-directions.png",dpi=145,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    main()
