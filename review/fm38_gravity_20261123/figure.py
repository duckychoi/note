"""Show the first rightward peaks of two matched swings from the simulator."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from swing import HERE

if "Malgun Gothic" in {x.name for x in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False


def xy(angle,r=1):
    return r*np.sin(angle),-r*np.cos(angle)


def draw():
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout="constrained")
    palette={"pass":"#0b7758","fail":"#a15238","ink":"#253645"}
    with np.load(HERE/"outcomes.npz") as d:
        s=485
        goal=d["goal_source"][s]
        segment=np.linspace(goal-np.deg2rad(7.5),goal+np.deg2rad(7.5),70)
        for ax,dm,title in zip(axes,("source","shift"),("약하게 되돌아오는 그네","강하게 되돌아오는 그네")):
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-.20,1.76)
            ax.set_ylim(-1.17,.20)
            ax.set_title(title,fontsize=15,weight="bold",loc="left",color=palette["ink"])
            arc=np.linspace(0,np.deg2rad(99),100)
            ax.plot(*xy(arc),color="#c4ccd0",lw=1.3)
            ax.plot(*xy(segment),lw=14,color="#acdbc7",solid_capstyle="butt")
            ax.plot([0,0],[0,-1],color="#89939c",lw=1.1,ls="--")
            ax.scatter([0],[0],s=65,color=palette["ink"],zorder=4)
            ax.text(-.06,-1.10,"출발",ha="center",size=10)
            for a in (1,2):
                q=d["peak_"+dm][s,a]
                x,y=xy(q)
                ok=abs(q-goal)<=np.deg2rad(7.5)
                color=palette["pass" if ok else "fail"]
                ax.plot([0,x],[0,y],color=color,lw=2.6)
                ax.scatter([x],[y],s=85,color=color,zorder=3,edgecolor="white",linewidth=1)
                ax.text(x+.09,y+(0.07 if a==2 else -.04),f"{a+1}단  {np.rad2deg(q):.0f}°  {'성공' if ok else '실패'}",
                        color=color,fontsize=12,weight="bold",va="center")
    fig.suptitle("같은 목표 60°  ·  초록 띠 안에 최고점이 들어오면 성공",fontsize=17,weight="bold",color=palette["ink"])
    fig.savefig(HERE/"swing_example.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    draw()
