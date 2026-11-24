"""Plot a paired physical counterfactual from each task using the saved solver positions."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from worlds import ROOT,BAND,labels

HERE=Path(__file__).resolve().parent
if "Malgun Gothic" in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False
results=json.loads((HERE/"results.json").read_text(encoding="utf8"))
fig,axes=plt.subplots(2,2,figsize=(9,5.8),sharex=True,sharey="row",layout="constrained")
times=np.arange(6)*.08
for row,(task,label,unit) in enumerate((("push","T자 물체 밀기","물체 중심 위치 (cm)"),
                                      ("press","스프링 버튼 누르기","버튼 눌림 깊이 (cm)"))):
    ex=results[task]["physical"]["example"]
    state=ex["state"]
    for col,(domain,title) in enumerate((("source","원 물성"),("shift","변경 물성"))):
        ax=axes[row,col]
        with np.load(ROOT/f"{task}_{domain}.npz") as d:
            ax.set_title(f"{label} · {title}",loc="left",fontsize=12,weight="bold")
            half=(BAND[task][1]-BAND[task][0])/2
            goal=d["path"][4*state,0,-1]
            ax.axhspan(100*(goal-half),100*(goal+half),color="#cfe6d8",label="목표 범위")
            ax.axvspan(.237,.403,color="#8c9bb0",alpha=.10)
            for a,c,sym in zip(ex["actions"],("#246a96","#b35433"),("o","D")):
                i=4*state+a-1
                success=labels(task,d)[i,-3:].all()
                desc=("성공" if success else "실패")
                value=d["path"][i,:,0]*100
                ax.plot(times,value,color=c,lw=2.1,marker=sym,markersize=4,label=f"행동 {a}  {desc}")
            ax.spines[["right","top"]].set_visible(False)
            ax.grid(axis="y",alpha=.16)
            ax.legend(frameon=False,fontsize=9,loc="upper left")
            ax.set_xticks(times)
            if row==1: ax.set_xlabel("행동 시작 후 시간 (초)")
            if col==0: ax.set_ylabel(unit)
fig.suptitle("같은 장면에서 물성만 바꾸자 성공하는 행동이 바뀌었다",weight="bold",fontsize=16)
fig.savefig(HERE/"physical_reversals.png",dpi=165)
