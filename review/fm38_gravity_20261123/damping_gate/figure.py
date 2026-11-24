"""Plot matched control and damped test results from the primary gate."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
if "Malgun Gothic" in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False


def draw():
    data=json.loads((HERE/"results.json").read_text(encoding="utf8"))
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout="constrained")
    domains=("clean","seen","strong")
    styles=(("prior","기본 운동식","#747e85","o"),("flow","공개 Flow 보정","#9453a2","D"),
            ("drag","감쇠 계수 1개","#32779c","s"),("ridge","각도 오차 회귀","#147657","^"))
    for axis,metric,title,factor in zip(axes,("success","trajectory_MAE_deg"),
            ("선택한 힘의 실제 성공률 (%)","예측 궤적의 평균 각도 오차 (도)"),(100,1)):
        for name,label,color,marker in styles:
            yy=[factor*data["metrics"][d][name][metric] for d in domains]
            axis.plot(range(3),yy,marker=marker,lw=2.2,color=color,label=label)
        if metric=="success":
            axis.plot(range(3),[100*data["controls"][d]["oracle_success"] for d in domains],ls=":",
                      color="#393939",label="도달 가능한 상한")
        axis.set_xticks(range(3),["감쇠 없음","익숙한 감쇠","강한 감쇠"])
        axis.set_title(title,loc="left",fontweight="bold")
        axis.set_ylim(bottom=0)
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y",alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=5,frameon=False,fontsize=9)
    fig.savefig(HERE/"damping_results.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    draw()
