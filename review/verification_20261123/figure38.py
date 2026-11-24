"""Plot the matched-marginal control experiment with paired, test-group uncertainty."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent


def main():
    for name in ("Malgun Gothic","Noto Sans CJK KR","NanumGothic"):
        if any(f.name==name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    data = json.loads((HERE/"action38.json").read_text())["results"]
    fig,ax = plt.subplots(figsize=(8.2,3.65))
    methods = ("flow","mdn","nearest32")
    for material,label,color,offset in [("source","학습과 같은 물성","#236B59",.1),("new","새 물성","#B55037",-.1)]:
        for i,m in enumerate(methods):
            group = [r for r in data if r["materials"]==material]
            by_test = []
            for s in sorted({r["test_seed"] for r in group}):
                paired = [r["methods"][m]["joint"]["value"]-r["methods"][m]["independent"]["value"]
                          for r in group if r["test_seed"]==s]
                by_test.append(np.mean(paired))
            half = t.ppf(.975,len(by_test)-1)*np.std(by_test,ddof=1)/np.sqrt(len(by_test))
            ax.errorbar(np.mean(by_test),2-i+offset,xerr=half,marker="o",capsize=3,color=color,linewidth=2,
                        label=label if i==0 else None)
    ax.axvline(0,color="#454545",linewidth=1)
    ax.set(yticks=[2,1,0],yticklabels=["조건부 flow","혼합 신경망","시범 검색"],ylim=(-.5,2.55),
           xlim=(-.04,.05),xlabel="공동 경로로 계산한 행동 점수에서 프레임 독립으로 계산한 점수를 뺀 값")
    ax.set_title("같은 장면별 확률도 물성이 바뀌면 행동 선택에 다르게 작용한다",loc="left",fontsize=14,pad=14)
    ax.grid(axis="x",alpha=.16)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.legend(frameon=False,loc="upper right",fontsize=9)
    fig.text(.13,.025,"점은 평균, 막대는 같은 시험 상태를 짝지은 95% t구간. 3개 학습 집합의 평균을 시험 묶음 8개에서 비교했다.",fontsize=8)
    fig.subplots_adjust(left=.17,right=.97,bottom=.23,top=.83)
    fig.savefig(HERE/"action38.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
