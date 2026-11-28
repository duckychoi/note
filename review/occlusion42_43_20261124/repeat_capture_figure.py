"""Display matched physical and rendered views of the held-out Books surface."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

from repeat_capture_acquire import OUT


def main(smooth=False):
    font=font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
    figure,axes=plt.subplots(3,4,figsize=(10.8,6.9),layout="constrained")
    scene="Books"
    with np.load(OUT/("renders-plane" if smooth else "renders")/(scene+".npz")) as data:
        frames=data["frames"][:,0]
    for col,(v,l) in enumerate(((3,1),(3,2),(5,1),(5,2))):
        axes[0,col].imshow(Image.open(OUT/f"physical/{scene}/Illum{l}/Exp1/view{v}.png"))
        axes[0,col].set_title(f"시점 {v} · 조명 {l}",fontproperties=font,fontsize=12)
        for material in range(2):
            linear=frames[material,int(v==5),l-1]/65535
            display=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
            axes[material+1,col].imshow(display,cmap="gray",vmin=0,vmax=1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    for row,title in enumerate(("공개 실사","합성 · 확산","합성 · 확산+반사")):
        axes[row,0].set_ylabel(title,fontproperties=font,fontsize=11)
    title="Middlebury Books  |  두 시점과 두 조명"
    if smooth:
        title+=" · 국소 평면 법선"
    figure.suptitle(title,fontproperties=font,fontsize=16)
    figure.savefig(OUT/("sample-books-plane.png" if smooth else "sample-books.png"),dpi=125)
    plt.close(figure)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-normals",action="store_true")
    main(parser.parse_args().smooth_normals)
