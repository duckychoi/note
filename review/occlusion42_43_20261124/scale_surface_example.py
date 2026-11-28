"""Render the previously selected Books RGB-only example on one common metric scale."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[3]/"research-sources/encoder42_eval"
    folder=root/"repeat-capture/physical/Books"
    out=root/"scale-surface-rgb"
    predicted=np.load(out/"Books-rgb-depth-m.npy")
    h,w=predicted.shape
    rgb=np.array(Image.open(folder/"Illum1/Exp1/view1.png").convert("RGB").resize((w,h),Image.Resampling.BILINEAR))
    disp=np.array(Image.open(folder/"disp1.png").resize((w,h),Image.Resampling.NEAREST),float)
    truth=np.where(disp>0,3740*.16/(disp+float((folder/"dmin.txt").read_text())),np.nan)
    with np.load(root/"repeat-capture/encoder/Books.npz") as data:
        base=1/data["inverse"]
    fig, ax=plt.subplots(1,4,figsize=(12,3.6))
    fig.subplots_adjust(left=.012,right=.988,top=.88,bottom=.22,wspace=.035)
    ax[0].imshow(rgb)
    cmap=plt.colormaps["viridis"].with_extremes(bad="#c9c9c9")
    for panel,depth in zip(ax[1:],(truth,base,predicted)):
        sm=panel.imshow(depth,cmap=cmap,vmin=0,vmax=7,interpolation="nearest")
    for panel,title in zip(ax,("Single RGB input","Measured depth","Frozen Depth Pro 2-S","RGB-only correction")):
        panel.set_title(title,fontsize=11)
        panel.set_axis_off()
    bar=fig.add_axes((.36,.13,.47,.032))
    fig.colorbar(sm,cax=bar,orientation="horizontal",
                 label="Optical-axis depth (m)   |   gray = missing ground truth")
    fig.savefig(out/"Books-depth-comparison.png",dpi=155,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
