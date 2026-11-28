"""Show one previously used physical scene and its RGB-only trust decisions."""
from PIL import Image,ImageDraw,ImageFont
import numpy as np

from light_encoder_score import middlebury_pairs
from light_encoder_study import Model
from material_light_assets import OUT
from material_light_depth import depth
from material_light_test import candidate,image_features


def main():
    for name,_,a,b,gt,mask in middlebury_pairs():
        if name=="Books":
            break
    else:
        raise RuntimeError("Books not in equal-exposure cohort")
    model=Model()
    da,db=depth(model,a),depth(model,b)
    gt,mask=gt.cpu().numpy(),mask.cpu().numpy()
    bins,side=candidate(image_features(a),image_features(b))
    def overlay(colors):
        bg=.38*(a.astype(float)+b.astype(float))/2
        for take,col in colors:
            bg[take]=.22*bg[take]+.78*np.array(col)
        return Image.fromarray(bg.astype(np.uint8))
    red=(246,127,67)
    blue=(73,187,242)
    # Panel 4 excludes negligible (<1 percentage point) absolute-relative differences.
    diff=(abs(db-gt)-abs(da-gt))/gt
    panels=[Image.fromarray(a),Image.fromarray(b),
            overlay([((side<0)&(bins>0),red),((side>0)&(bins>0),blue)]),
            overlay([(mask&(diff>.01),red),(mask&(diff<-.01),blue)])]
    w=380
    h=round(w*a.shape[0]/a.shape[1])
    out=Image.new("RGB",(2*w+24,2*h+128),"#f6f6f2")
    draw=ImageDraw.Draw(out)
    font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",16)
    title=ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf",21)
    draw.text((8,5),"Middlebury Books · 같은 카메라와 셔터에서 바꾼 두 조명",font=title,fill="#20262b")
    for i,(pic,label) in enumerate(zip(panels,("조명 A","조명 B","RGB 규칙이 더 믿은 사진","정답에 더 가까웠던 사진"))):
        x=8+(i%2)*(w+8)
        y=39+(i//2)*(h+31)
        draw.text((x,y),label,font=font,fill="#20262b")
        out.paste(pic.resize((w,h),Image.Resampling.LANCZOS),(x,y+24))
    draw.text((8,2*h+101),"주황 = A     파랑 = B     어두운 부분 = 기권 또는 오른쪽에서는 정답 무효·오차 차이 0.01 이하",font=font,fill="#20262b")
    out.save(OUT/"middlebury-books.jpg",quality=92)


if __name__=="__main__":
    main()
