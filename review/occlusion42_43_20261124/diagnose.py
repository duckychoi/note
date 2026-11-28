"""A privileged synthetic cue-readability diagnostic and endpoint quality illustrations."""
import cv2
import numpy as np

from common import ROOT,STORE,SEEDS,TEST,save
from junctions import render


def front_choice(im,axes):
    cx,cy,a,b=axes
    def sample(x,y):
        return cv2.getRectSubPix(im.astype(np.float32),(1,1),(float(x),float(y))).ravel()
    center=sample(cx,cy)
    costs=[]
    for angle in (a,b):
        distance=[np.mean(abs(center-sample(cx+u*np.cos(angle),cy+u*np.sin(angle))))
                  for u in (-14,-12,12,14)]
        costs.append(np.mean(distance))
    return np.sign(costs[1]-costs[0])  # positive = A is continuous at the center


def sanity():
    rec=[]
    for seed in SEEDS:
        for base in range(8):
            states=[]
            for state in (0,1):
                fr,_,_,axes=render(seed,22,base,state,True,return_axes=True)
                im,z,q=fr[0]
                signed=[]
                for number,axis in enumerate(axes):
                    ids=q[number*24:number*24+12,:2]
                    target=z.ravel()[ids[:,1]]-z.ravel()[ids[:,0]]
                    assert len(target)==12 and (np.sign(target)==np.sign(target[0])).all()
                    pred=front_choice(im,axis)
                    signed.append((float(pred),float(np.sign(target[0]))))
                states.append(signed)
            a=np.array(states)
            assert np.array_equal(a[0,:,1],-a[1,:,1])
            good=a[:,:,0]*a[:,:,1]
            rec.append(dict(seed=seed,base=base,n=4,order=float(((good>0)+.5*(good==0)).mean()),
                        both=float((good>0).all(0).mean()),tie=float((good==0).mean())))
    save("cue_readability.json",rec)
    print("Synthetic supplied-axis continuity diagnostic",np.mean([[r["order"],r["both"]] for r in rec],axis=0))


def sheet():
    cells=[]
    for scene in TEST:
        if not scene.endswith("1"):
            continue
        with np.load(STORE/(scene+"_wide.npz")) as d:
            im=cv2.cvtColor(d["rgb"],cv2.COLOR_RGB2BGR)
            _,w=im.shape[:2]
            tri=d["triples"]
            for t in tri[np.linspace(0,len(tri)-1,min(30,len(tri)),dtype=int)]:
                for i,idx in enumerate(t):
                    color=(0,255,255) if i==0 else (255,255,0)
                    cv2.circle(im,(int(idx%w),int(idx//w)),1,color,-1)
            scale=min(300/im.shape[1],185/im.shape[0])
            im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            canvas=np.full((212,300,3),32,np.uint8)
            canvas[:im.shape[0],:im.shape[1]]=im
            cv2.putText(canvas,scene,(5,205),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1,cv2.LINE_AA)
            cells.append(canvas)
    cv2.imencode(".jpg",np.vstack([np.hstack(cells[i:i+3]) for i in range(0,9,3)]))[1].tofile(ROOT/"real_endpoints.jpg")


if __name__=="__main__":
    sanity()
    sheet()
