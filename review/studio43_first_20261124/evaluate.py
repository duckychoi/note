"""Three fixed endpoints: measured depth, video bokeh proxy, and optical photo edges."""
from collections import defaultdict
import json
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from data import CACHE, DEV, HERE, load_image
from studio import base_depth, real_frame
from train import ARMS, SEEDS, DepthNet, infer, tensor_input


def luminance(rgb):
    s=rgb.astype(np.float32)/255
    linear=np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)
    return linear@np.array([.2126,.7152,.0722],np.float32)


LEVELS=np.array([-16,-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12,16])


def disk(radius):
    y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
    ker=np.float32(x*x+y*y<=radius*radius)
    return ker/ker.sum()


KERNELS={int(abs(v)):disk(int(abs(v))) for v in LEVELS}


def blur(gray,z,fx,lens,focus):
    radius=.5*lens*fx*focus/(2*(focus-lens))*(1/np.maximum(z,.1)-1/focus)
    bins=np.digitize(radius,(LEVELS[:-1]+LEVELS[1:])/2)
    out=gray.copy()
    for i,level in enumerate(LEVELS):
        selected=bins==i
        if not selected.any():
            continue
        if level==0:
            out[selected]=gray[selected]
            continue
        alpha=selected.astype(np.float32)
        kernel=KERNELS[abs(level)]
        out=out*(1-cv2.filter2D(alpha,-1,kernel))+cv2.filter2D(gray*alpha,-1,kernel)
    return out


def highpass(gray):
    return gray-cv2.GaussianBlur(gray,(0,0),1.15)


def metric_setup(row,image,gt,fx):
    gray=luminance(image)
    ker=np.ones((3,3),np.uint8)
    if row["kind"]=="eth":
        focus,lens=1.5,.05
        valid=(gt>=.2)&(gt<=10)
        assert valid.any()
        holes=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        filled=gt[tuple(holes)]
        # Valid edge endpoints only; do not define an edge from a zero-depth hole.
        jump=np.zeros(valid.shape,bool)
        for axis in (0,1):
            sl=[slice(None)]*2
            sl[axis]=slice(1,None)
            sr=[slice(None)]*2
            sr[axis]=slice(None,-1)
            a,b=tuple(sl),tuple(sr)
            strong=valid[a]&valid[b]&(np.abs(gt[a]-gt[b])>.12)
            jump[a]|=strong
            jump[b]|=strong
        band=(cv2.dilate(jump.astype(np.uint8),ker,iterations=2)>0)&valid
        band&=cv2.blur(valid.astype(np.float32),(5,5))>=.72
        target=blur(gray,filled,fx,lens,focus)
    else:
        valid=None
        focus,lens=row["focus"],row["lens_mm"]/1000
        target=luminance(cv2.resize(load_image(row["target"]),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_AREA))
        # Remove global exposure difference once for all depth methods.
        target=target*(gray.mean()/max(float(target.mean()),1e-6))
        energy=cv2.blur(highpass(gray)**2,(7,7))
        energy_target=cv2.blur(highpass(target)**2,(7,7))
        in_focus=np.uint8(energy_target>=.25*(energy+1e-8))
        in_focus=cv2.morphologyEx(in_focus,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        band=(cv2.dilate(in_focus,ker,iterations=3)!=cv2.erode(in_focus,ker,iterations=3))&(energy>1e-6)
    band[:3]=False
    band[-3:]=False
    band[:,:3]=False
    band[:,-3:]=False
    hp=highpass(target)
    def evaluate(z):
        rendered=blur(gray,z,fx,lens,focus)
        result={}
        if valid is not None:
            result["depth_mm"]=float(1000*np.abs(z[valid]-gt[valid]).mean())
            result["absrel"]=float(np.mean(np.abs(z[valid]-gt[valid])/gt[valid]))
        if band.sum()>=32:
            key="video_edge" if valid is not None else "optical_edge"
            result[key]=float(1000*np.abs(highpass(rendered)[band]-hp[band]).mean())
        return result
    return evaluate,int(band.sum())


def run():
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    ledger=json.loads((HERE/"training.json").read_text())
    assert len(ledger)==2*len(SEEDS)*len(ARMS)
    checkpoints={}
    for line in ledger:
        name=line["checkpoint"]
        if name not in checkpoints:
            model=DepthNet(line["model"]=="adapter").cuda()
            model.load_state_dict(torch.load(CACHE/"checkpoints"/name,weights_only=True))
            checkpoints[name]=model.eval()
    inputs=json.loads((HERE/"inputs.json").read_text())["rows"]
    output=[]
    for row in inputs:
        if row["scene"] in DEV:
            continue
        im,gt,fx=real_frame(row)
        metric,n=metric_setup(row,im,gt,fx)
        base=base_depth(row["key"],im,fx)
        x={"adapter":tensor_input(im,fx,base),"scratch":tensor_input(im,fx)}
        pred={name:infer(model,x["adapter" if model.prior else "scratch"]) for name,model in checkpoints.items()}
        scores={"H+":metric(base)}
        for line in ledger:
            key=f"{line['model']}/{line['seed']}/{line['arm']}"
            z=np.exp(pred[line["checkpoint"]]+line["log_gain"])
            scores[key]=metric(z)
        output.append(dict(scene=row["scene"],key=row["key"],kind=row["kind"],edge_pixels=n,scores=scores))
    (HERE/"outcomes.json").write_text(json.dumps(output,separators=(",",":")),encoding="utf8")


def summarize():
    rows=json.loads((HERE/"outcomes.json").read_text())
    result={}
    for metric in ("depth_mm","video_edge","optical_edge","absrel"):
        groups=defaultdict(list)
        for r in rows:
            if metric in r["scores"]["H+"]:
                groups[r["scene"]].append(r)
        scenes=sorted(groups)
        reduced={k:np.array([np.mean([r["scores"][k][metric] for r in groups[s]]) for s in scenes])
                 for k in next(iter(groups.values()))[0]["scores"]}
        item={"units":scenes,"frames":[len(groups[s]) for s in scenes],"H+":reduced["H+"].mean(),"methods":{},"paired":{}}
        resample=np.random.default_rng(43251).integers(0,len(scenes),(20000,len(scenes)))
        for model in ("scratch","adapter"):
            arms={}
            for arm in ARMS:
                vals=np.stack([reduced[f"{model}/{s}/{arm}"] for s in SEEDS])
                arms[arm]=vals
                item["methods"][model+"/"+arm]=dict(mean=vals.mean(),by_seed=vals.mean(1).tolist(),by_unit=vals.mean(0).tolist())
            for candidate,baseline in (("few_guided","few_random"),("zero_synthetic","zero_random"),("few_random","zero_random")):
                paired=arms[candidate]-arms[baseline]
                blocks=paired.mean(0)
                b=blocks[resample].mean(1)
                item["paired"][model+"/"+candidate+"-"+baseline]=dict(delta=blocks.mean(),
                      upper95=float(np.quantile(b,.95)),two_sided95=np.quantile(b,[.025,.975]).tolist(),
                      wins=int((blocks<0).sum()),by_seed=paired.mean(1).tolist())
        result[metric]=item
    (HERE/"summary.json").write_text(json.dumps(result,indent=1),encoding="utf8")


if __name__=="__main__":
    run()
    summarize()
