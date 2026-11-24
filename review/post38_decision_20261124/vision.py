"""Real near/far-view counterfactuals and one simulated metric point."""
import json
import re
from zipfile import ZipFile

import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import HERE, CACHE, SCENES

PRICES = (.0025,.005,.01,.02)
SCALES = np.array([.8,1.,1.2])
COST = np.array([0.,1.,3.])
ARMS = ("keep","point","view")


def stereo(a,b,fraction):
    disparities = int(16*np.ceil(86*fraction/16))
    model = cv2.StereoSGBM_create(minDisparity=0,numDisparities=disparities,blockSize=5,
        P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=8,speckleWindowSize=40,speckleRange=2)
    left = model.compute(a,b).astype(float)/16
    right = model.compute(np.ascontiguousarray(b[:,::-1]),np.ascontiguousarray(a[:,::-1]))[:,::-1]/16
    y,x = np.indices(left.shape)
    xr = np.rint(x-left).astype(int)
    valid = (left>=0)&(xr>=0)&(xr<left.shape[1])
    rd = right[y,np.clip(xr,0,left.shape[1]-1)]
    lr = np.where(valid&(rd>=0),np.abs(left-rd),np.inf)
    return left,valid,lr


def absrel(ratio):
    ordered = np.sort(ratio)
    prefix = np.r_[0.,np.cumsum(ordered)]
    def loss(scale):
        scale = np.asarray(scale)
        k = np.searchsorted(ordered,1/scale)
        return (2*k-len(ordered)+scale*(prefix[-1]-2*prefix[k]))/len(ordered)
    return loss


def arms_for_scene(year,scene,idx):
    with ZipFile(CACHE/(scene+".zip")) as z:
        def image(name):
            return cv2.imdecode(np.frombuffer(z.read(f"{scene}/{name}.png"),np.uint8),cv2.IMREAD_GRAYSCALE)
        images = {i:image(f"view{i}") for i in (1,2,3,5)}
        gt = image("disp1")
        offset = float(z.read(scene+"/dmin.txt"))/3
    ref = images[1]
    truth = gt.astype(float)/3+offset
    y,x = np.indices(gt.shape)
    lattice = (x%4==0)&(y%4==0)
    good_gt = (gt>0)&(truth>0)
    far,far_valid,_ = stereo(ref,images[5],1.)
    far_valid &= far+offset>0
    texture = np.hypot(cv2.Sobel(ref,cv2.CV_32F,1,0),cv2.Sobel(ref,cv2.CV_32F,0,1))
    rows = []
    for view in (2,3):
        fraction = (view-1)/4
        for blur in (0.,2.5):
            other = images[view] if blur==0 else cv2.GaussianBlur(images[view],(0,0),blur)
            d,valid,lr = stereo(ref,other,fraction)
            denom = d/fraction+offset
            valid &= denom>0
            eval_set = good_gt&valid&~lattice
            ratio = np.divide(truth,denom,out=np.ones_like(denom),where=valid)
            base = absrel(ratio[eval_set])
            alt = ratio.copy()
            np.divide(truth,far+offset,out=alt,where=far_valid)
            far_loss = absrel(alt[eval_set])
            points = np.flatnonzero(lattice&valid)
            ranked = np.lexsort((-texture.flat[points],lr.flat[points]))
            pool = points[ranked[:max(1,int(np.ceil(.1*len(points))))]]
            rng = np.random.default_rng([861702,year,idx,view,int(blur*2)])
            hits = rng.choice(pool,24)
            obs_ok = good_gt.flat[hits]
            corr = np.exp(rng.normal(0,.02,24))/ratio.flat[hits]
            scores = np.stack([base(SCALES),
                    [np.mean(base(np.where(obs_ok,corr,s))) for s in SCALES],far_loss(SCALES)],axis=1)
            # All deployment features use only the two initial pictures and relative rail geometry.
            features = [fraction,valid.mean(),(lr[valid]<=.5).mean(),np.median(d[valid])/fraction,
                        np.subtract(*np.percentile(d[valid],[75,25]))/fraction,
                        np.log1p(cv2.Laplacian(other,cv2.CV_64F).var()),
                        np.log1p(cv2.Laplacian(ref,cv2.CV_64F).var())]
            assert np.allclose(base(1.),np.abs(ratio[eval_set]-1).mean())
            rows.append(dict(scene=scene,family=re.sub(r"\d+$","",scene),year=year,near=view,blur=blur,
                     features=list(map(float,features)),loss=scores.tolist(),anchor_hits=int(obs_ok.sum()),
                     coverage=float(eval_set.sum()/(good_gt&~lattice).sum()),far_fallback=float((~far_valid[eval_set]).mean()),
                     median_near_disp_error=float(np.median(np.abs(d[eval_set]/fraction-gt[eval_set]/3)))))
    return rows


MODELS = {"ridge":(.01,.1,1.,10.),"local":(2,4,8,12)}


def predict(kind,value,train_x,train_y,test_x):
    if kind=="fixed":
        return np.tile(train_y.mean(axis=0),(len(test_x),1))
    if kind=="baseline_length":
        return np.array([train_y[train_x[:,0]==x[0]].mean(axis=0) for x in test_x])
    predictor = Ridge(alpha=value) if kind=="ridge" else KNeighborsRegressor(n_neighbors=value,weights="distance")
    return make_pipeline(StandardScaler(),predictor).fit(train_x,train_y).predict(test_x)


def choose_hyper(kind,x,y,scene,price):
    scores = []
    for value in MODELS[kind]:
        by_scene = []
        for s in np.unique(scene):
            hold = scene==s
            actions = (predict(kind,value,x[~hold],y[~hold],x[hold])+price*COST).argmin(1)
            by_scene.append(np.mean(y[hold][np.arange(hold.sum()),actions]+price*COST[actions]))
        scores.append(np.mean(by_scene))
    return MODELS[kind][np.argmin(scores)]


def summarize(rows):
    x = np.array([r["features"] for r in rows])
    losses = np.array([r["loss"] for r in rows])
    y = losses.mean(axis=1)
    train = np.array([r["year"]==2005 for r in rows])
    scene = np.array([r["scene"] for r in rows])
    family = np.array([r["family"] for r in rows])[~train]
    families = sorted(set(family))
    def clusters(values):
        return np.array([values[family==f].mean(axis=0) for f in families])
    rng = np.random.default_rng(31682)
    boot = rng.integers(len(families),size=(10000,len(families)))
    out = {}
    for price in PRICES:
        dl = y[~train]
        J = dl+price*COST
        ix = np.arange(len(J))
        by_method,block_scores,parameters = {},{},{}
        chosen = {}
        for kind in ("always_keep","always_point","always_view","fixed","baseline_length","ridge","local","cell_oracle"):
            if kind.startswith("always_"):
                a = np.full(len(ix),ARMS.index(kind[7:]))
            elif kind=="cell_oracle":
                a = J.argmin(1)
            else:
                knob = choose_hyper(kind,x[train],y[train],scene[train],price) if kind in MODELS else None
                parameters[kind]=knob
                a = (predict(kind,knob,x[train],y[train],x[~train])+price*COST).argmin(1)
            chosen[kind]=a.tolist()
            block_scores[kind]=clusters(J[ix,a])
            by_method[kind]=dict(J=float(block_scores[kind].mean()),AbsRel=float(clusters(dl[ix,a]).mean()),
                      cost=float(clusters(COST[a]).mean()),actions=clusters(np.eye(3)[a]).mean(0).tolist())
        by_method["scale_oracle"]={"J":float(clusters((losses[~train]+price*COST).min(2).mean(1)).mean())}
        freq = by_method["local"]["actions"]
        by_method["matched_random"]={"J":float(clusters(J@freq).mean()),"cost":float(COST@freq)}
        baselines=("fixed","baseline_length","ridge","always_keep","always_point","always_view")
        paired = {}
        for kind in baselines:
            delta=block_scores[kind]-block_scores["local"]
            paired[kind]=dict(relative=float(delta.mean()/block_scores[kind].mean()),
                              ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist())
        out[str(price)]=dict(methods=by_method,choices=chosen,params=parameters,paired=paired,families=families)
    return out


if __name__=="__main__":
    cv2.setNumThreads(1)
    rows = [row for year,names in SCENES.items() for i,s in enumerate(names) for row in arms_for_scene(year,s,i)]
    result = dict(arms=ARMS,scales=SCALES.tolist(),rows=rows,summary=summarize(rows),opencv=cv2.__version__)
    (HERE/"vision.json").write_text(json.dumps(result,indent=1),encoding="utf8")
    print(json.dumps(result["summary"]["0.01"]["methods"],indent=2))
