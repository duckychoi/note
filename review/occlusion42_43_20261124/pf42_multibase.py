"""Matched multi-baseline and repeated stereo teachers on Middlebury photographs."""
import itertools
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from parallax42_data import read_rgb, REAL, FOCAL_BASELINE


ROOT = REAL.parents[1] / "pf42-trials"
OUT = ROOT / "multibase-b"
RATIOS = np.arange(1, 6)/4
K = FOCAL_BASELINE*RATIOS
LOW, HIGH = 1/30, 4.
MINIMUM = -16
SIGMAS = (.2, .4, .8)
TAUS = (.2, .5, 1.)
OUTLIERS = (.05, .15)
PHI_CONST = 1/np.sqrt(2*np.pi)


def log_median(values, axis=1):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(values, axis=axis)


def stereo(first, second, search):
    engine = cv2.StereoSGBM_create(minDisparity=MINIMUM, numDisparities=search, blockSize=5,
                                   P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=12,
                                   speckleWindowSize=80, speckleRange=2)
    a = engine.compute(first, second).astype(np.float32)/16
    b = engine.compute(second[:, ::-1].copy(), first[:, ::-1].copy())[:, ::-1].astype(np.float32)/16
    y, x = np.indices(a.shape)
    partner = np.rint(x-a).astype(np.int32)
    other = b[y, partner.clip(0, a.shape[1]-1)]
    keep = (a >= MINIMUM)&(other >= MINIMUM)&(abs(a-other) <= 1)&(partner >= 0)&(partner < a.shape[1])
    return np.where(keep, a, np.nan)


def measure():
    cv2.setNumThreads(1)
    partitions = {}
    for name in ("cal", "held", "fit"):
        with np.load(ROOT / f"{name}.npz", allow_pickle=False) as old:
            partitions[name] = {k: old[k] for k in ("y", "truth", "group", "scene")}
    cache = {}
    for name, refs in partitions.items():
        entries = []
        for scene in dict.fromkeys(refs["scene"]):
            path = REAL / scene
            dmin = float((path/"dmin.txt").read_text())/3
            measurements = []
            for light, exp in itertools.product((1, 2), repeat=2):
                folder = path / f"Illum{light}" / f"Exp{exp}"
                def gray(view):
                    return cv2.cvtColor(read_rgb(folder / f"view{view}.png"), cv2.COLOR_RGB2GRAY)
                first = gray(1)
                h, w = first.shape
                yy, xx = np.mgrid[8:h:16, 8:w:16].reshape(2, -1)
                measurements.append([stereo(first, gray(j), 128 if j == 6 else 112)[yy,xx]
                                     for j in range(2, 7)])
            raw = np.array(measurements).transpose(2, 1, 0)
            long = FOCAL_BASELINE/np.maximum(raw[:, 3]+dmin, .001)
            keep = (long >= .25)&(long <= 30)
            row = keep.sum(axis=1) >= 2
            reference = refs["y"][refs["scene"] == scene]
            np.testing.assert_allclose(np.where(keep,long,np.nan)[row], reference, atol=1e-7, equal_nan=True)
            entries.append((raw[row].astype(float) + (dmin*RATIOS)[None,:,None], yy[row], xx[row]))
            print("camera views", name, scene, int(row.sum()), flush=True)
        data = {**refs, "d": np.concatenate([r[0] for r in entries]),
                "row": np.concatenate([r[1] for r in entries]), "column": np.concatenate([r[2] for r in entries])}
        cache[name] = data
        np.savez_compressed(OUT / f"{name}.npz", **data)
    return cache


def load(name):
    with np.load(OUT/f"{name}.npz", allow_pickle=False) as src:
        return {k: src[k] for k in src.files}


def subset(data, mask):
    return {k:v[mask] for k,v in data.items()}


def valid(data):
    z = data["truth"]
    return np.isfinite(z)&(z >= .25)&(z <= 30)


def corrections(cal):
    keep = valid(cal)
    e = log_median(cal["d"][keep], axis=2)-K[None,:]/cal["truth"][keep,None]
    names = cal["group"][keep]
    centers = np.array([log_median(e[names == g], axis=0) for g in np.unique(names)])
    offset = log_median(centers, axis=0)
    offset[3] = 0.  # Official metric calibration of the historical 16-cm control.
    if not np.isfinite(offset).all():
        raise ValueError("Some camera positions have no calibration observations")
    return offset


def observations(data, offset):
    single = log_median(data["d"]-offset[None,:,None], axis=2)
    with np.errstate(divide="ignore",invalid="ignore"):
        depths = K[None,:]/single
    good = (depths >= .25)&(depths <= 30)
    # Compare the same baseline-group medians under the calibrated depth range.
    matched = np.where(good, depths, np.nan)
    single = np.where(good,single,np.nan)
    direct = np.nanmedian(data["y"],axis=1).astype(float)
    all_median = np.nanmedian(matched,axis=1)
    return single, direct, all_median


def component(disparity, sigma, tau, eta):
    """Normal evidence and inverse-depth posteriors for zero or one bad baseline."""
    d = np.nan_to_num(disparity, nan=0.)
    original = np.isfinite(disparity)
    count = original.sum(1)
    mean, std, logw = [], [], []
    for j in range(-1, len(K)):
        use = original.copy()
        present = count >= 2
        if j >= 0:
            present &= original[:,j] & (count >= 3) & (d[:,j] >= -64) & (d[:,j] <= 320)
            use[:,j] = False
        n = use.sum(1)
        dim = sigma**2
        u = tau**2/(dim+n*tau**2)
        k1 = use @ K
        d1 = (d*use).sum(1)
        k2 = (use @ (K**2)-u*k1**2)/dim
        kd = ((d*use) @ K-u*k1*d1)/dim
        d2 = ((d*d*use).sum(1)-u*d1**2)/dim
        s0, m0 = 1.5, .55
        var = 1/(k2+1/s0**2)
        m = var*(kd+m0/s0**2)
        q = np.maximum(0, d2+m0**2/s0**2-m*m/var)
        logdet = n*np.log(dim)+np.log1p(n*tau**2/dim)
        loge = -.5*(q+logdet+np.log(s0**2/var)+n*np.log(2*np.pi))
        # Pr(I=all)=1-eta, and distribute one bad baseline uniformly if observable.
        if j < 0:
            mix = np.where(count >= 3,np.log1p(-eta),0.)
        else:
            mix = np.log(eta/np.maximum(count,1)/384)
        mean.append(m)
        std.append(np.sqrt(var))
        logw.append(np.where(present, loge+mix, -np.inf))
    return tuple(np.stack(v,axis=1) for v in (mean,std,logw))


def moment(x, mu, sd):
    t = (x-mu)/sd
    return mu*ndtr(t)-sd*PHI_CONST*np.exp(-.5*t*t)


def infer(disparity, fallback, sigma, tau, eta, draws=False):
    mu, sd, logw = component(disparity,sigma,tau,eta)
    support = np.isfinite(logw).any(axis=1)
    logw[~support,0] = 0.
    logw = logw-logsumexp(logw,axis=1,keepdims=True)
    w = np.exp(logw)
    left = moment(LOW,mu,sd)
    right = moment(HIGH,mu,sd)
    mass = np.sum(w*np.maximum(0,right-left),axis=1)
    support &= mass > 1e-20
    lo, hi = np.full(len(mu),LOW), np.full(len(mu),HIGH)
    for _ in range(35):
        mid = .5*(lo+hi)
        value = np.sum(w*(moment(mid[:,None],mu,sd)-left),axis=1)
        lower = value < .5*mass
        lo, hi = np.where(lower,mid,lo),np.where(lower,hi,mid)
    pred = np.where(support,2/(lo+hi),fallback)
    if not draws:
        return pred
    # Positive part of the mixture for posterior targets, before the 1/z loss weighting.
    lower = ndtr((LOW-mu)/sd)
    upper = ndtr((HIGH-mu)/sd)
    p = np.maximum(0,upper-lower)*w
    levels = (np.arange(41)+.5)/41
    r = mu[:,:,None]+sd[:,:,None]*ndtri((lower[:,:,None]+(upper-lower)[:,:,None]*levels).clip(1e-15,1-1e-15))
    r = r.clip(LOW,HIGH)
    p[~support] = 0
    p[~support,0] = 1
    r[~support,0] = (1/fallback[~support])[:,None]
    p = p/p.sum(1,keepdims=True)
    return pred, (1/r).reshape(len(r),-1), np.broadcast_to(p[:,:,None]/41,r.shape).reshape(len(r),-1)


def subjects(data,pred):
    use = valid(data)
    e = abs(pred[use]/data["truth"][use]-1)
    names = data["group"][use]
    return {str(g):float(e[names == g].mean()) for g in np.unique(names)}


def mean(d):
    return float(np.mean(list(d.values())))


def choice(cal):
    params = {"joint":list(itertools.product(SIGMAS,TAUS,OUTLIERS)),
              "origin":list(itertools.product(SIGMAS,(0.,),OUTLIERS))}
    folds = {a:{str(p):{} for p in items} for a,items in params.items()}
    for g in np.unique(cal["group"]):
        where = cal["group"] == g
        x = subset(cal,where)
        d,old,_ = observations(x,corrections(subset(cal,~where)))
        for a,items in params.items():
            for p in items:
                folds[a][str(p)][str(g)] = mean(subjects(x,infer(d,old,*p)))
    winners = {a:min(ps,key=lambda p:mean(folds[a][str(p)])) for a,ps in params.items()}
    return winners,folds


def bound_checks():
    k1,k2 = 100.,200.
    u,v = 52.,102.
    rho = (v-u)/(k2-k1)
    np.testing.assert_allclose([rho,u-k1*rho],[.5,2.],atol=1e-15)
    np.testing.assert_allclose(K*.49+K*.04,K*.53,atol=1e-12)
    # With no outlier slot and zero prior bias, the normal posterior comes from precision addition.
    y = (K*.55)[None,:]
    mu,sd,lp = component(y,.4,0.,.05)
    np.testing.assert_allclose(mu[0,0],.55,atol=1e-13)
    np.testing.assert_allclose(sd[0,0]**-2, np.sum(K*K)/.4**2+1/1.5**2,atol=1e-8)
    assert np.isfinite(lp).all()


def run():
    OUT.mkdir(exist_ok=True)
    names = ("cal","fit","held")
    data = {k:load(k) for k in names} if all((OUT/f"{k}.npz").is_file() for k in names) else measure()
    bound_checks()
    cal, held = data["cal"],data["held"]
    offsets = corrections(cal)
    selected, folds = choice(cal)
    d, direct, all_median = observations(held,offsets)
    direct_log = np.exp(np.nanmedian(np.log(held["y"].astype(float)),axis=1))
    sigma,tau,eta = selected["joint"]
    preds = dict(continuous_long=direct, continuous_log_long=direct_log, continuous_all=all_median,
                 origin=infer(d,direct,*selected["origin"]), same_origin=infer(d,direct,sigma,0,eta),
                 joint=infer(d,direct,sigma,tau,eta))
    scores = {k:subjects(held,v) for k,v in preds.items()}
    avg = {k:mean(v) for k,v in scores.items()}
    use = valid(held)
    count = np.isfinite(d[use]).sum(1)
    nspan = np.isfinite(d[use]).sum(0)
    old_bad = use & (abs(direct/held["truth"]-1) > .035)
    joint_error = abs(preds["joint"]/held["truth"]-1)
    deltas = {str(g):int(np.sum(old_bad & (held["group"] == g))) for g in np.unique(held["group"])}
    old_gate = avg["joint"] < avg["continuous_long"]
    student = old_gate and avg["joint"] < avg["continuous_all"]
    result = dict(means=avg,scores=scores,settings=selected,offsets=offsets.tolist(),folds=folds,
                  valid_points=int(use.sum()),held_scenes=np.unique(held["scene"]).tolist(),
                  available_baseline_counts=np.bincount(count,minlength=6).tolist(),available_views=nspan.tolist(),
                  bad_long_count=int(old_bad.sum()),bad_long_joint_recovered=int((old_bad&(joint_error<=.035)).sum()),
                  bad_long_all_median_recovered=int((old_bad&(abs(all_median/held["truth"]-1)<=.035)).sum()),
                  bad_long_group=deltas,student_gate=student,legacy_gate=old_gate)
    np.savez_compressed(OUT/"held_predictions.npz",truth=held["truth"],scene=held["scene"],group=held["group"],
                        disparity=d,**preds)
    if old_gate:
        x = data["fit"]
        d0,m0,a0 = observations(x,offsets)
        _,z,p = infer(d0,m0,*selected["joint"],draws=True)
        np.savez_compressed(OUT/"student-targets.npz",direct=a0,direct_long=m0,depth=z,probability=p)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print(result["means"],"student gate",student,"availability",result["available_baseline_counts"],flush=True)


if __name__ == "__main__":
    run()
