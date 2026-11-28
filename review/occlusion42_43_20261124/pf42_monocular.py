"""Counterfactual scale swaps and matched shape-supervision training for #42 B."""
import argparse
from collections import defaultdict
import copy
import json
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import pf42_visibility_encoder as study
from pf42_crossview import comparison, load


ORIGINAL = study.RUN
RUN = study.OUT / "monocular-identifiability"
DATASETS = ("hammer", "ibims")
ARMS = ("direct", "factor_direct", "factor_shared")


def objective(arm, p, direct, probability, log_grid):
    if arm == "direct":
        return F.smooth_l1_loss(p, direct, beta=.06)
    scale = F.smooth_l1_loss(p.mean(), direct.mean(), beta=.06)
    centered = p - p.mean()
    if arm == "factor_direct":
        return scale + F.smooth_l1_loss(centered, direct-direct.mean(), beta=.06)
    teacher_mean = (probability * log_grid[None]).sum(1).mean()
    target = (log_grid - teacher_mean)[None].expand_as(probability)
    error = F.smooth_l1_loss(centered[:,None].expand_as(probability), target, beta=.06, reduction="none")
    return scale + (probability*error).sum(1).mean()


def load_models(model, original, arms):
    out = {}
    folder = ORIGINAL if original else RUN
    for seed in study.SEEDS:
        payload = torch.load(folder / f"seed-{seed}.pt", weights_only=True, map_location="cpu")
        if payload["step"]!=study.WARM+study.JOINT:
            raise ValueError(f"Incomplete training for {seed}")
        for a in arms:
            key = f"{seed}-{a}"
            out[key] = copy.deepcopy(model.encoder.blocks[model.start:]).requires_grad_(False)
            out[key].load_state_dict(payload["weights"][a])
    return out


def fold_keys(dataset):
    ds = load(study.OUT.parent / f"{dataset}.npz")
    chosen = {}
    for g in np.unique(ds["group"]):
        names = np.unique(ds["key"][ds["group"]==g])
        main = names[np.linspace(0,len(names)-1,min(8,len(names)),dtype=int)]
        remaining = np.setdiff1d(names, main)
        if len(remaining):
            extra = remaining[np.linspace(0,len(remaining)-1,min(8,len(remaining)),dtype=int)]
            chosen.update({str(k):"extra" for k in extra})
        chosen.update({str(k):"same" for k in main})
    return chosen


def pixel_sample(z, valid):
    y, x = np.meshgrid(np.linspace(0,z.shape[0]-1,16,dtype=int),
                       np.linspace(0,z.shape[1]-1,24,dtype=int),indexing="ij")
    keep = valid[y,x] & np.isfinite(z[y,x]) & (z[y,x]>=.25) & (z[y,x]<=30)
    return y[keep], x[keep]


def absrel(p, log_truth):
    return float(np.mean(abs(np.exp(p-log_truth)-1)))


def optimal_absrel(p, z):
    ratio = np.exp(p-z)
    order = np.argsort(1/ratio)
    s = np.cumsum(ratio[order])
    best = order[np.searchsorted(s, .5*s[-1])]
    return float(abs(ratio/ratio[best]-1).mean())


def swaps(direct, shared, true):
    d0, s0 = direct.mean(), shared.mean()
    old, new = absrel(direct,true), absrel(shared,true)
    ds = absrel(d0+(shared-s0), true)
    sd = absrel(s0+(direct-d0), true)
    dz, sz = direct-true, shared-true
    return dict(direct=old, shared=new, direct_scale_shared_shape=ds, shared_scale_direct_shape=sd,
                scale_change=.5*((sd-old)+(new-ds)), shape_change=.5*((ds-old)+(new-sd)),
                direct_oracle=optimal_absrel(direct,true), shared_oracle=optimal_absrel(shared,true),
                direct_scale_mse=float(dz.mean()**2), shared_scale_mse=float(sz.mean()**2),
                direct_shape_mse=float(dz.var()), shared_shape_mse=float(sz.var()),
                direct_log_bias=float(dz.mean()), shared_log_bias=float(sz.mean()))


def groups_for(rows, fold):
    data = defaultdict(list)
    for row in rows:
        if row["fold"]==fold:
            data[row["group"]].append(row["scores"])
    return {g: {m:float(np.mean([r[m] for r in values])) for m in values[0]} for g,values in data.items()}


def summarize(rows, pairs):
    result = {}
    for fold in ("same","extra"):
        group = groups_for(rows,fold)
        vals = list(group.values())
        means = {name:float(np.mean([r[name] for r in vals])) for name in vals[0]}
        relative = {new+"_vs_"+ref:comparison({g:v[new] for g,v in group.items()}, {g:v[ref] for g,v in group.items()})
                    for new,ref in pairs}
        result[fold] = dict(means=means, group=group, relative=relative, n=sum(r["fold"]==fold for r in rows))
    return result


@torch.no_grad()
def assess(model, diagnostic):
    model.arms = load_models(model, diagnostic, ("direct","shared") if diagnostic else ARMS)
    prefix = "diagnostic" if diagnostic else "matched-transfer"
    for ds in DATASETS:
        path = RUN / f"{prefix}-{ds}.json"
        if path.is_file():
            continue
        keys = fold_keys(ds)
        if not diagnostic:
            prior = json.loads((RUN / f"diagnostic-{ds}.json").read_text(encoding="utf8"))
            old = {r["key"]:r for r in prior["rows"]}
        rows = []
        for key, group, rgb, truth, valid in study.frames(ds):
            if key not in keys:
                continue
            yy, xx = pixel_sample(truth, valid)
            if not len(xx):
                continue
            rgb, xy = study.rescale(rgb, np.column_stack((xx,yy)))
            state = model.encode(rgb)
            x, y = xy.T
            log_truth = np.log(truth[yy,xx].astype(np.float64))
            log_depths = {k:model.depth(state,k)[y,x].cpu().numpy().astype(np.float64) for k in model.arms}
            if diagnostic:
                by_seed = [swaps(log_depths[f"{s}-direct"],log_depths[f"{s}-shared"],log_truth) for s in study.SEEDS]
                score = {m:float(np.mean([r[m] for r in by_seed])) for m in by_seed[0]}
            else:
                score = {a:float(np.mean([absrel(log_depths[f"{s}-{a}"],log_truth) for s in study.SEEDS])) for a in ARMS}
                score["shared"] = old[key]["scores"]["shared"]
                score["previous_direct"] = old[key]["scores"]["direct"]
                by_seed = [{a:absrel(log_depths[f"{s}-{a}"],log_truth) for a in ARMS} for s in study.SEEDS]
            rows.append(dict(key=key, group=group, fold=keys[key], points=len(x), scores=score, seeds=by_seed))
        if not diagnostic and {r["key"] for r in rows}!=set(old):
            raise ValueError(f"The evaluation key list changed for {ds}")
        pairs = [("shared","direct")] if diagnostic else [
            ("factor_shared","direct"), ("factor_shared","factor_direct"), ("shared","previous_direct")]
        summary = summarize(rows,pairs)
        temp = path.with_suffix(".partial.json")
        temp.write_text(json.dumps(dict(rows=rows,summary=summary),indent=2,allow_nan=False),encoding="utf8")
        temp.replace(path)
        for fold, values in summary.items():
            print(prefix,ds,fold,values["n"],len(values["group"]),values["means"],values["relative"],flush=True)


def run(phase):
    RUN.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    with study.gpu_slot():
        while not study.capacity():
            time.sleep(5)
        model = study.Trainer()
        if phase=="diagnose":
            assess(model, True)
            return
        if phase=="evaluate":
            assess(model, False)
            return
        real = study.real_data()
        study.SYN, pool, study.SYN_ALTERNATIVES, study.SYN_REPLACEMENTS = study.syn_rows()
        study.RUN, study.ARMS = RUN, ARMS
        model.arms = {a:copy.deepcopy(model.encoder.blocks[model.start:]).requires_grad_(True) for a in ARMS}
        study.train(model, real, pool, real_objective=objective)
        assess(model, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("diagnose","train","evaluate"))
    run(parser.parse_args().phase)
