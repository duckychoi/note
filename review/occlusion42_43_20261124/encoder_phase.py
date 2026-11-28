"""Layer diagnosis and encoder-only adaptation of public Depth Pro 2-S."""
import argparse
from dataclasses import replace
from io import BytesIO
import json
import sys
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, STORE, TEST, save


SOURCES = ROOT.parents[2] / "research-sources"
DATA = SOURCES / "hypersim42_pilot" / "prepared.npz"
LIGHT = STORE / "lighting" / "paired_conditions.npz"
IBIMS = SOURCES/"encoder42_eval"/"iBims-1.zip"


def read(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def phase_distance(a, b, grid):
    def spectra(tokens):
        x = tokens.float().reshape(*grid, -1).permute(2, 0, 1)
        wy = torch.hann_window(grid[0], device=x.device, periodic=False)
        wx = torch.hann_window(grid[1], device=x.device, periodic=False)
        x = (x-x.mean(dim=(-2,-1), keepdim=True)) * wy[:,None] * wx[None,:]
        return torch.fft.rfft2(x)

    fa, fb = spectra(a), spectra(b)
    amplitude = (fa.abs()*fb.abs()).detach()
    similarity = (fa*fb.conj()).real/(amplitude+1e-6)
    weight = torch.sqrt(amplitude).detach()
    return ((1-similarity)*weight).sum()/weight.sum().clamp_min(1e-6)


def tapped(model, tensor):
    from depth_pro_2.model import INPUT_MULTIPLE
    from depth_pro_2.network.encoder import autocast_for

    x = F.pad(tensor[None], (0, -tensor.shape[-1]%INPUT_MULTIPLE, 0, -tensor.shape[-2]%INPUT_MULTIPLE),
              mode="replicate")
    with autocast_for(x.device, model.precision)():
        features, grid = model.encoder.encode(x)
    q = model.decode_frame(features,grid)[0,0,:tensor.shape[-2],:tensor.shape[-1]]
    return q, [feat[0] for feat in features], grid


def representation(model, transform):
    a = read(LIGHT)
    rows = []
    references = {group:i for i, (domain,group,label,_) in enumerate(a["metadata"])
                  if (domain,label) in (("synthetic","reference"),("real","L0e2"))}
    with torch.no_grad():
        for i,(domain,group,label,_) in enumerate(a["metadata"]):
            if (domain,label) not in (("synthetic","global"),("synthetic","beam"),
                                      ("real","L0e0"),("real","T0e0")):
                continue
            j = references[group]
            _, t0, grid = tapped(model, transform(a[f"{j}_rgb"]))
            _, t1, _ = tapped(model, transform(a[f"{i}_rgb"]))
            layers = []
            for x,y in zip(t0,t1):
                x,y = x.float(),y.float()
                correct = F.cosine_similarity(x,y).mean()
                shifted = y.reshape(*grid,-1).roll(2, dims=1).reshape_as(y)
                wrong = F.cosine_similarity(x,shifted).mean()
                layers.append(dict(cosine_distance=float(1-correct), phase_distance=float(phase_distance(x,y,grid)),
                                   same_minus_shifted=float(correct-wrong)))
            rows.append(dict(domain=str(domain),group=str(group),label=str(label),layers=layers))
    save("encoder_representation.json",dict(blocks=list(model.encoder.reassembled_blocks),rows=rows))
    print("Feature diagnosis",len(rows),flush=True)


def losses(q, gt, valid, focal, width):
    target = torch.log(focal/(width*gt))
    prediction = torch.log(q.clamp_min(1e-5))
    e = (prediction-target).masked_fill(~valid, 0)
    x = valid[:,1:] & valid[:,:-1]
    y = valid[1:] & valid[:-1]
    return e[valid].abs().mean()+.25*((e[:,1:]-e[:,:-1])[x].abs().mean()
                                     +(e[1:]-e[:-1])[y].abs().mean())


def load_independent(transform):
    from PIL import Image

    rows = []
    with ZipFile(IBIMS) as archive:
        names = sorted(name.rsplit("/",1)[0] for name in archive.namelist() if name.endswith("/meta.json"))
        assert len(names) == 100
        for name in names:
            raw = json.loads(archive.read(name+"/meta.json"))
            with Image.open(BytesIO(archive.read(name+"/depth.png"))) as file:
                values = np.asarray(file)
                t = (values.astype(np.float32)-1)/65533
                z = float(file.info["near"])**(1-t)*float(file.info["far"])**t
                z *= float(file.info.get("unit",1))
            with Image.open(BytesIO(archive.read(name+"/image.jpg"))) as file:
                assert file.size == (values.shape[1],values.shape[0])
                rgb = np.array(file.convert("RGB").resize((320,240),Image.Resampling.BOX))
            yy = ((np.arange(240)+.5)*values.shape[0]/240).astype(int)
            xx = ((np.arange(320)+.5)*values.shape[1]/320).astype(int)
            z,values = z[np.ix_(yy,xx)],values[np.ix_(yy,xx)]
            valid = (values>0)&(values<65535)&(z>=.25)&(z<=30)
            rows.append((name,transform(rgb),float(raw["intrinsics"][0][0])*320,z,valid))
    return rows


def validation(model, scenes):
    with torch.no_grad():
        valid_rows = []
        for a,b,z,v,focal in scenes:
            for x in (a,b):
                q = model.predict_inverse_depth(x[None])[0]
                predicted = focal/(x.shape[-1]*q.clamp_min(1e-5))
                valid_rows.append(float(((predicted[v]-z[v]).abs()/z[v]).mean()))
    return float(np.mean(valid_rows))


def evaluate(model, synth, real, independent):
    with torch.no_grad():
        depths = {}
        for scene,(x,fx) in real.items():
            q = model.predict_inverse_depth(x[None])[0]
            depths[scene] = (fx/(x.shape[-1]*q.clamp_min(1e-5))).cpu().numpy()
        independent_rows = {}
        for name,x,fx,z,mask in independent:
            q = model.predict_inverse_depth(x[None])[0]
            p = (fx/(x.shape[-1]*q.clamp_min(1e-5))).cpu().numpy()
            assert mask.any()
            d = p[mask]/z[mask]
            independent_rows[name] = [float(np.abs(d-1).mean()),
                                      float((np.maximum(d,1/d)<1.25).mean())]
    rows = {}
    for scene in TEST:
        a = read(STORE/f"{scene}_wide.npz")
        p,z,v,tri = depths[scene],a["z"],a["valid"],a["triples"]
        i,j = tri[:,0,None],tri[:,1:]
        jump = p.ravel()[j]-p.ravel()[i]
        truth = z.ravel()[j]-z.ravel()[i]
        rows[scene] = [float(np.abs(p[v]-z[v]).mean()),
                       float(np.abs(jump-truth).mean()),
                       float(((jump>0)+.5*(jump==0)).mean())]
    families = [np.mean([rows[s] for s in TEST if s[:-1]==name],axis=0)
                for name in sorted({s[:-1] for s in TEST})]
    result = dict(synthetic_absrel=validation(model,synth), real=np.mean(families,axis=0).tolist(),
                  scene_rows=rows)
    if independent_rows:
        result.update(ibims=np.mean(list(independent_rows.values()),axis=0).tolist(),
                      ibims_rows=independent_rows)
    return result


def run(steps, expanded, select_dev):
    sys.path.insert(0,str(SOURCES/"ml-depth-pro-2"/"src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    torch.set_num_threads(2)
    preset = replace(PRESETS["depthpro2-s"], fov_checkpoint_uri=None)
    model,transform = create_model_and_transforms(preset,device="cuda")
    if not expanded:
        representation(model,transform)
    model.requires_grad_(False)
    encoder = list(model.encoder.blocks[-2:].parameters())
    decoder = list(model.decoder.parameters())+list(model.head.parameters())
    initial = [p.detach().clone() for p in encoder+decoder]
    data = read(DATA.with_name("encoder_expanded_prepared.npz") if expanded else DATA)
    synth = []
    for a,b,z,fx in zip(*(data[k] for k in ("clean","changed","truth","focal"))):
        valid = np.isfinite(z) & (z >= .25) & (z <= 30)
        synth.append((transform(a),transform(b),torch.from_numpy(np.where(valid,z,1)).cuda(),
                      torch.from_numpy(valid).cuda(),float(fx)))
    real = {}
    for scene in TEST:
        row = read(STORE/f"{scene}.npz")
        real[scene] = transform(row["rgb"]),float(row["fx"])
    split = 20 if expanded else 4
    train = np.flatnonzero(data["scenes"]<split)
    dev = [synth[i] for i in np.flatnonzero(data["scenes"]>=split)]
    train = [i for i in train if bool(synth[i][3].any())]
    indices = np.random.default_rng(4243).choice(train,size=steps)
    holdout = load_independent(transform) if expanded else []
    scores = {"frozen":evaluate(model,dev,real,holdout)}
    arms = ("supervised","cosine","phase","decoder") if expanded else ("supervised","cosine","phase")
    for arm in arms:
        with torch.no_grad():
            for weight,original in zip(encoder+decoder,initial):
                weight.copy_(original)
                weight.requires_grad_(False)
        trainable = decoder if arm=="decoder" else encoder
        base = [p.detach().clone() for p in trainable]
        for weight in trainable:
            weight.requires_grad_(True)
        opt = torch.optim.AdamW(trainable,lr=2e-5,weight_decay=.01)
        curves = []
        total = []
        best = None
        for step,i in enumerate(indices,1):
            x1,x2,z,mask,fx = synth[i]
            q1,t1,grid = tapped(model,x1)
            q2,t2,_ = tapped(model,x2)
            loss = (losses(q1,z,mask,fx,x1.shape[-1])+losses(q2,z,mask,fx,x2.shape[-1]))/2
            if arm == "phase":
                loss = loss + .1*phase_distance(t1[-1],t2[-1],grid)
            elif arm == "cosine":
                loss = loss + .1*(1-F.cosine_similarity(t1[-1].float(),t2[-1].float()).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.)
            opt.step()
            assert torch.isfinite(loss)
            total.append(float(loss.detach()))
            if step%32==0 or step==steps:
                curves.append(dict(step=step,loss=float(np.mean(total))))
                if expanded and step%128==0:
                    curves[-1]["synthetic_absrel"] = validation(model,dev)
                    score = curves[-1]["synthetic_absrel"]
                    if select_dev and (best is None or score < best[0]):
                        best = (score,step,[p.detach().clone() for p in trainable])
                total.clear()
                print(arm,curves[-1],flush=True)
        if best is not None:
            with torch.no_grad():
                for parameter,selected in zip(trainable,best[2]):
                    parameter.copy_(selected)
        scores[arm] = evaluate(model,dev,real,holdout)
        scores[arm].update(curve=curves,max_weight_change=max(float((a-b).detach().abs().max())
                                                            for a,b in zip(trainable,base)),
                           trainable=sum(p.numel() for p in trainable),
                           selected_step=steps if best is None else best[1])
        print(arm,scores[arm]["real"],scores[arm].get("ibims"),flush=True)
        path = ("encoder_phase_dev_selected_results.json" if select_dev else
                "encoder_phase_expanded_results.json" if expanded else "encoder_phase_results.json")
        save(path,dict(steps=steps,train_images=len(train),scores=scores))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--steps",type=int,default=128)
    cli.add_argument("--expanded",action="store_true")
    cli.add_argument("--select-dev",action="store_true")
    args=cli.parse_args()
    if args.select_dev and (not args.expanded or args.steps<128):
        cli.error("--select-dev requires an expanded run with at least 128 steps")
    run(args.steps, args.expanded, args.select_dev)
