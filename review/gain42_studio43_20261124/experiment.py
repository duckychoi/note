import copy
import hashlib
import json
import time

import cv2
import numpy as np
import torch
from torch import nn

from footage import DEV, read
from scenes import CACHE, COUNTS, HERE, DepthNet, make_views

SEEDS = [4211, 4217, 4229, 4231, 4241]
ARMS = ["uniform","source","replay","gain","visible","shuffle","learned"]


def bank(seed, split, count):
    if split == 3:
        parents = np.concatenate([np.repeat(np.arange(8)+8*b, COUNTS) for b in range(4)])
    else:
        parents = np.arange(count)
    rows = []
    for i, parent in enumerate(parents):
        variant = int(np.sum(parents[:i] == parent))
        rows.extend(make_views(seed, split, int(parent), variant))
    inputs = torch.stack([r[0] for r in rows]).cuda()
    labels = np.stack([r[1] for r in rows])
    fields = [np.stack([r[j] for r in rows]) for j in range(2,6)]
    return dict(x=inputs, z=torch.from_numpy(np.log(labels)).cuda(), truth=1/labels, obs=fields[0],
                good=fields[1], gradient=fields[2], consistency=fields[3],
                visibility=sum(r[6] for r in rows)/labels.size, parent=parents.repeat(2),
                checksum=hashlib.sha256(inputs.cpu().numpy().tobytes()+labels.tobytes()).hexdigest())


def fit_steps(net, data, steps, seed, arm, weight=None, dev=None, warm=False):
    opt = torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001)
    sampled = []
    rng = np.random.default_rng(seed)
    n = len(data["x"])
    unique, counts = np.unique(data["parent"],return_counts=True)
    prob = 1/counts[np.searchsorted(unique,data["parent"])]
    prob /= prob.sum()
    for step in range(steps):
        if arm == "source":
            idx = rng.choice(n,4,p=prob)
        else:
            limit = n if warm or arm=="uniform" else (step//48+1)*n//4
            idx = rng.integers(limit,size=4)
        sampled.extend(idx.tolist())
        prediction = net(data["x"][idx])[:,0]
        diff = prediction - data["z"][idx]
        if weight is None:
            loss = diff.abs().mean()+.5*(diff.diff(dim=1).abs().mean()+diff.diff(dim=2).abs().mean())
        else:
            w = weight[idx]
            loss = (w*diff.abs()).mean()+.5*(
                ((w[:,1:]+w[:,:-1])*.5*diff.diff(dim=1).abs()).mean()+
                ((w[:,:,1:]+w[:,:,:-1])*.5*diff.diff(dim=2).abs()).mean())
        if dev is not None and (step+1)%8 == 0:
            real_log = net(dev[0])[:,0]
            errors = [(real_log[b,y,x]-target).abs() for b,(y,x,target) in enumerate(dev[1])]
            loss = loss+.25*torch.cat(errors).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    return {"batch_stream":hashlib.sha256(np.int32(sampled).tobytes()).hexdigest(),
            "views_drawn":len(sampled),"unique_views":len(set(sampled)),"final_loss":float(loss.detach())}


def features(teacher, data):
    with torch.no_grad():
        m = torch.cat([(-teacher(x)[:,0]).exp() for x in data["x"].split(8)]).cpu().numpy()
    v = data["obs"]
    g = data["gradient"]
    innovation = v-m
    bins = np.digitize(g,[.035,.14])*3+np.digitize(np.abs(innovation),[.17,.55])
    feature = np.stack((m/2, v/2, innovation, np.abs(innovation), g*3,
                        data["consistency"]),axis=-1).clip(-5,5)
    return m, bins, feature


def fit_gains(teacher, data, seed):
    m,bins,features_ = features(teacher,data)
    ok = data["good"]
    error = (m-data["truth"])[ok]
    delta = (m-data["obs"])[ok]
    def coefficient(e, d):
        denominator = np.sum(np.square(d),dtype=np.float64)
        return float(np.clip(np.sum(e*d,dtype=np.float64)/denominator,0,1)) if denominator else 0.
    constant = coefficient(error,delta)
    table = []
    for index in range(9):
        take = bins[ok] == index
        table.append(coefficient(error[take],delta[take]) if take.any() else constant)
    x = torch.from_numpy(features_[ok]).cuda()
    e = torch.from_numpy(error).cuda()
    d = torch.from_numpy(delta).cuda()
    head = nn.Sequential(nn.Linear(6,16), nn.SiLU(), nn.Linear(16,1), nn.Sigmoid()).cuda()
    opt = torch.optim.Adam(head.parameters(),lr=.005)
    rng = np.random.default_rng(seed+71)
    for _ in range(160):
        i = rng.integers(len(e),size=1024)
        loss = ((e[i]-head(x[i]).flatten()*d[i])**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return constant, np.array(table,np.float32), head.eval()


def maps(teacher, gains, data, seed, audit=False):
    constant,table,head = gains
    m,bins,x = features(teacher,data)
    ok = data["good"]
    k = np.where(ok,table[bins],0).astype(np.float32)
    learned = np.zeros_like(k)
    with torch.no_grad():
        features_gpu = torch.from_numpy(x[ok]).cuda()
        learned[ok] = torch.cat([head(b) for b in features_gpu.split(131072)]).flatten().cpu().numpy()
    if audit:
        truth = data["truth"][ok]
        prior,obs = m[ok],data["obs"][ok]
        metrics = {name:float(np.mean((prior+weight*(obs-prior)-truth)**2))
                   for name,weight in [("mono",0),("stereo",1),("constant",constant),
                                       ("gain",k[ok]),("learned",learned[ok])]}
        return dict(mse=metrics, visible_fraction=data["visibility"],accepted_fraction=float(ok.mean()),
                    accepted=int(ok.sum()), mean_gain=float(k[ok].mean()),gain_bins=table.tolist(),
                    constant=constant,calibration_is_disjoint=True)
    visible = np.zeros_like(k)
    shuffle = k.copy()
    rng = np.random.default_rng(seed+29)
    for i,mask in enumerate(ok):
        if mask.any():
            visible[i,mask] = k[i,mask].mean()
            shuffle[i,mask] = rng.permutation(k[i,mask])
    result = {}
    for name,values in [("gain",k),("visible",visible),("shuffle",shuffle),("learned",learned)]:
        w = 1+2*values
        w /= w.mean(axis=(1,2),keepdims=True)
        result[name] = torch.from_numpy(w).cuda()
    return result


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    checkpoint_dir = CACHE/"checkpoints"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    # Few-shot images are only passed to the 'few' branch, never used for tuning.
    anchors = read(DEV)
    dev = (torch.stack([row[0] for row in anchors]).cuda(),
           [tuple(torch.from_numpy(a).cuda() for a in row[1:4]) for row in anchors])
    for seed in SEEDS:
        started=time.perf_counter()
        torch.manual_seed(seed)
        net = DepthNet(False).cuda()
        warm = bank(seed,0,48)
        info = {"seed":seed, "splits":{"warm":warm["checksum"]}, "anchors":[
            dict(scene=row[4][0],frame=row[4][1],coordinates=list(map(list,zip(row[1].tolist(),row[2].tolist()))),
                 valid=len(row[1]),queries=32) for row in anchors],"models":{}}
        info["warm"] = fit_steps(net,warm,80,seed,"uniform",warm=True)
        del warm
        cal = bank(seed,1,24)
        info["splits"]["calibration"] = cal["checksum"]
        gains = fit_gains(net.eval(),cal,seed)
        torch.save(gains[2].state_dict(),checkpoint_dir/f"{seed}_gain_head.pt")
        del cal
        audit_data = bank(seed,2,16)
        info["splits"]["audit"] = audit_data["checksum"]
        info["audit"] = maps(net,gains,audit_data,seed,audit=True)
        del audit_data
        full = bank(seed,3,128)
        info["splits"]["main"] = full["checksum"]
        weights = maps(net,gains,full,seed)
        for budget in ("zero","few"):
            for arm in ARMS:
                student = copy.deepcopy(net).train()
                record = fit_steps(student,full,192,seed+109,arm,weights.get(arm),
                                   dev if budget=="few" else None)
                record.update(physical_renders=432, unique_real_queries=64 if budget=="few" else 0,
                              real_gradient_steps=24 if budget=="few" else 0,student_updates=272)
                info["models"][budget+"/"+arm] = record
                torch.save(student.state_dict(),checkpoint_dir/f"{seed}_{budget}_{arm}.pt")
            print("trained",seed,budget,round(time.perf_counter()-started,1),flush=True)
        (HERE/f"ledger_{seed}.json").write_text(json.dumps(info,indent=1),encoding="utf8")
        del full,weights


if __name__ == "__main__":
    main()
