"""Measure a two-dimensional tangent of the trained small encoder on real anchors."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image
import psutil
import torch

from repeat_capture_acquire import OUT

SRC = OUT.parents[1]
DESIGN = ("Art", "Aloe", "Midd1", "Wood1")
LIMIT = .25
RIDGE = .001


def family(name):
    return re.sub(r"\d+$", "", name)


def quadrants(height, width):
    y,x = np.mgrid[:height,:width]
    inner = (x > .12*width) & (x < .88*width) & (y > .12*height) & (y < .88*height)
    return [inner & ((x >= width/2) == side) & ((y >= height/2) == bottom)
            for bottom in (False,True) for side in (False,True)]


def trust_ridge(H, r):
    H, r = H/np.sqrt(len(r)), r/np.sqrt(len(r))
    def solve(lam):
        return np.linalg.solve(H.T@H+lam*np.eye(H.shape[1]),H.T@r)
    e = solve(RIDGE)
    if np.linalg.norm(e) <= LIMIT:
        return e
    lo,hi = RIDGE,1.
    while np.linalg.norm(solve(hi)) > LIMIT:
        hi *= 2
    for _ in range(55):
        mid = (lo+hi)/2
        if np.linalg.norm(solve(mid)) > LIMIT:
            lo = mid
        else:
            hi = mid
    return solve(hi)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    os.environ["TORCH_HOME"] = str(SRC/"ml-depth-pro-2-cache")
    sys.path.insert(0,str(SRC/"ml-depth-pro-2/src"))
    from depth_pro_2 import create_model_and_transforms
    model,transform = create_model_and_transforms("depthpro2-s",device=torch.device("cpu"))
    ck = OUT.parent/"control43-geometry-pairs-v1/full-students-4244/d2/best.pt"
    model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True),strict=True)
    model.requires_grad_(False)
    par = list(model.encoder.blocks[-2:].parameters())
    for p in par:
        p.requires_grad_(True)
    cache = OUT/"encoder"
    cache.mkdir(exist_ok=True)
    def derivatives(name):
        while psutil.virtual_memory().available < 10*2**30:
            time.sleep(12)
        src = OUT/"physical"/name
        rgb = Image.open(src/"Illum1/Exp1/view1.png").convert("RGB")
        width,height = rgb.size
        shape = (256,round(256*height/width))
        arr = np.array(rgb.resize(shape,Image.Resampling.BILINEAR))
        inp = transform(arr).unsqueeze(0)
        with torch.no_grad():
            focal = model.fov.focal_length_px(inp).detach()
        inv = model.predict_inverse_depth(inp)[0]*shape[0]/focal
        disp = np.array(Image.open(src/"disp1.png").resize(shape,Image.Resampling.NEAREST),dtype=float)
        truth = (disp + float((src/"dmin.txt").read_text()))/(3740*.16)
        outputs, gradients, residual = [], [], []
        for mask in quadrants(shape[1],shape[0]):
            mask = mask & (disp > 0)
            if not mask.any():
                raise RuntimeError(f"{name} empty anchor quadrant")
            outputs.append(inv[torch.from_numpy(mask)].mean())
            residual.append(truth[mask].mean()-outputs[-1].detach().item())
        for i,out in enumerate(outputs):
            grad = torch.autograd.grad(out,par,retain_graph=i<len(outputs)-1)
            gradients.append(torch.cat([g.detach().reshape(-1) for g in grad]))
        return torch.stack(gradients),np.asarray(residual),inv.detach().numpy(),float(focal)
    probes = torch.cat([derivatives(scene)[0] for scene in DESIGN])
    eig,vec = torch.linalg.eigh((probes@probes.T).double())
    if eig[-2]<=0:
        raise ValueError("Rank-deficient encoder design")
    basis = (vec[:,-2:].T.float()@probes)/eig[-2:].float().sqrt()[:,None]
    assert torch.max(torch.abs(basis@basis.T-torch.eye(2))) < .001
    torch.save(basis,cache/"basis.pt")
    del probes
    meta = dict(design=DESIGN, rank=2, parameters=sum(p.numel() for p in par),
                eigenvalues=eig[-2:].tolist(),ridge=RIDGE,trust_radius=LIMIT,
                weights=str(ck.relative_to(SRC)),sha256=digest(ck),device="CPU",input_width=256)
    for src in sorted((OUT/"physical").iterdir()):
        grad,residual,pred,focal = derivatives(src.name)
        H = (grad@basis.T).double().numpy()
        e = trust_ridge(H,residual)
        np.savez_compressed(cache/(src.name+".npz"),H=H,residual=residual,e=e,
                            inverse=pred,focal=focal)
        print(src.name,flush=True)
    (cache/"model.json").write_text(json.dumps(meta,indent=1),encoding="utf8")


if __name__=="__main__":
    main()
