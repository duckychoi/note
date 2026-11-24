"""S3 수정 1: leave-one-demonstrator-out regularisation selection for the FM prior (noise sigma x iterations) and the AMP
discriminator (gradient-penalty weight), using ONLY the training demonstrators (seed 0 / seed 1).  Held-out seed-2 demos
(Xho_*) are never loaded.   python select_reg34.py s2/demos.npz s3b/reg_select.json
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sac34 as C  # noqa: E402,F401  (puts src/v2/il on the path)
from common import LOG_INV_DELTA  # noqa: E402
from vec import bmlp  # noqa: E402
from phase_util import BPhaseFlow, phase_feats  # noqa: E402

K = 17; PER = 12 * 900          # windows per training demonstrator (12 episodes x 900 windows, demos34 order: seed 0 then seed 1)
dev = torch.device("cuda"); torch.manual_seed(0)
D = np.load(sys.argv[1]); out_path = sys.argv[2]
X = {l: torch.tensor(D[f"X_{l}"], device=dev) for l in "FR"}; P = {l: torch.tensor(D[f"P_{l}"], device=dev) for l in "FR"}
assert all(len(X[l]) == 2 * PER for l in "FR"), {l: len(X[l]) for l in "FR"}
allx = torch.cat([X["F"], X["R"]])[:, :K]
MU = allx.mean(0).repeat(2); SD = (allx.std(0) + 1e-3).repeat(2)
XE = {l: (X[l] - MU) / SD for l in "FR"}
part = lambda l, a: (XE[l][a * PER:(a + 1) * PER], P[l][a * PER:(a + 1) * PER])


def loss_ps(net, x1, phi, M):
    B = x1.shape[0]; xr, pr = x1[None].repeat(1, M, 1), phi[None].repeat(1, M)
    s_ = torch.rand(1, B * M, device=dev) * LOG_INV_DELTA; t = 1 - torch.exp(-s_); w = LOG_INV_DELTA * t
    x0 = torch.randn_like(xr); xt = (1 - t[..., None]) * x0 + t[..., None] * xr
    return (((net(xt, t, pr) - (xr - x0)) ** 2).sum(-1) * w).view(M, B).mean(0)


def fm_scores(net, x, ph):
    with torch.no_grad():
        return torch.cat([loss_ps(net, x[a:a + 4096], ph[a:a + 4096], 16) for a in range(0, len(x), 4096)])


def train_fm(x, ph, sigma, iters):
    net = BPhaseFlow(1, 2 * K, "film4").to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        j = torch.randint(0, len(x), (1024,), device=dev)
        x1 = (x[j] + sigma * torch.randn_like(x[j]))[None]; p1 = ph[j][None]
        t = torch.where(torch.rand(1, 1024, device=dev) < 0.5, torch.rand(1, 1024, device=dev),
                        1 - torch.exp(-torch.rand(1, 1024, device=dev) * LOG_INV_DELTA))
        x0 = torch.randn_like(x1); xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        l = ((net(xt, t, p1) - (x1 - x0)) ** 2).sum(-1).mean()
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
    net.requires_grad_(False); return net


def auc(a, b, n=4000):
    r = np.random.default_rng(0); a = r.choice(a, n); b = r.choice(b, n)
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())


def train_disc(xr, pr, xf, pf, lam, iters=5000):
    dn = bmlp(1, [2 * K + 8, 256, 256, 1], nn.ReLU).to(dev); opt = torch.optim.Adam(dn.parameters(), 1e-4)
    for _ in range(iters):
        i = torch.randint(0, len(xr), (256,), device=dev); j = torch.randint(0, len(xf), (256,), device=dev)
        xe = torch.cat([xr[i], phase_feats(pr[i], 4)], -1)[None].requires_grad_(True)
        xp = torch.cat([xf[j], phase_feats(pf[j], 4)], -1)[None]
        de, dp = dn(xe), dn(xp)
        gp = torch.autograd.grad(de.sum(), xe, create_graph=True)[0].pow(2).sum(-1).mean()
        ld = ((de - 1) ** 2).mean() + ((dp + 1) ** 2).mean() + lam * gp
        opt.zero_grad(); ld.backward(); opt.step()
    return dn


def amp_r(dn, x, ph):
    with torch.no_grad():
        return (2 * torch.clamp(1 - 0.25 * (dn(torch.cat([x, phase_feats(ph, 4)], -1)[None]).squeeze(-1)[0] - 1) ** 2, min=0) - 1).cpu().numpy()


res = dict(fm={}, amp={})
for sigma in (0.0, 0.1, 0.3, 0.5):
    for iters in (3000, 30000):
        aucs, meds = [], []
        for a, b in ((0, 1), (1, 0)):
            for own, oth in (("F", "R"), ("R", "F")):
                xa, pa = part(own, a); net = train_fm(xa, pa, sigma, iters)
                q = float(fm_scores(net, xa, pa).median())
                rs = lambda x, p: np.clip(1 - np.log1p(fm_scores(net, x, p).cpu().numpy() / q) / math.log(2), -1, 1)
                ro, rt = rs(*part(own, b)), rs(*part(oth, b))
                aucs.append(auc(ro, rt)); meds.append(float(np.median(ro)))
        res["fm"][f"s{sigma}_i{iters}"] = dict(sigma=sigma, iters=iters, auc=float(np.mean(aucs)), med_own=float(np.mean(meds)), aucs=aucs, meds=meds)
        print("FM", sigma, iters, res["fm"][f"s{sigma}_i{iters}"], flush=True)
for lam in (1.0, 10.0, 100.0):
    aucs, meds = [], []
    for a, b in ((0, 1), (1, 0)):
        for own, oth in (("F", "R"), ("R", "F")):
            dn = train_disc(*part(own, a), *part(oth, a), lam)
            ro, rt = amp_r(dn, *part(own, b)), amp_r(dn, *part(oth, b))
            aucs.append(auc(ro, rt)); meds.append(float(np.median(ro)))
    res["amp"][f"gp{lam}"] = dict(lam=lam, auc=float(np.mean(aucs)), med_own=float(np.mean(meds)), aucs=aucs, meds=meds)
    print("AMP", lam, res["amp"][f"gp{lam}"], flush=True)


def pick(d):
    ok = {k: v for k, v in d.items() if v["med_own"] > -0.9}
    return max(ok, key=lambda k: ok[k]["auc"]) if ok else max(d, key=lambda k: d[k]["med_own"])


res["chosen_fm"] = pick(res["fm"]); res["chosen_amp"] = pick(res["amp"])
res["rule"] = "max mean LODO AUC among candidates with mean own-style median reward > -0.9; else max median"
json.dump(res, open(out_path, "w"), indent=1)
print("CHOSEN", res["chosen_fm"], res["fm"][res["chosen_fm"]], res["chosen_amp"], res["amp"][res["chosen_amp"]], flush=True)
