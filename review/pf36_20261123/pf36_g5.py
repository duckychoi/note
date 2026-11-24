"""#36 gate 5: reproduce the wide-prior conditional-flow proposal PF. See PREREG.md (관문 5).
Usage: python pf36_g5.py diag | main | calib"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import pf36_g3 as g  # noqa: E402
import pf36_g4b as f  # noqa: E402

torch.set_num_threads(16)
HERE = Path(__file__).resolve().parent
OLD = dict(name="old", lr=1e-3, sched="const", clip=5., epochs=60, patience=5)
NEW = dict(name="new", lr=1e-3, sched="cosine", clip=1., epochs=40, patience=None)


def train5(lib, M, seed, cfg):
    pos, _, st0, im0, id0 = lib
    pos, z, cls = pos[:M], st0[:M][:, f.ZD].astype(np.float32), (im0[:M]*12+id0[:M]).astype(np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    nv = M//10
    vi, ti = perm[:nv], perm[nv:]
    torch.manual_seed(seed)
    model = f.Proposal()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    steps = cfg["epochs"]*math.ceil(len(ti)/1024)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5) if cfg["sched"] == "cosine" else None

    def batch(idx, r):
        obs = pos[idx] + r.normal(0, g.SIG, pos[idx].shape)
        mask = np.zeros((len(idx), g.T))
        mask[:, :g.CUT] = r.uniform(size=(len(idx), g.CUT)) > .4
        return torch.tensor(f.features(obs, mask)), torch.tensor(cls[idx]), torch.tensor(z[idx])
    vx, vc, vz = batch(vi, np.random.default_rng(seed+1))
    best, best_state, bad, curve = np.inf, None, 0, []
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(ti)
        tl = []
        for s in range(0, len(order), 1024):
            x, c, zz = batch(order[s:s+1024], rng)
            loss = -model.log_prob(x, c, zz).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            if sched:
                sched.step()
            tl.append(float(loss))
        model.eval()
        with torch.no_grad():
            vl = float(-model.log_prob(vx, vc, vz).mean())
        curve.append(dict(epoch=ep+1, train=float(np.mean(tl)), train_max_batch=float(np.max(tl)), val=vl))
        if vl < best-1e-3:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if cfg["patience"] and bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, dict(cfg=cfg["name"], epochs=len(curve), best_val_nll=best, curve=curve)


def setup():
    val = g.make_cases(51000, 48, test=False)
    tests = [g.make_cases(s, 48, test=True) for s in range(51200, 51204)]
    broad = g.library(g.POOL, 12, 12, 196_608, 52003)
    tr = g.library(g.TRPOOL, 8, 8, 196_608, 52001)
    xthr = float(np.median(tr[0][:, -1, 0]-tr[0][:, g.CUT-1, 0]))
    return val, tests, broad, xthr


def summarize(r):
    return {k: float(np.mean(v)) for k, v in r.items()}


def pf_eval(model, cases, N, tp, xthr, seed, alpha=None):
    old = f.ALPHA
    if alpha is not None:
        f.ALPHA = alpha
    try:
        return g.run_pf_full(cases, N, tp, xthr, seed, init=f.proposal_init(model, cases, N, 0), resample_after=g.CUT-1)
    finally:
        f.ALPHA = old


def diag(log):
    val, _, broad, xthr = setup()
    rows = []
    for M, N in ((98_304, 512), (147_456, 256)):
        for s in (7100, 7200, 7300):
            for cfg in (OLD, NEW):
                model, info = train5(broad, M, s, cfg)
                e = float(np.mean(pf_eval(model, val, N, 1, xthr, 3)["energy"]))
                rows.append(dict(M=M, N=N, seed=s, val_case_energy_t1=e, **info))
                log(M, s, cfg["name"], "epochs", info["epochs"], "nll", round(info["best_val_nll"], 3), "val energy", round(e, 4),
                    "max batch loss", round(max(c["train_max_batch"] for c in info["curve"]), 2))
    wins = sum(1 for i in range(0, len(rows), 2) if rows[i+1]["best_val_nll"] < rows[i]["best_val_nll"])
    sd_old = np.std([r["best_val_nll"] for r in rows if r["cfg"] == "old"])
    sd_new = np.std([r["best_val_nll"] for r in rows if r["cfg"] == "new"])
    adopt = bool(wins >= 5 and sd_new < sd_old)
    out = dict(rows=rows, new_wins=wins, sd_old=float(sd_old), sd_new=float(sd_new), adopt_new=adopt)
    (HERE/"gate5_diag.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    log("adopt_new", adopt, "wins", wins, "sd old/new", sd_old, sd_new)


def chosen_cfg():
    d = json.loads((HERE/"gate5_diag.json").read_text(encoding="utf-8"))
    return NEW if d["adopt_new"] else dict(OLD, name="old_patience15", patience=15)


def main(log):
    val, tests, broad, xthr = setup()
    cfg = chosen_cfg()
    a4 = json.loads((HERE/"gate4a.json").read_text(encoding="utf-8"))
    rk = a4["best_retrieval"]
    ret, ret_seeds = a4["table"][rk]["energy"], np.array(a4["table"][rk]["energy_per_seed"])
    ref = a4["reference"]["pf_N8192"]["energy"]
    runs = []
    models = {}
    for M, N, role in ((147_456, 256, "primary"), (98_304, 512, "secondary")):
        for s in (8100, 8200, 8300):
            model, info = train5(broad, M, s, cfg)
            models[(M, s)] = model
            vt = {tp: float(np.mean(pf_eval(model, val, N, tp, xthr, 3)["energy"])) for tp in (1, 2, 4)}
            tp = min(vt, key=vt.get)
            per = [summarize(pf_eval(model, c, N, tp, xthr, 500+N+i)) for i, c in enumerate(tests)]
            es = np.array([p["energy"] for p in per])
            e = float(es.mean())
            r = dict(role=role, M=M, N=N, train_seed=s, cfg=cfg["name"], epochs=info["epochs"], val_nll=info["best_val_nll"], pf_temp=tp, val=vt,
                     **{k: float(np.mean([p[k] for p in per])) for k in per[0]}, energy_per_seed=es.tolist(),
                     vs_ref=e/ref-1, vs_retrieval=e/ret-1, all_seeds_beat_retrieval=bool(np.all(es < ret_seeds)),
                     passed=bool(e <= 1.05*ref and e <= .95*ret and np.all(es < ret_seeds)))
            runs.append(r)
            log(role, M, N, s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                if k not in ("energy_per_seed", "val", "role", "M", "N", "train_seed", "cfg")})
    agg = {}
    for role in ("primary", "secondary"):
        rs = [r for r in runs if r["role"] == role]
        agg[role] = dict({k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))]
                          for k in ("energy", "brier_bounce2", "brier_xpast", "cover_x", "cover_z")},
                         all_passed=all(r["passed"] for r in rs), n_passed=sum(r["passed"] for r in rs))
    reproduced = agg["primary"]["all_passed"]
    out = dict(cfg=cfg, reference=dict(pf_N8192=ref, retrieval=rk, retrieval_energy=ret), runs=runs, aggregate=agg, reproduced=reproduced)
    (HERE/"gate5.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    log("aggregate", agg, "reproduced", reproduced)
    torch.save({f"{M}_{s}": m.state_dict() for (M, s), m in models.items() if M == 147_456}, HERE/"gate5_primary_models.pt")


def calib(log):
    """Coverage calibration on validation only: choose alpha in {.1,.3,.5} (MH step not implemented, see RESULT)."""
    val, tests, broad, xthr = setup()
    cfg = chosen_cfg()
    g5 = json.loads((HERE/"gate5.json").read_text(encoding="utf-8"))
    sd = torch.load(HERE/"gate5_primary_models.pt")
    rows = []
    for s in (8100, 8200, 8300):
        model = f.Proposal()
        model.load_state_dict(sd[f"147456_{s}"])
        model.eval()
        run = next(r for r in g5["runs"] if r["role"] == "primary" and r["train_seed"] == s)
        tp = run["pf_temp"]
        vres = {}
        for a in (.1, .3, .5):
            r = summarize(pf_eval(model, val, 256, tp, xthr, 3, alpha=a))
            vres[a] = r
        base = vres[.1]["energy"]
        ok = [a for a in vres if vres[a]["energy"] <= 1.05*base]
        a_star = min(ok, key=lambda a: abs((vres[a]["cover_x"]+vres[a]["cover_z"])/2-.8))
        per = [summarize(pf_eval(model, c, 256, tp, xthr, 500+256+i, alpha=a_star)) for i, c in enumerate(tests)]
        after = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        rows.append(dict(train_seed=s, alpha=a_star, val=vres, before={k: run[k] for k in ("energy", "cover_x", "cover_z", "brier_bounce2", "brier_xpast")},
                         after=after, energy_per_seed_after=[p["energy"] for p in per]))
        log(s, "alpha", a_star, "before", rows[-1]["before"], "after", {k: round(v, 4) for k, v in after.items()})
    (HERE/"gate5_calib.json").write_text(json.dumps(dict(rows=rows), indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    dict(diag=diag, main=main, calib=calib)[sys.argv[1]](log)
