"""#36 gate 8: action design for material identifiability, with a cheap material-posterior carry-over.
Two past throws per object under three designs {drop,drop} / {drop,oblique} / {oblique,oblique}; next throw = one drop AND one
oblique throw per object (mixed set), forecast frames 25-49 from the 25-frame prefix.
Material posterior over the 144-class grid:
  cheap  : one joint IS pass per past throw (classes ~ (1-eps)(1/2 cat + 1/2 pi_prev) + eps/144, launch state ~ flow mixture),
           weight pi_prev(c) p(z) L / (q(c) q(z|c)); pi_new(c) = normalised weight mass of class c (unsampled classes -> 0, counted).
  exact  : gate-7 per-class marginal likelihood, 64 IS particles x 144 classes per object (reference for identifiability + cost).
Baselines: fresh (uniform material prior) at N in {256, 1024, 4096, 8192}; oracle (true continuous material).
See PREREG.md (관문 8).  Usage: python pf36_g8.py profile | run"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402

import pf36_g3 as g  # noqa: E402
import pf36_g4b as f  # noqa: E402
import pf36_g6a as a6  # noqa: E402
import pf36_g7 as g7  # noqa: E402

torch.set_num_threads(16)
HERE = Path(__file__).resolve().parent
NP_CHEAP = 2048
M_MIN = 4
LMU, DD = np.log(g.PF_MU)[np.arange(144)//12], g.PF_D[np.arange(144) % 12]


def launch(rng, n, kind):
    s = g.init_states(rng, n)
    if kind == "drop":          # vertical drop: no horizontal speed, no spin
        s[:, 4], s[:, 6] = 0., 0.
    else:                       # oblique sliding throw: fast horizontal speed, no spin -> slides on contact
        s[:, 4], s[:, 6] = rng.uniform(.8, 1.5, n), 0.
    return s


def make_throw(models, rng, kind, prefix_only=False):
    n = len(models)
    pos, bnc = g.simulate(models, launch(rng, n, kind), rng, g.T)
    obs = pos + rng.normal(0, g.SIG, pos.shape)
    mask = (rng.uniform(size=(n, g.T)) > .4).astype(float)
    if prefix_only:
        mask[:, g.CUT:] = 0
    return dict(pos=pos, bnc=bnc, obs=obs, mask=mask)


def objects(seed, n):
    rng = np.random.default_rng(seed)
    mu = np.where(rng.uniform(size=n) < .5, rng.uniform(.08, .22, n), rng.uniform(.9, 1.2, n))
    d = np.where(rng.uniform(size=n) < .5, rng.uniform(.07, .17, n), rng.uniform(.55, .8, n))
    models = [g.mk(a, b) for a, b in zip(mu, d)]
    past = {k: make_throw(models, rng, k.rstrip("12")) for k in ("drop1", "drop2", "obl1", "obl2")}
    nxt = {k: make_throw(models, rng, k, prefix_only=True) for k in ("drop", "oblique")}
    return dict(mu=mu, d=d, models=models, past=past, nxt=nxt)


def cheap_update(model, thr, logpi, rng, gen, Np=NP_CHEAP, m_min=M_MIN):
    """One IS pass per past throw with a STRATIFIED minimum: every class gets m_min launch-state draws (so every class has an
    unbiased, if noisy, marginal-likelihood estimate; no class is silently truncated), the remaining Np-144*m_min draws go to
    classes ~ 1/2 cat + 1/2 pi_prev.  Given c, all draws are iid from q(z|c), so p_hat(y|c) = mean_i p(z_i)L_i/q(z_i|c) over the
    draws of class c.  Returns log pi_new ∝ log pi_prev + log p_hat (n,144) and diagnostics."""
    n = len(thr["obs"])
    t0 = time.time()
    x = torch.tensor(f.features(thr["obs"], thr["mask"]))
    with torch.no_grad():
        cat = torch.softmax(model.cat(model.enc(x)), -1).numpy().astype(float)
    pi = np.exp(logpi - logsumexp(logpi, 1, keepdims=True))
    qc = .5*cat + .5*pi
    n_ad = Np - 144*m_min
    cls_ad = np.stack([rng.choice(144, n_ad, p=q/q.sum()) for q in qc])
    cls = np.concatenate([np.repeat(np.arange(144), m_min)[None].repeat(n, 0), cls_ad], 1).ravel()
    obj = np.repeat(np.arange(n), Np)
    z, lp, lq = g7.propose(model, thr["obs"][obj], thr["mask"][obj], cls, rng, gen)
    t_prop = time.time()-t0
    t1 = time.time()
    ll, _, _ = g7.simulate_ll([g.POOL[c//12][c % 12] for c in cls], z, thr["obs"][obj], thr["mask"][obj], rng, g.T)
    t_sim = time.time()-t1
    lr = lp + ll - lq                                         # log p(z)L/q(z|c) per draw
    key = obj*144 + cls
    cnt = np.bincount(key, minlength=n*144).reshape(n, 144)
    mx = np.full(n*144, -np.inf); np.maximum.at(mx, key, lr)
    mx = np.where(np.isfinite(mx), mx, 0.)
    sm = np.zeros(n*144); np.add.at(sm, key, np.exp(lr - mx[key]))
    with np.errstate(divide="ignore"):
        lhat = (np.log(sm) + mx).reshape(n, 144) - np.log(cnt)
    lpost = logpi + lhat
    lpost = lpost - logsumexp(lpost, 1, keepdims=True)
    post = np.exp(lpost)
    only_min = cnt == m_min                                   # adaptive stage never picked this class
    return lpost, dict(zero_estimate_classes=float(np.mean((~np.isfinite(lhat)).sum(1))),
                       only_min_classes=float(only_min.sum(1).mean()), mass_in_only_min=float((post*only_min).sum(1).mean()),
                       mass_in_only_min_per_obj=(post*only_min).sum(1).tolist(), only_min_mask=only_min,
                       t_prop=t_prop, t_sim=t_sim, sec=time.time()-t0)


def exact_update(model, thr, rng, gen):
    t0 = time.time()
    cll = g7.class_loglik(model, thr, 64, rng, gen)
    return cll, time.time()-t0


def ident(logpost, mu, d):
    post = softmax(logpost, 1)
    mh, dh = post@LMU, post@DD
    e_mu, e_d = np.abs(mh-np.log(mu)), np.abs(dh-d)
    out = dict(logmu_err=float(e_mu.mean()), d_err=float(e_d.mean()),
               logmu_sd=float(np.mean(np.sqrt(np.maximum(post@LMU**2-mh**2, 0)))),
               d_sd=float(np.mean(np.sqrt(np.maximum(post@DD**2-dh**2, 0)))))
    lo_mu, lo_d = mu < .5, d < .4
    for name, m in (("mu_low", lo_mu), ("mu_high", ~lo_mu)):
        out[f"logmu_err_{name}"] = float(e_mu[m].mean())
    for name, m in (("d_low", lo_d), ("d_high", ~lo_d)):
        out[f"d_err_{name}"] = float(e_d[m].mean())
    return out, dict(logmu_abs=e_mu.tolist(), d_abs=e_d.tolist())


def forecast(model, ob, logprior, N, rng, gen, oracle=False):
    res, sec = {}, 0.
    for k, thr in ob["nxt"].items():
        t0 = time.time()
        r = g7.next_throw(model, thr, logprior, N, rng, gen, true_models=ob["models"] if oracle else None)
        sec += time.time()-t0
        for m, v in r.items():
            res[f"{k}_{m}"] = float(np.mean(v))
    for m in ("energy", "x_cover", "z_cover", "x_width", "z_width", "x_iscore", "z_iscore"):
        res[m] = float(np.mean([res[f"{k}_{m}"] for k in ob["nxt"]]))
    res["forecast_sec"] = sec
    return res


def true_cell(ob):
    """Nearest grid cell to each object's true (continuous, off-grid) material."""
    return (np.abs(np.log(g.PF_MU)[None]-np.log(ob["mu"])[:, None]).argmin(1)*12 + np.abs(g.PF_D[None]-ob["d"][:, None]).argmin(1))


DESIGNS = {"DD": ("drop1", "drop2"), "DO": ("drop1", "obl1"), "OO": ("obl1", "obl2")}


def profile(log):
    """Where does gate-7 carry-over time go?  One block, 48 objects, one past throw."""
    model = a6.load_models()[8100]
    ob = objects(58900, 48)
    rng, gen = np.random.default_rng(1), torch.Generator().manual_seed(1)
    thr = ob["past"]["drop1"]
    n, K = 48, 64
    cls = np.tile(np.repeat(np.arange(144), K), n)
    obj = np.repeat(np.arange(n), 144*K)
    t0 = time.time(); z, lp, lq = g7.propose(model, thr["obs"][obj], thr["mask"][obj], cls, rng, gen); tp = time.time()-t0
    t0 = time.time(); g7.simulate_ll([g.POOL[c//12][c % 12] for c in cls], z, thr["obs"][obj], thr["mask"][obj], rng, g.T); ts = time.time()-t0
    _, info = cheap_update(model, thr, np.zeros((n, 144)), rng, gen)
    info = {k: v for k, v in info.items() if k not in ("only_min_mask", "mass_in_only_min_per_obj")}
    t0 = time.time(); g7.next_throw(model, ob["nxt"]["drop"], np.zeros((n, 144)), 4096, rng, gen); tf = time.time()-t0
    out = dict(exact_rows=int(len(cls)), exact_flow_sec=tp, exact_sim_sec=ts, cheap_rows=n*NP_CHEAP, cheap=info, fresh4096_one_type_sec=tf)
    log(out)
    (HERE/"gate8_profile.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")


def run(log):
    model = a6.load_models()[8100]
    rows, per_obj = [], []
    t_all = time.time()
    for block in range(4):
        ob = objects(58000+block, 48)
        rng = np.random.default_rng(58100+block)
        gen = torch.Generator().manual_seed(58100+block)
        u0 = np.full((48, 144), -np.log(144))
        # cheap chains (DD and DO share the first drop update)
        cheap, cinfo = {}, {}
        first = {}
        for k in ("drop1", "obl1"):
            first[k] = cheap_update(model, ob["past"][k], u0, rng, gen)
        for dname, (a, b) in DESIGNS.items():
            lp1, i1 = first[a]
            lp2, i2 = cheap_update(model, ob["past"][b], lp1, rng, gen)
            tc = true_cell(ob)
            om = i1["only_min_mask"] | i2["only_min_mask"]
            cinfo[dname] = dict(update_sec=i1["sec"]+i2["sec"],
                                zero_estimate_classes=[i1["zero_estimate_classes"], i2["zero_estimate_classes"]],
                                only_min_classes=[i1["only_min_classes"], i2["only_min_classes"]],
                                mass_in_only_min=[i1["mass_in_only_min"], i2["mass_in_only_min"]],
                                true_cell_only_min_frac=float(om[np.arange(48), tc].mean()),
                                per_obj_mass_in_only_min=i2["mass_in_only_min_per_obj"])
            cheap[dname] = lp2
        # exact per-class marginal likelihoods (reference)
        exact_cll, exact_sec = {}, {}
        for k in ("drop1", "drop2", "obl1", "obl2"):
            exact_cll[k], exact_sec[k] = exact_update(model, ob["past"][k], rng, gen)
        for dname, (a, b) in DESIGNS.items():
            for meth, lpst, usec in (("cheap", cheap[dname], cinfo[dname]["update_sec"]),
                                     ("exact", exact_cll[a]+exact_cll[b], exact_sec[a]+exact_sec[b])):
                idt, po = ident(lpst, ob["mu"], ob["d"])
                fc = forecast(model, ob, lpst, 256, rng, gen)
                row = dict(block=block, arm=f"carry_{meth}", design=dname, N=256, update_sec=usec,
                           sec=usec+fc["forecast_sec"], **idt, **fc)
                if meth == "cheap":
                    row.update(**{f"cheap_{k}": v for k, v in cinfo[dname].items() if k != "update_sec"})
                row["true_cell_mass"] = float(np.mean(softmax(lpst, 1)[np.arange(48), true_cell(ob)]))
                rows.append(row)
                per_obj.append(dict(block=block, arm=row["arm"], design=dname, mu=ob["mu"].tolist(), d=ob["d"].tolist(), **po))
                log(block, row["arm"], dname, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                               if k in ("sec", "energy", "x_cover", "z_cover", "logmu_err", "d_err", "update_sec")})
        for N in (256, 1024, 4096, 8192):
            fc = forecast(model, ob, u0, N, rng, gen)
            idt, _ = ident(u0, ob["mu"], ob["d"])
            rows.append(dict(block=block, arm="fresh", design="-", N=N, update_sec=0., sec=fc["forecast_sec"], **idt, **fc))
            log(block, "fresh", N, {k: round(rows[-1][k], 4) for k in ("sec", "energy", "x_cover", "z_cover")})
        fc = forecast(model, ob, u0, 256, rng, gen, oracle=True)
        rows.append(dict(block=block, arm="oracle", design="-", N=256, update_sec=0., sec=fc["forecast_sec"], **fc))
        log(block, "oracle", {k: round(rows[-1][k], 4) for k in ("sec", "energy", "x_cover", "z_cover")})
        (HERE/"gate8.json").write_text(json.dumps(dict(rows=rows, per_object=per_obj, total_sec=time.time()-t_all), indent=1)+"\n",
                                       encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    dict(run=run, profile=profile)[sys.argv[1]](log)
