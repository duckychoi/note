"""Separate synthetic-prior error and common repeated stereo error on disjoint anchors."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pf42_trials as prior


OUT = prior.OUT / "separated-b"
TEMPERATURES = (.1, .25, .5, 1.)
TARGET_GRID = np.linspace(-3.2, 3.2, 129)


def valid(data):
    z = data["truth"]
    return np.isfinite(z) & (z >= .25) & (z <= 30)


def subsets(data, keep):
    return {k: v[keep] for k, v in data.items()}


def scene_weights(group):
    names, inv, counts = np.unique(group, return_counts=True, return_inverse=True)
    return 1. / (len(names)*counts[inv])


def observe(data):
    y = np.log(data["y"].astype(np.float64))
    m = np.nanmedian(y, axis=1)
    span = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
    count = np.isfinite(y).sum(axis=1)
    bin_index = 2*np.digitize(span, [.002, .02]) + (count < 4)
    return y, m, span, bin_index


def raw_priors(data):
    # Reuse the frozen synthetic models that preceded the real adaptation.
    ckpt = torch.load(prior.OUT / "classifiers.pt", weights_only=False, map_location="cpu")
    out = {}
    for name, rows in data.items():
        x = torch.from_numpy(np.clip((rows["features"] - ckpt["mean"])/ckpt["sd"], -8, 8))
        ps = []
        for models in ckpt["weights"].values():
            model = prior.class_head(x.shape[1])
            model.load_state_dict(models["synthetic_head"])
            ps.append(np.exp(prior.predict(model.eval(), x)))
        p = np.mean(ps, axis=0).astype(np.float64)
        mu = p @ prior.GRID
        sd = np.sqrt(np.maximum(p @ prior.GRID**2-mu**2, .05**2))
        out[name] = dict(mu=mu+np.log(rows["base"]), sd=sd)
    return out


def quantiles(values, weights, levels):
    i = np.argsort(values)
    mass = weights[i].cumsum()
    k = np.searchsorted(mass, levels*mass[-1]).clip(0, len(i)-1)
    return values[i[k]]


class Calibrator:
    def __init__(self, data, raw):
        clean = valid(data)
        data, raw = subsets(data, clean), subsets(raw, clean)
        _, medium, _, bins = observe(data)
        truth = np.log(data["truth"].astype(np.float64))
        weights = scene_weights(data["group"])
        e = medium - truth
        self.offsets = {}
        levels = (np.arange(256)+.5)/256
        for i in range(6):
            select = bins == i
            n = int(select.sum())
            step = n/(n+100.)
            local = scene_weights(data["group"][select]) if n else np.array([])
            w = (1-step)*weights
            w[select] += step*local
            self.offsets[i] = np.sort(np.r_[quantiles(e, w, levels), 0.])
        self.centers = np.column_stack((raw["mu"], np.log(data["base"])))
        self.center = weights @ self.centers
        self.scale = np.sqrt(weights @ (self.centers-self.center)**2).clip(.1)
        design = self.design(data, raw)
        self.beta = np.linalg.solve(design.T @ (weights[:, None]*design)+np.diag([0., .12, .12]),
                                    design.T @ (weights*(truth-raw["mu"])))
        residual = truth - raw["mu"]-design @ self.beta
        self.prior_std = float(max(.15, np.sqrt(weights @ residual**2)))

    def design(self, data, raw):
        x = np.column_stack((raw["mu"], np.log(data["base"])))
        return np.column_stack((np.ones(len(x)), (x-self.center)/self.scale))

    def distribution(self, data, raw, power, adjusted):
        _, m, _, bins = observe(data)
        e = np.stack([self.offsets[c] for c in bins])
        z = m[:, None]-e[:, ::-1]
        mu = raw["mu"] + self.design(data, raw) @ self.beta if adjusted else raw["mu"]
        sd = np.full(len(mu), self.prior_std) if adjusted else raw["sd"]
        logw = -.5*power*((z-mu[:, None])/sd[:, None])**2
        logw[(z < np.log(.25)) | (z > np.log(30))] = -np.inf
        # The zero residual is included, and every original median has valid range.
        w = np.exp(logw-prior.logsumexp(logw, axis=1, keepdims=True))
        return z, w


def decision(z, weights):
    weighted = weights*np.exp(-z)
    n = (weighted.cumsum(axis=1) < (.5*weighted.sum(axis=1))[:, None]).sum(1)
    return np.exp(z[np.arange(len(z)), n])


def scores(data, predicted):
    return prior.scores(data, np.asarray(predicted)/data["base"])


def mean_score(data, pred):
    return float(np.mean(list(scores(data, pred).values())))


def tune(cal, raw):
    approach = {name: {str(t): {} for t in TEMPERATURES} for name in ("joint_raw", "joint")}
    for g in np.unique(cal["group"]):
        test = cal["group"] == g
        model = Calibrator(subsets(cal, ~test), subsets(raw, ~test))
        dst, aux = subsets(cal, test), subsets(raw, test)
        for name, temperatures in approach.items():
            for key, by_group in temperatures.items():
                z, w = model.distribution(dst, aux, float(key), name == "joint")
                by_group[str(g)] = mean_score(dst, decision(z, w))
    choices = {name: float(min(vals, key=lambda p: np.mean(list(vals[p].values()))))
               for name, vals in approach.items()}
    return choices, approach


def bias_stats(data, raw):
    use = valid(data)
    data, raw = subsets(data, use), subsets(raw, use)
    y, medium, span, _ = observe(data)
    truth = np.log(data["truth"])
    e = y-truth[:, None]
    error = medium-truth
    w = scene_weights(data["group"])
    individual = e-np.nanmean(e, axis=1)[:, None]
    pair = np.isfinite(e[:, 0]) & np.isfinite(e[:, 2])
    small = span <= .002
    return dict(points=int(use.sum()), prior_log_bias=float(w @ (raw["mu"]-truth)),
                median_stereo_log_bias=float(w @ error), prior_log_rmse=float(np.sqrt(w @ (raw["mu"]-truth)**2)),
                within_point_log_rms=float(np.sqrt(w @ np.nanmean(individual**2, axis=1))),
                aggregate_log_rms=float(np.sqrt(w @ error**2)),
                cross_light_correlation=float(np.corrcoef(e[pair, 0], e[pair, 2])[0, 1]),
                small_span=int(small.sum()), small_span_error_3_5pct=int((small & (abs(error)>.035)).sum()))


def prior_diagnostic(model, data, raw):
    stats = {}
    for name in ("cal", "held"):
        split, p = data[name], raw[name]
        keep = valid(split)
        w = scene_weights(split["group"][keep])
        z = np.log(split["truth"][keep])
        raw_error = p["mu"][keep]-z
        adjusted_error = raw_error+model.design(split, p)[keep] @ model.beta
        stats[name] = dict(raw_bias=float(w@raw_error), adjusted_bias=float(w@adjusted_error),
                           raw_log_rmse=float(np.sqrt(w@(raw_error**2))),
                           adjusted_log_rmse=float(np.sqrt(w@(adjusted_error**2))))
    return stats


def bootstrap(a, b):
    names = sorted(a)
    arr = np.array([[arm[n] for n in names] for arm in (a, b)])
    rng = np.random.default_rng(421334)
    ids = rng.integers(len(names), size=(10000, len(names)))
    diff = 100*(1-arr[0, ids].mean(1)/arr[1, ids].mean(1))
    return dict(relative=float(100*(1-arr[0].mean()/arr[1].mean())), interval=np.quantile(diff, [.025, .975]).tolist())


def bin_targets(z, probability):
    """Deposit each continuous value linearly on adjacent common output bins."""
    grid = TARGET_GRID
    u = np.clip((z-grid[0])/(grid[1]-grid[0]), 0, len(grid)-1)
    low = u.astype(int)
    mix = u-low
    value = np.zeros((len(z), len(grid)), dtype=np.float32)
    rows = np.broadcast_to(np.arange(len(z))[:, None], z.shape)
    np.add.at(value, (rows, low), probability*(1-mix))
    np.add.at(value, (rows, np.minimum(low+1,len(grid)-1)), probability*mix)
    return torch.from_numpy(value)


def train_student(data, target, output=OUT, extra_direct=None, *, soft_controls=None):
    """Same joint budget for a direct continuous target and posterior distribution."""
    syn, fit = data["synthetic"], data["fit"]
    mu, sd = syn["features"].mean(0), syn["features"].std(0).clip(.03)
    x = {n: torch.from_numpy(np.clip((r["features"]-mu)/sd, -8, 8)) for n, r in data.items()}
    r_syn = np.log(syn["truth"]/syn["base"]).astype(float)[:, None]
    sy = bin_targets(r_syn, np.ones(r_syn.shape))
    sm = observe(fit)[1][:, None]-np.log(fit["base"])[:, None]
    dy = bin_targets(sm, np.ones(sm.shape))
    z, p = target
    py = bin_targets(z-np.log(fit["base"])[:, None], p)
    arms = {"direct":dy,"separated":py}
    for name, (depth, mass) in (soft_controls or {}).items():
        arms[name] = bin_targets(depth-np.log(fit["base"])[:, None], mass)
    if extra_direct is not None:
        extra_log = np.log(np.asarray(extra_direct)[:,None]/fit["base"][:,None])
        arms["direct_long"] = bin_targets(extra_log,np.ones(extra_log.shape))
    weights, runs = {}, []
    for seed in prior.SEEDS:
        rng = np.random.default_rng(seed+10)
        ss = rng.integers(len(sy), size=(660,192))
        rr = [prior.balanced(fit,192,rng) for _ in range(300)]
        torch.manual_seed(seed+10)
        model = nn.Sequential(nn.Linear(x["fit"].shape[1],48), nn.SiLU(),nn.Linear(48,len(TARGET_GRID)))
        opt = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.001)
        for ids in ss[:360]:
            loss = -(model(x["synthetic"][ids]).log_softmax(-1)*sy[ids]).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        row = {}
        weights[seed] = {}
        for name, y in arms.items():
            from copy import deepcopy
            trained = deepcopy(model)
            opt = torch.optim.AdamW(trained.parameters(), lr=.0015, weight_decay=.001)
            for a,b in zip(ss[360:],rr):
                syn_loss = -(trained(x["synthetic"][a]).log_softmax(-1)*sy[a]).sum(-1).mean()
                real_loss = -(trained(x["fit"][b]).log_softmax(-1)*y[b]).sum(-1).mean()
                loss = .5*(syn_loss+real_loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            row[name] = {}
            for key in prior.DATASETS:
                p_model = np.exp(prior.predict(trained.eval(),x[key]))
                values = np.broadcast_to(TARGET_GRID,p_model.shape)
                row[name][key] = prior.scores(data[key],decision(values,p_model))
            weights[seed][name] = trained.state_dict()
        runs.append(dict(seed=seed, scores=row))
    torch.save(dict(weights=weights, mean=mu, sd=sd, grid=TARGET_GRID), output/"students.pt")
    return runs


def run():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    data = {n: prior.load(n) for n in ("cal","held","fit")}
    raw = raw_priors(data)
    choices, folds = tune(data["cal"],raw["cal"])
    model = Calibrator(data["cal"],raw["cal"])
    (OUT/"prior-diagnostic.json").write_text(json.dumps(prior_diagnostic(model,data,raw),indent=2),
                                             encoding="utf8")
    test = data["held"]
    m_log = observe(test)[1]
    old_label = prior.labels(m_log-np.log(test["base"]))
    normal = np.exp(m_log)
    baselines = dict(legacy_grid=scores(test,test["base"]*np.exp(prior.GRID[old_label])),
                     direct=scores(test,normal),
                     direct_meter=scores(test,np.nanmedian(test["y"],axis=1)))
    variants = {"sensor": (0.,False), "joint_raw": (choices["joint_raw"],False), "joint": (choices["joint"],True)}
    predictions = {"direct": normal}
    for name,(power,adjusted) in variants.items():
        z,w = model.distribution(test,raw["held"],power,adjusted)
        predictions[name] = decision(z,w)
        baselines[name] = scores(test,predictions[name])
    relative = {k: bootstrap(v,baselines["direct"]) for k,v in baselines.items() if k != "direct"}
    result = dict(teacher=baselines, choices=choices, folds=folds,
                  mean={k:float(np.mean(list(v.values()))) for k,v in baselines.items()}, relative=relative,
                  bias={k:bias_stats(data[k],raw[k]) for k in ("cal","held")},
                  student_ran=False, frames=len(np.unique(test["scene"])))
    np.savez_compressed(OUT/"teacher-depths.npz",group=test["group"],scene=test["scene"],truth=test["truth"],**predictions)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print("B teacher",result["mean"],"relative",relative["joint"],flush=True)
    if result["mean"]["joint"] < result["mean"]["direct"]:
        full = {n:prior.load(n) for n in ("synthetic","fit",*prior.DATASETS)}
        result["students"] = train_student(full,model.distribution(data["fit"],raw["fit"],choices["joint"],True))
        result["student_ran"] = True
        (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
        print("matched student transfer done",flush=True)


if __name__ == "__main__":
    run()
