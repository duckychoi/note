"""Three separate PF learning tests on fixed public Depth Pro 2-S features."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUT = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval/pf42-trials"
GRID = np.linspace(-1.6, 1.6, 33)
SEEDS = (421327, 421328, 421329)
DATASETS = ("hammer", "ibims", "diode", "diode_outdoor")


def logsumexp(x, axis=None, keepdims=False):
    return np.logaddexp.reduce(x, axis=axis, keepdims=keepdims)


def load(name):
    with np.load(OUT / f"{name}.npz", allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def labels(log_ratio):
    return np.clip(np.rint((log_ratio-GRID[0])/(GRID[1]-GRID[0])), 0, len(GRID)-1).astype(int)


def normal_log(x, mu, sd):
    return -.5*((x-mu)/sd)**2 - np.log(sd*np.sqrt(2*np.pi))


def calibrate(cal):
    err = np.log(cal["y"]/cal["truth"][:, None])
    valid = np.isfinite(err)
    cls = abs(err) > .035
    split = [err[valid & (cls == i)] for i in (0, 1)]
    mean = np.array([np.median(r) for r in split])
    spread = np.array([np.sqrt(np.mean((r-m)**2)) for r, m in zip(split, mean)]).clip([.012, .060], None)
    init = np.bincount(cls[valid[:, 0], 0].astype(int), minlength=2) + 1.
    both = valid[:, :-1] & valid[:, 1:]
    trans = np.bincount(2*cls[:, :-1][both].astype(int)+cls[:, 1:][both], minlength=4).reshape(2, 2)+1.
    counts = trans.astype(int)-1
    return mean, spread, init/init.sum(), trans/trans.sum(axis=1, keepdims=True), counts


def emissions(data, sensor):
    mu, sd = sensor[:2]
    obs = np.log(data["y"]/data["base"][:, None])
    z = normal_log(obs[:, :, None, None] - GRID[None, None, :, None], mu, sd)
    return np.where(np.isfinite(obs)[:, :, None, None], z, 0.)


def likelihood(e, sensor, particles=0, rng=None, independent=False, defensive=False):
    """Sum the modes exactly, or sample their population counts under bootstrap PF."""
    init, transition = sensor[2:4]
    steps = e.shape[1]
    prior = np.broadcast_to(np.log(init), e[:, 0].shape).copy()
    lp = np.zeros(e[:, 0, :, 0].shape)
    marg = init.copy()
    for t in range(steps):
        if independent:
            prior = np.log(marg)
            marg = marg @ transition
        if particles:
            proposal = np.exp(prior)
            if defensive:
                proposal = .5*proposal+.25
            n = rng.binomial(particles, np.minimum(proposal[..., 1], 1.))
            with np.errstate(divide="ignore"):
                frequencies = np.log(np.stack((particles-n, n), axis=-1)/particles)
            prior = frequencies+prior-np.log(proposal) if defensive else frequencies
        scores = prior+e[:, t]
        normalizer = logsumexp(scores, axis=-1)
        lp += normalizer
        post = scores-normalizer[..., None]
        if not independent:
            prior = logsumexp(post[..., :, None]+np.log(transition), axis=-2)
    return lp


def math_checks():
    # One particle and two modes: enumerate all four paths to test unbiased evidence,
    # log Jensen, and the derivative of the *expected sampled objective*.
    init = np.array([.7, .3])
    transition = np.array([[.9, .1], [.2, .8]])
    emission = np.array([[[[.9, .2], [.4, .6]], [[.2, .95], [.5, .3]]]])
    sensor = (None, None, init, transition)
    exact = np.exp(likelihood(np.log(emission), sensor))[0]
    values, weights = [], []
    for i in range(2):
        for j in range(2):
            values.append(emission[0, 0, :, i]*emission[0, 1, :, j])
            weights.append(init[i]*transition[i, j])
    np.testing.assert_allclose(np.array(weights) @ np.array(values), exact, atol=1e-14)
    rng = np.random.default_rng(1)
    p = np.tile(emission, (100000, 1, 1, 1))
    np.testing.assert_allclose(np.exp(likelihood(np.log(p), sensor, 1, rng)).mean(0), exact, atol=.002)
    np.testing.assert_allclose(np.exp(likelihood(np.log(p), sensor, 2, rng, defensive=True)).mean(0), exact, atol=.002)
    # Each r may use independent mode draws. Enumerate their joint law.
    vals = np.array(values)
    draws = np.stack(np.meshgrid(vals[:, 0], vals[:, 1], indexing="ij"), axis=-1).reshape(-1, 2)
    prob = np.outer(weights, weights).ravel()
    theta = torch.tensor([.4, -.1], dtype=torch.float64, requires_grad=True)
    z = theta.softmax(0)
    estimate = torch.as_tensor(draws) @ z
    exact_z = torch.as_tensor(exact) @ z
    sample_loss = -(torch.as_tensor(prob) @ estimate.log())
    exact_loss = -exact_z.log()
    assert sample_loss > exact_loss
    grad = torch.autograd.grad(sample_loss, theta, retain_graph=True)[0].numpy()
    exact_grad = torch.autograd.grad(exact_loss, theta)[0].numpy()
    def finite(t):
        pi = np.exp(t-logsumexp(t))
        return -prob @ np.log(draws @ pi)
    x = theta.detach().numpy()
    step = np.array([1e-5, 0])
    np.testing.assert_allclose(grad[0], (finite(x+step)-finite(x-step))/(2e-5), atol=2e-10)
    return dict(exact=exact.tolist(), jensen_gap=float((sample_loss-exact_loss).detach()),
                sampled_gradient=grad.tolist(), exact_gradient=exact_grad.tolist())


def class_head(dim):
    return nn.Sequential(nn.Linear(dim, 48), nn.SiLU(), nn.Linear(48, len(GRID)))


def balanced(data, batch, rng):
    groups = [np.flatnonzero(data["group"] == g) for g in np.unique(data["group"])]
    chosen = rng.integers(len(groups), size=batch)
    indices = np.empty(batch, dtype=int)
    for i, g in enumerate(groups):
        use = chosen == i
        indices[use] = rng.choice(g, int(use.sum()))
    return indices


def train_head(seed, sx, rx, syn_label, synthetic_batches, real_batches, prior=None, observation=None, target=None):
    torch.manual_seed(seed)
    net = class_head(sx.shape[1])
    if prior is not None:
        net.load_state_dict(prior.state_dict())
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.001)
    for i, a in enumerate(synthetic_batches):
        loss = F.cross_entropy(net(sx[a]), syn_label[a])
        if real_batches is not None:
            b = real_batches[i]
            logits = net(rx[b]).log_softmax(-1)
            if observation is not None:
                loss = .5*(loss-torch.logsumexp(logits+observation[b], dim=-1).mean())
            else:
                loss = .5*(loss-(logits*target[b]).sum(dim=-1).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return net.eval()


def posterior(log_prior, evidence):
    combined = log_prior+evidence
    return np.exp(combined-logsumexp(combined, axis=-1, keepdims=True))


def decision(p):
    weights = p*np.exp(-GRID)
    limit = .5*weights.sum(axis=-1, keepdims=True)
    ids = (weights.cumsum(axis=-1) < limit).sum(axis=-1).clip(0, len(GRID)-1)
    return np.exp(GRID[ids])


@torch.no_grad()
def predict(net, x):
    return torch.cat([net(part).log_softmax(-1) for part in x.split(8192)]).numpy()


def scores(data, k):
    valid = np.isfinite(data["truth"]) & (data["truth"] >= .25) & (data["truth"] <= 30)
    errors = abs(data["base"][valid]*np.broadcast_to(k, data["base"].shape)[valid]/data["truth"][valid]-1)
    groups = {}
    unit = data["key"] if "key" in data else data["group"]
    for group in np.unique(data["group"][valid]):
        use = data["group"][valid] == group
        keys, inverse = np.unique(unit[valid][use], return_inverse=True)
        metrics = np.bincount(inverse, weights=errors[use], minlength=len(keys))/np.bincount(inverse)
        groups[str(group)] = float(metrics.mean())
    return groups


def score_arm(data, correction):
    return {key: scores(s, correction(key)) for key, s in data.items()}


def teacher_report(held, p):
    valid = np.isfinite(held["truth"]) & (held["truth"] >= .25) & (held["truth"] <= 30)
    true = labels(np.log(held["truth"][valid]/held["base"][valid]))
    nll = -np.log(p[valid, true].clip(1e-300))
    families = held["group"][valid]
    return dict(nll=float(np.mean([nll[families == g].mean() for g in np.unique(families)])),
                absrel=scores(held, decision(p)),
                log_clip_count=int((abs(np.log(held["truth"][valid]/held["base"][valid])) > GRID[-1]).sum()))


def teacher_diagnostic(cal, held, sensor):
    out = {}
    for name, data in (("cal", cal), ("held", held)):
        err = np.log(data["y"]/data["truth"][:, None])
        observed = np.isfinite(err)
        bad = abs(err) > .035
        both = observed[:, :-1] & observed[:, 1:]
        counts = np.bincount(2*bad[:, :-1][both].astype(int)+bad[:, 1:][both], minlength=4)
        out[name] = dict(observed=int(observed.sum()), errors=int((observed & bad).sum()),
                         transitions=counts.reshape(2, 2).tolist())
    flat_prior = np.full((len(held["base"]), len(GRID)), -np.log(len(GRID)))
    e = emissions(held, sensor)
    for name, independent in (("iid_flat", True), ("hmm_flat", False)):
        out[name] = teacher_report(held, posterior(flat_prior, likelihood(e, sensor, independent=independent)))
    return out


def class_research(data, sensor, datasets):
    syn, fit, held = [data[k] for k in ("synthetic", "fit", "held")]
    groups = {k: data[k] for k in datasets}
    mu = syn["features"].mean(0)
    sd = syn["features"].std(0).clip(.03)
    x = {k: torch.from_numpy(np.clip((s["features"]-mu)/sd, -8, 8)) for k, s in data.items() if k != "cal"}
    se = torch.as_tensor(labels(np.log(syn["truth"]/syn["base"])))
    median = np.nanmedian(np.log(fit["y"]/fit["base"][:, None]), axis=1)
    hard = np.eye(len(GRID))[labels(median)]
    ef, eh = emissions(fit, sensor), emissions(held, sensor)
    exact = likelihood(ef, sensor)
    independent = likelihood(ef, sensor, independent=True)
    held_med = np.nanmedian(np.log(held["y"]/held["base"][:, None]), axis=1)
    hei = likelihood(eh, sensor, independent=True)
    he = likelihood(eh, sensor)
    runs = []
    output = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        batch_s = rng.integers(len(se), size=(660, 192))
        batch_r = [balanced(fit, 192, rng) for _ in range(300)]
        prior = train_head(seed, x["synthetic"], None, se, batch_s[:360], None)
        prior_real = predict(prior, x["fit"])
        prior_held = predict(prior, x["held"])
        arms = {}
        times = {}
        errors = {}
        for n in (0, 16, 64):
            start = time.perf_counter()
            arms[n] = likelihood(ef, sensor, n, rng)
            times[n] = time.perf_counter()-start
            errors[n] = float(np.mean(abs(arms[n]-exact)))
        defensive_rng = np.random.default_rng(seed+500)
        start = time.perf_counter()
        defense = likelihood(ef, sensor, 64, defensive_rng, defensive=True)
        times["defensive64"] = time.perf_counter()-start
        errors["defensive64"] = float(np.mean(abs(defense-exact)))
        z_exact = logsumexp(prior_real+exact, axis=1)
        evidence_errors = {name: float(np.mean(abs(logsumexp(prior_real+v, axis=1)-z_exact)))
                           for name, v in (("pf16", arms[16]), ("pf64", arms[64]), ("defensive64", defense))}
        soft_exact = posterior(prior_real, exact)
        hard_exact = np.eye(len(GRID))[labels(np.log(decision(soft_exact)))]
        table = dict(loss_exact=(exact, None), loss_pf16=(arms[16], None), loss_pf64=(arms[64], None),
                     loss_defensive=(defense, None),
                     hard=(None, hard), teacher_iid=(None, posterior(prior_real, independent)),
                     teacher_exact=(None, soft_exact), teacher_hard=(None, hard_exact),
                     teacher_pf=(None, posterior(prior_real, arms[64])),
                     teacher_defensive=(None, posterior(prior_real, defense)))
        models = dict(synthetic_head=prior)
        models["synthetic_equal"] = train_head(seed, x["synthetic"], None, se, batch_s[360:], None, prior)
        for name, (e, target) in table.items():
            models[name] = train_head(seed, x["synthetic"], x["fit"], se, batch_s[360:], batch_r, prior,
                                      None if e is None else torch.from_numpy(e).float(),
                                      None if target is None else torch.from_numpy(target).float())
        rescored = {}
        for name, net in models.items():
            rescored[name] = score_arm(groups, lambda k: decision(np.exp(predict(net, x[k]))))
        teachers = dict(iid=posterior(prior_held, hei), exact=posterior(prior_held, he),
                        pf=posterior(prior_held, likelihood(eh, sensor, 64, rng)),
                        defensive=posterior(prior_held, likelihood(eh, sensor, 64, defensive_rng, defensive=True)))
        teacher_scores = {k: teacher_report(held, p) for k, p in teachers.items()}
        teacher_scores["direct_median"] = dict(nll=None, absrel=scores(held, np.exp(GRID[labels(held_med)])))
        runs.append(dict(seed=seed, models=rescored, times=times, errors=errors,
                         evidence_errors=evidence_errors, teachers=teacher_scores))
        output[seed] = {k: m.state_dict() for k, m in models.items()}
        print("PF losses and teacher", seed, flush=True)
    torch.save(dict(weights=output, mean=mu, sd=sd, grid=GRID), OUT / "classifiers.pt")
    return runs


def parameter_inputs(data):
    # Quantize this single covariate so the posterior decision can be cached per value.
    return np.rint(np.clip(np.log(data["base"])/2, -1, 1)*36).astype(int)


def parameter_research(data, datasets):
    syn, fit = [data[k] for k in ("synthetic", "fit")]
    xs = parameter_inputs(syn)/36
    z = np.log(syn["truth"]/syn["base"]).clip(-1.6, 1.6)
    a0 = np.linalg.lstsq(np.column_stack((np.ones(len(z)), xs)), z, rcond=None)[0]
    a0 = a0.clip([-.8, -.55], [.8, .55])
    x = parameter_inputs(fit)/36
    y = np.nanmedian(np.log(fit["y"]/fit["base"][:, None]), axis=1)
    grouped = [np.flatnonzero(fit["group"] == g) for g in np.unique(fit["group"])]
    axis0 = np.linspace(-.8, .8, 51)
    axis1 = np.linspace(-.55, .55, 51)
    a = np.column_stack([c.ravel() for c in np.meshgrid(axis0, axis1, indexing="ij")])
    prior = -.5*np.sum(((a-a0)/.4)**2, axis=1)
    prior -= logsumexp(prior)
    # Generalized likelihood: family means avoid counting correlated pixels as fully independent.
    terms = [-(8*np.log1p(((a[:, :1]+a[:, 1:]*x[g]-y[g])/.18)**2)).mean(axis=1) for g in grouped]
    total = prior + np.sum(terms, axis=0)
    np.testing.assert_allclose(total, prior+np.sum(list(reversed(terms)), axis=0), atol=1e-12)
    posterior_exact = np.exp(total-logsumexp(total))
    map_a = a[np.argmax(total)]
    dataset = {k: data[k] for k in datasets}

    def evaluate(values, w=None):
        lookup = np.ones(73)
        for i, covariate in enumerate(np.linspace(-1, 1, 73)):
            k = np.exp(values[:, 0]+values[:, 1]*covariate)
            if w is not None:
                ids = np.argsort(k)
                v = w[ids]/k[ids]
                lookup[i] = k[ids[np.searchsorted(v.cumsum(), .5*v.sum())]]
            else:
                lookup[i] = k[0]
        return score_arm(dataset, lambda k: lookup[parameter_inputs(dataset[k])+36])

    raw = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
    prior_mean = raw.detach().clone()
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    optim = torch.optim.AdamW([raw], lr=.008, weight_decay=0.)
    bounds_low = torch.tensor([-.8, -.55], dtype=raw.dtype)
    bounds_high = -bounds_low
    for _ in range(600):
        residual = torch.log1p(((raw[0]+raw[1]*tx-ty)/.18)**2)
        loss = .5*((raw-prior_mean)/.4).square().sum()+8*sum(residual[g].mean() for g in grouped)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            raw.copy_(raw.maximum(bounds_low).minimum(bounds_high))
    result = dict(prior=a0.tolist(), map=map_a.tolist(), adamw=raw.detach().tolist(),
                  exact_map=evaluate(map_a[None]), exact_posterior=evaluate(a, posterior_exact),
                  adamw_map=evaluate(raw.detach().numpy()[None]))
    result["runs"] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n = 256
        ids = rng.choice(len(a), size=n, p=np.exp(prior))
        logw = np.full(n, -np.log(n))
        cumulative = prior.copy()
        ess, resamples = [], 0
        for term in terms:
            cumulative += term
            logw += term[ids]
            logw -= logsumexp(logw)
            ess.append(float(np.exp(-logsumexp(2*logw))))
            if ess[-1] < n/2:
                ids = rng.choice(ids, size=n, p=np.exp(logw))
                logw.fill(-np.log(n))
                resamples += 1
                for _ in range(8):
                    # Out-of-range proposals become self transitions, which keeps symmetry.
                    d = rng.integers(0, 4, size=n)
                    a0i, a1i = np.divmod(ids, len(axis1))
                    n0 = a0i+(d == 0)-(d == 1)
                    n1 = a1i+(d == 2)-(d == 3)
                    safe = (n0 >= 0)&(n0 < len(axis0))&(n1 >= 0)&(n1 < len(axis1))
                    j = np.where(safe, n0*len(axis1)+n1, ids)
                    ids = np.where(np.log(rng.random(n)) < cumulative[j]-cumulative[ids], j, ids)
        w = np.exp(logw)
        result["runs"].append(dict(seed=seed, ess=ess, resamples=resamples, unique=int(len(np.unique(ids))),
                                   results=evaluate(a[ids], w),
                                   mean=(w@a[ids]).tolist(), exact_mean=(posterior_exact@a).tolist()))
    return result


def main(core):
    torch.set_num_threads(2)
    datasets = DATASETS[:2] if core else DATASETS
    data = {name: load(name) for name in ("synthetic", "fit", "cal", "held", *datasets)}
    sensor = calibrate(data["cal"])
    sizes = {k: dict(points=len(s["base"]), groups=np.unique(s["group"]).tolist(),
                     frames=len(np.unique(s.get("key", s.get("scene", s["group"]))))) for k, s in data.items()}
    result = dict(sizes=sizes, baseline=score_arm({k: data[k] for k in datasets}, lambda _: 1.),
                  sensor=[a.tolist() for a in sensor], math=math_checks(),
                  sensor_diagnostic=teacher_diagnostic(data["cal"], data["held"], sensor))
    result["loss_and_teacher"] = class_research(data, sensor, datasets)
    result["parameter"] = parameter_research(data, datasets)
    path = OUT / ("results-core.json" if core else "results.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print("Three PF research tests done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", action="store_true", help="evaluate HAMMER and iBims only")
    main(parser.parse_args().core)
