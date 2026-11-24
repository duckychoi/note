"""Independent #63 cue-shift diagnostic. All randomness is shared across methods."""
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp, ndtr


OUT = Path(__file__).resolve().parent
METHODS = ("fixed", "quality", "jump_fixed", "jump_quality", "adaptive")
CONDITIONS = ("aligned", "unrelated", "inverted", "midflip")
JUMPS = [40, 80, 120, 160]
WINDOW = np.concatenate([np.arange(t, t + 5) for t in JUMPS])
BINS = 12


def generate(seed, n, coupled=True):
    rng = np.random.default_rng(seed)
    shape = (200, n)
    dx = rng.normal(0, .1, size=shape)
    dx[JUMPS] += rng.choice([-5., 5.], size=(4, n))
    x = dx.cumsum(axis=0)
    u = rng.random(shape)
    if coupled:
        u[JUMPS] = rng.uniform(.85, 1., size=(4, n))
    outlier = rng.random(shape) < .05
    if coupled:
        outlier[JUMPS] = False
    u[outlier] = rng.uniform(0., .15, size=outlier.sum())
    r = np.where(outlier, 100., .05 + 1.95 * (1 - u)**2)
    z = x + np.sqrt(r) * rng.normal(size=shape)
    q = .8 * u + .2 * rng.random(shape)
    mid = q.copy()
    mid[100:] = 1 - mid[100:]
    return x, z, dict(aligned=q, unrelated=rng.random(shape), inverted=1-q, midflip=mid)


def bins(q):
    return np.minimum((q*BINS).astype(int), BINS-1)


def fit_map(z, quality):
    labels = bins(quality)
    table = np.ones(BINS)
    for _ in range(2):
        innovation = np.empty_like(z)
        m = np.zeros(z.shape[1])
        var = np.zeros_like(m)
        for t, obs in enumerate(z):
            pred = var + .02
            gain = pred / (pred + table[labels[t]])
            innovation[t] = (obs-m)**2 - pred
            m += gain * (obs-m)
            var = (1-gain)*pred
        table = np.clip([np.median(innovation[labels == b]) for b in range(BINS)], .02, 5.)
    return table


def mixture_quantile(means, variances, weights, probability):
    sigma = np.sqrt(variances)
    lo = (means - 9*sigma).min(axis=1)
    hi = (means + 9*sigma).max(axis=1)
    for _ in range(29):
        mid = (lo + hi) / 2
        below = (weights*ndtr((mid[:, None]-means)/sigma)).sum(1) < probability
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo+hi)/2


def run(x, z, cue, table, method, scale, intervals=True, sensor_mixture=False):
    count = z.shape[1]
    experts = 3 if method == "adaptive" else 1
    m = np.zeros((count, experts))
    p = np.zeros_like(m)
    expert_weights = np.full_like(m, 1/experts)
    jumping = method.startswith("jump") or method == "adaptive"
    qs = np.array([.02, 25.02]) if jumping else np.array([.02])
    prior = np.array([.975,.025]) if jumping else np.array([1.])
    extra_r = np.zeros_like(qs)
    if sensor_mixture:
        extra_r = np.tile([0.,64.], len(qs))
        qs = np.repeat(qs,2)
        prior = (prior[:,None]*np.array([.96,.04])).ravel()
    errors = np.zeros_like(x)
    coverage = np.zeros(count)
    width = np.zeros(count)
    learned = table[bins(cue)]
    reverse = table[bins(1-cue)] if experts == 3 else None
    for t in range(len(z)):
        if experts == 3:
            r = np.stack((np.ones(count), learned[t], reverse[t]), axis=1)
        elif "quality" in method:
            r = learned[t, :, None]
        else:
            r = np.ones((count,1))
        r = scale*r[:, :, None] + extra_r
        pred = p[:, :, None] + qs
        variance = pred + r
        residual = z[t, :, None, None] - m[:, :, None]
        logprob = np.log(prior) - .5 * (np.log(variance) + residual**2 / variance)
        evidence = logsumexp(logprob, axis=2)
        component = np.exp(logprob - evidence[:, :, None])
        gains = pred / variance
        mus = m[:, :, None] + gains * residual
        ps = (1-gains)*pred
        m = (component * mus).sum(2)
        p = (component * (ps + (mus - m[:, :, None])**2)).sum(2)
        if experts > 1:
            weights = np.log(.98*expert_weights+.02/experts) + evidence
            expert_weights = np.exp(weights - logsumexp(weights, axis=1, keepdims=True))
        estimate = (expert_weights*m).sum(1)
        errors[t] = estimate-x[t]
        if intervals:
            mixture = (expert_weights[:, :, None] * component).reshape(count,-1)
            means = mus.reshape(count,-1)
            variances = ps.reshape(count,-1)
            if means.shape[1] == 1:
                offset = 1.2815515655446004*np.sqrt(variances[:,0])
                low, high = estimate-offset, estimate+offset
            else:
                low = mixture_quantile(means, variances, mixture, .1)
                high = mixture_quantile(means, variances, mixture, .9)
            coverage += (low <= x[t]) & (x[t] <= high)
            width += high-low
    return dict(rmse=np.sqrt(np.mean(errors**2,axis=0)), postjump=np.abs(errors[WINDOW]).mean(0),
                coverage=coverage/len(z), width=width/len(z))


def main():
    _, z_fit, q_fit = generate(5933, 50)
    table = fit_map(z_fit,q_fit["aligned"])
    x_dev,z_dev,q_dev = generate(5944, 60)
    x,z,q = generate(5955, 240)
    output = dict(numpy=np.__version__,scipy=scipy.__version__, table=table.tolist(),
                  dev={}, summary={}, contrasts={}, trajectories={})
    choices = {}
    for method in METHODS:
        dev = {a: float(run(x_dev,z_dev,q_dev["aligned"],table,method,a,False)["rmse"].mean())
               for a in (.5,1.,2.,4.)}
        choices[method] = min(dev,key=dev.get)
        output["dev"][method] = dict(grid=dev,chosen=choices[method])
    bootstrap = np.random.default_rng(613).integers(240,size=(10000,240))
    for condition in CONDITIONS:
        summary, samples = {}, {}
        for method in METHODS:
            result = run(x,z,q[condition],table,method,choices[method])
            summary[method] = {metric:float(arr.mean()) for metric,arr in result.items()}
            samples[method] = {metric:arr.tolist() for metric,arr in result.items()}
        output["summary"][condition] = summary
        output["trajectories"][condition] = samples
        output["contrasts"][condition] = {}
        for baseline in ("jump_fixed","jump_quality"):
            ref = np.array(samples[baseline]["rmse"])
            gain = ref - samples["adaptive"]["rmse"]
            output["contrasts"][condition][baseline] = dict(relative=float(gain.mean()/ref.mean()),
                    ci=np.quantile(gain[bootstrap].mean(1),[.025,.975]).tolist())
    assert all(output["summary"][c]["jump_fixed"] == output["summary"]["aligned"]["jump_fixed"] for c in CONDITIONS)
    (OUT/"filter_results.json").write_text(json.dumps(output, separators=(",",":")),encoding="utf-8")
    print(json.dumps({key:output[key] for key in ("dev","summary","contrasts")},indent=2))


if __name__ == "__main__":
    main()
