"""Audit a finite noise grid on analytic and frozen neural motion priors (CPU)."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/v2/il"))
from common import DATASET, FEAT_DIM, FlowNet


DEFAULT = (8, 15, 22)
SCALE = 6.0
REPEATS = 32


def snr_schedule():
    # Diffusers squaredcos_cap_v2, 50 training steps, zero-based indices.
    alpha = np.cos((np.arange(51) / 50 + .008) / 1.008 * np.pi / 2) ** 2
    beta = np.minimum(1 - alpha[1:] / alpha[:-1], .999)
    cumulative = np.cumprod(1 - beta)
    return cumulative / (1 - cumulative)


def level_terms(eigenvalues, gamma, c):
    z = gamma * eigenvalues
    noise_variance = (z / (1 + z)) ** 2
    factor = 1 + 2 * c * noise_variance
    coefficient = c * gamma / ((1 + z) ** 2 * factor)
    log_constant = -.5 * np.log(factor).sum()
    return coefficient, log_constant


def model_terms(eigenvalues):
    terms = []
    for gamma in snr_schedule():
        z = gamma * eigenvalues
        normalizer = np.maximum(np.mean(z / (1 + z)), 1e-4)
        c = SCALE / (3 * len(z) * normalizer)
        terms.append(level_terms(eigenvalues, gamma, c)[0])
    return np.stack(terms)


def fit_grid(eigenvalues, terms):
    # Same three distinct timesteps. Minimize log-slope dispersion on TRAIN axes.
    grids = np.array(list(combinations(range(50), 3)))
    slopes = eigenvalues * terms[grids].sum(axis=1)
    spread = np.std(np.log(slopes), axis=1)
    return tuple(int(i) for i in grids[np.argmin(spread)])


def monte_carlo_check():
    lambdas = np.array([.1, 10.])
    grid = snr_schedule()[list(DEFAULT)]
    points = np.array([[0., 0.], [np.sqrt(.1), 0.], [0., np.sqrt(10.)]])
    rng = np.random.default_rng(77)
    rows = []
    for point in points:
        total_loss = np.zeros(150_000)
        log_expected = 0.
        for gamma in grid:
            z = gamma * lambdas
            c = SCALE / (3 * len(z) * np.mean(z / (1 + z)))
            coeff, log_const = level_terms(lambdas, gamma, c)
            log_expected += log_const - coeff @ point**2
            a = np.sqrt(gamma / (1 + gamma))
            b = np.sqrt(1 / (1 + gamma))
            noise = rng.normal(size=(len(total_loss), 2))
            noisy = a * point + b * noise
            predicted = b * noisy / (a*a*lambdas + b*b)
            total_loss += c * ((predicted - noise)**2).sum(axis=1)
        reward = np.exp(-total_loss)
        se = reward.std(ddof=1) / np.sqrt(len(reward))
        exact = np.exp(log_expected)
        z_error = (reward.mean() - exact) / se
        assert abs(z_error) < 5
        rows.append(dict(point=point.tolist(), exact=exact, sampled=reward.mean(),
                         se=se, z_error=z_error))
    return rows


def read_windows(root, env):
    width = FEAT_DIM[env]
    folder = root / "minari/mujoco" / DATASET[env] / "expert-v0/data"
    with h5py.File(folder / "main_data.hdf5", "r") as store:
        ids = np.random.default_rng(0).permutation(len(store))

        def load(i):
            obs = store[f"episode_{i}"]["observations"][:, :width].astype("float32")
            return np.concatenate((obs[:-1], obs[1:]), axis=1)

        train = np.concatenate([load(i) for i in ids[:10]])
        test = []
        rng = np.random.default_rng(875)
        for i in ids[30:62]:
            windows = load(i)
            if not len(windows):
                raise ValueError(f"Empty test trajectory {i}")
            test.append(windows[rng.choice(len(windows), 24, replace=len(windows) < 24)])
    mean = np.tile(train[:, :width].mean(0), 2)
    scale = np.tile(train[:, :width].std(0) + 1e-3, 2)
    return (train - mean) / scale, (np.stack(test) - mean) / scale, ids, mean, scale


def versions(test):
    first, second = np.split(test, 2, axis=-1)
    return dict(real=test, reverse=np.concatenate((second, first), axis=-1),
                repeat=np.concatenate((first, first), axis=-1))


def preference(scores):
    # Score higher = preferred. Average within independent test trajectories first.
    out = {}
    for variant in ("reverse", "repeat"):
        gap = scores["real"] - scores[variant]
        out[variant] = ((gap > 0) + .5 * (gap == 0)).mean(axis=1)
    return out


def summary(values):
    rng = np.random.default_rng(82)
    samples = rng.choice(values, size=(4000, len(values))).mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist())


def eps_mse(net, x, noise, time):
    # VP sample divided by (a+b) is exactly the linear interpolation used by FlowNet.
    t = x.new_full((len(x),), time)
    xt = (1-time)*noise + time*x
    eps = xt - time * net(xt, t)
    return ((eps-noise)**2).mean(1)


@torch.inference_mode()
def neural_scores(net, train, variants, grids):
    levels = sorted(set().union(*grids.values()))
    variants = {name: torch.from_numpy(x.reshape(-1, x.shape[-1])) for name, x in variants.items()}
    length, d = next(iter(variants.values())).shape
    cal_ids = np.random.default_rng(10).choice(len(train), 1024, replace=False)
    calibration = torch.from_numpy(train[cal_ids])
    results = {name: {v: np.zeros(length) for v in variants} for name in grids}
    normalizers = {}
    times = np.sqrt(snr_schedule()) / (1 + np.sqrt(snr_schedule()))
    for level in levels:
        generator = torch.Generator().manual_seed(600 + level)
        ref = calibration.repeat(8, 1)
        noise = torch.randn(ref.shape, generator=generator)
        normalizers[level] = max(float(eps_mse(net, ref, noise, times[level]).mean()), 1e-4)
    # Share noise across variants and grids at each level. Independent across levels.
    for rep in range(REPEATS):
        losses = {}
        for level in levels:
            noise = torch.randn((length, d), generator=torch.Generator().manual_seed(5000 + 50*rep + level))
            losses[level] = {v: eps_mse(net, x, noise, times[level]).numpy() / normalizers[level]
                             for v, x in variants.items()}
        for name, grid in grids.items():
            for v in variants:
                loss = sum(losses[level][v] for level in grid) / len(grid)
                results[name][v] += np.exp(-SCALE * loss.astype(float)) / REPEATS
    return results, normalizers


def auc(left, right):
    n = len(left)
    ranks = rankdata(np.concatenate((left, right)))
    return (ranks[:n].sum() - n*(n+1)/2) / (n*len(right))


def hopper_policies(root, net, train, test, mean, scale, grids, center, axes, eigenvalues, tree):
    # Actual simulator transitions, including low-performing seeds. Equal windows per policy.
    rng = np.random.default_rng(691)
    d = train.shape[1]
    sources = {"expert": test.reshape(-1, d)}
    velocity = {}
    paths = {}
    expert_speed = float(np.mean(test[:, :, 5]*scale[5] + mean[5]))
    for method in ("amp", "fm_like_state", "smp"):
        for seed in range(5):
            label = f"{method}_s{seed}"
            path = root / "main" / f"Hopper-v5_{label}_feats.npz"
            with np.load(path) as archive:
                features = archive["F"]
            velocity[label] = float(features[:, 5].mean())
            paths[label] = sha256(path.read_bytes()).hexdigest()
            sources[label] = (features - mean) / scale
    n = 256
    blocks = [v[rng.choice(len(v), n, replace=len(v) < n)] for v in sources.values()]
    windows = np.concatenate(blocks)
    neural, _ = neural_scores(net, train, {"windows": windows}, grids)
    scores = {k:v["windows"] for k,v in neural.items()}
    scores["gaussian"] = -(((windows-center) @ axes)**2) @ (1/eigenvalues)
    scores["nearest"] = -tree.query(windows)[0]
    metrics = {}
    for name, values in scores.items():
        by_policy = dict(zip(sources, values.reshape(-1, n)))
        discrimination = {}
        for method in ("amp","fm_like_state","smp"):
            discrimination[method] = [auc(by_policy["expert"], by_policy[f"{method}_s{i}"]) for i in range(5)]
        closeness = [-abs(v-expert_speed) for v in velocity.values()]
        averages = [by_policy[k].mean() for k in velocity]
        metrics[name] = dict(expert_over_policy_auc=discrimination,
                             rho_score_speed_closeness=float(spearmanr(averages, closeness).statistic))
    return dict(n_per_source=n, expert_speed=expert_speed, speed=velocity,
                feature_archive_sha256=paths, metrics=metrics)


def audit_env(root, env):
    train, test, ids, mean, scale = read_windows(root, env)
    d = train.shape[1]
    cov = np.cov(train, rowvar=False)
    raw, axes = np.linalg.eigh(cov)
    eigenvalues = np.maximum(raw, 1e-4)
    terms = model_terms(eigenvalues)
    grids = dict(default=DEFAULT, fitted=fit_grid(eigenvalues, terms),
                 fixed_early=(0, 8, 22), fixed_two_early=(0, 1, 8))
    inputs = versions(test)
    center = train.mean(0, dtype=float)
    squares = {key: ((v - center) @ axes) ** 2 for key, v in inputs.items()}
    energies = {"gaussian": {k: v @ (1 / eigenvalues) for k, v in squares.items()}}
    for name, grid in grids.items():
        energies[name] = {k: v @ terms[list(grid)].sum(0) for k, v in squares.items()}
    ref_tree = cKDTree(train)
    energies["nearest"] = {k: ref_tree.query(v.reshape(-1, d))[0].reshape(test.shape[:2])
                           for k, v in inputs.items()}
    preferences = {name: preference({k: -v for k, v in values.items()})
                   for name, values in energies.items()}
    # Compare all distinct pairs in each actual expert trajectory; ignore reference ties.
    pairs = np.triu_indices(test.shape[1], 1)
    diffs = {k: v["real"][:, pairs[0]] - v["real"][:, pairs[1]] for k, v in energies.items()}
    ranks = {}
    for key in grids:
        valid = diffs["gaussian"] != 0
        product = diffs[key] * diffs["gaussian"]
        ranks[key] = (((product < 0) + .5 * (product == 0)) * valid).sum(1) / valid.sum(1)
    checkpoint = root / "cache" / f"{env}_n10_flow.pt"
    net = FlowNet(d)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    scores, normalizers = neural_scores(net.eval(), train, inputs, grids)
    for key, values in scores.items():
        preferences["net_" + key] = preference({k:v.reshape(test.shape[:2]) for k,v in values.items()})
    preferences["net_change"] = {k:preferences["net_fitted"][k] - preferences["net_default"][k]
                                 for k in ("reverse", "repeat")}
    for fixed in ("fixed_early", "fixed_two_early"):
        preferences["net_fitted_minus_" + fixed] = {
            k:preferences["net_fitted"][k] - preferences["net_" + fixed][k]
            for k in ("reverse","repeat")}
    policies = None
    if env == "Hopper-v5":
        policies = hopper_policies(root, net, train, test, mean, scale, grids, center, axes,
                                   eigenvalues, ref_tree)
    return dict(train_episodes=ids[:10].tolist(), test_episodes=ids[30:62].tolist(),
                train_transitions=len(train), test_shape=list(test.shape),
                checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                eigenvalue_range=[float(raw[0]), float(raw[-1])], floored_axes=int(sum(raw < 1e-4)),
                grids=grids, neural_normalizers=normalizers, saved_hopper_policies=policies,
                gaussian_rank_disagreement={k:summary(v) for k,v in
                                            {**ranks,"change":ranks["fitted"]-ranks["default"]}.items()},
                expert_preference={m:{k:summary(v) for k,v in vals.items()} for m,vals in preferences.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/0.Project/fm34_env"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(monte_carlo=monte_carlo_check(), repeats=REPEATS, environments={})
    for environment in DATASET:
        result["environments"][environment] = audit_env(args.root, environment)
        print(environment, result["environments"][environment]["grids"], flush=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
