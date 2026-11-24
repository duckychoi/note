"""Matched-stream, query-limited replication of the #33 streaming regression toy."""
import argparse
import json
from pathlib import Path

import numpy as np


P, K, S, T = 50, 6, 3, 720
SWITCH = 120
MODES = ("fixed", "random", "limited_abs", "limited_signed", "full_credit", "all")
RATES = (.025, .075, .2, .5)


def streams(seeds):
    xs, ys, active = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(T, P))
        beta = np.zeros((T // SWITCH, P))
        for block in beta:
            block[rng.choice(P, S, replace=False)] = rng.normal(size=S)
        weights = beta[np.arange(T) // SWITCH]
        y = (x * weights).sum(axis=1) + .3 * rng.normal(size=T)
        xs.append(x)
        ys.append(y)
        active.append(weights != 0)
    return np.stack(xs), np.stack(ys), np.stack(active)


def simulate(data, mode, rate):
    xs, ys, active = data
    n = len(xs)
    row = np.arange(n)[:, None]
    # Exploration is independent of features, labels and the data-generating RNG.
    rng = np.random.default_rng(12062)
    tie = rng.uniform(0, 1e-10, (n, P))
    fixed = np.argsort(tie, axis=1)[:, :K]
    weight, credit = np.zeros((n, P)), np.zeros((n, P))
    squared, recall = np.zeros((n, T)), np.zeros((n, T))
    for t in range(T):
        priority = rng.random((n, P))
        if mode == "all":
            chosen = np.broadcast_to(np.arange(P), (n, P))
        elif mode == "fixed":
            chosen = fixed
        elif mode == "random":
            chosen = np.argsort(priority, axis=1)[:, :K]
        else:
            best = np.argsort(-(np.abs(credit) + tie), axis=1)[:, :K-2]
            priority[row, best] = 2.0
            probe = np.argsort(priority, axis=1)[:, :2]
            chosen = np.concatenate((best, probe), axis=1)
        # This is the only feature read by the budget-limited learners.
        values = xs[row, t, chosen]
        prediction = (values * weight[row, chosen]).sum(axis=1)
        err = ys[:, t] - prediction
        weight[row, chosen] += rate / np.sqrt(chosen.shape[1]) * err[:, None] * values
        squared[:, t] = err**2
        recall[:, t] = active[row, t, chosen].sum(axis=1) / S
        if mode == "full_credit":
            credit = .995 * credit + .2 * np.abs(xs[:, t] * err[:, None])
        elif mode in ("limited_abs", "limited_signed"):
            credit *= .995
            signal = np.abs(values * err[:, None]) if mode == "limited_abs" else values * ys[:, t, None]
            # EMA per observation prevents frequent selection from increasing credit by itself.
            credit[row, chosen] += .2 * (signal - credit[row, chosen])
    return squared.mean(axis=1), recall.mean(axis=1)


def stats(values):
    rng = np.random.default_rng(98)
    boot = rng.choice(values, (5000, len(values))).mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())


def experiment():
    train = streams(range(100, 112))
    test = streams(range(500, 548))
    scores = {}
    record = {}
    for mode in MODES:
        validation = [float(simulate(train, mode, lr)[0].mean()) for lr in RATES]
        rate = RATES[np.argmin(validation)]
        scores[mode], coverage = simulate(test, mode, rate)
        record[mode] = dict(reads=P if mode in ("all", "full_credit") else K,
                            chosen_lr=rate, validation_mse=validation,
                            mse=stats(scores[mode]), recall=stats(coverage))
    return dict(settings=dict(P=P, K=K, S=S, T=T, switch=SWITCH,
                              lr_candidates=RATES, validation_seeds=[100,111], test_seeds=[500,547]),
                methods=record, paired_mse_change={base:stats(scores["limited_signed"] - scores[base])
                for base in ("fixed","random","limited_abs","full_credit","all")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(experiment(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
