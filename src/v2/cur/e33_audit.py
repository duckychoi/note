"""
#33 audit (11-22): does the toy support "RL self-improving feature curation beats using all / fixed features"?

Original (note:origin/rl-feature-curation src/toy_feature_curation.py, single seed): P=50 features, S=3 relevant,
relevant set + coefficients switch every 120 rounds, curator keeps top-k=6 of q = EMA |x_f * e| over ALL features.
Result: curator MSE 2.31 < all 2.92 < fixed 3.03 (oracle 0.72).

Checks
  1 seeds       : 20 seeds, mean +- 95% t-interval.
  2 baselines   : 'all' in the original is LMS with step lr/sqrt(P) (7x smaller than the curator's lr/sqrt(6)).
                  Add all-features learners that are standard for drifting linear regression:
                  all_lms_tuned (step tuned on validation seeds), all_rls (recursive least squares, forgetting factor
                  tuned), all_l1 (online truncated-gradient / soft-threshold LMS, tuned).
  3 'RL'        : the curator has no policy and no reward; it is a screening rule.  Ablations: curator with random
                  selection (q ignored), curator without decay.
  4 acquisition : the original computes |x_f e| for EVERY feature each round, i.e. it observes all 50 features, so
                  'active acquisition' (paying only for chosen features) is not tested.  Honest variant: q updated only
                  for observed (selected) features, epsilon exploration swaps in an unobserved one.
Tuning: hyper-parameters picked on validation seeds 100-104, reported on evaluation seeds 0-19.
Results -> results/v2/e33_audit.json
"""
import os, json
import numpy as np
from scipy import stats

P, S, K, T, SWITCH, NOISE = 50, 3, 6, 720, 120, 0.3


def stream(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P)); Y = np.empty(T); REL = []
    for t in range(T):
        r = np.random.default_rng(seed * 1000 + t // SWITCH)
        rel = r.choice(P, S, replace=False); coef = r.standard_normal(S)
        Y[t] = X[t, rel] @ coef + NOISE * rng.standard_normal(); REL.append(rel)
    return X, Y, REL


def run(method, seed, hp):
    X, Y, REL = stream(seed); rng = np.random.default_rng(seed + 7)
    W = np.zeros(P); q = np.zeros(P); err = np.empty(T); rec = np.empty(T)
    fixedA = rng.choice(P, K, replace=False)
    if method == "all_rls":
        Pm = np.eye(P) * 100.0; lam = hp["lam"]
    for t in range(T):
        x, y = X[t], Y[t]
        if method in ("all_orig", "all_lms_tuned", "all_rls", "all_l1"):
            A = np.arange(P)
        elif method == "fixed":
            A = fixedA
        elif method == "oracle":
            A = REL[t]
        elif method == "cur_random":
            A = rng.choice(P, K, replace=False)
        else:                                   # curator variants
            top = np.argsort(-q)[:K].copy()
            if rng.random() < hp.get("eps", 0.1):
                top[rng.integers(K)] = rng.integers(P)
            A = np.unique(top)
        yhat = x[A] @ W[A]; e = y - yhat; err[t] = e * e; rec[t] = len(set(A) & set(REL[t])) / S
        if method == "all_rls":
            Px = Pm @ x; g = Px / (lam + x @ Px); W += g * e; Pm = (Pm - np.outer(g, Px)) / lam
        elif method == "all_l1":
            W[A] += hp["lr"] * e * x[A]; W = np.sign(W) * np.maximum(np.abs(W) - hp["l1"], 0)
        elif method == "all_lms_tuned":
            W[A] += hp["lr"] * e * x[A]
        else:
            W[A] += (hp.get("lr", 0.05) / np.sqrt(len(A))) * e * x[A]
        if method in ("cur_orig", "cur_nodecay"):
            q = (1.0 if method == "cur_nodecay" else hp.get("decay", 0.995)) * q + 0.2 * np.abs(x * e)
        elif method == "cur_honest":            # only observed features get a usefulness update
            q[A] = hp.get("decay", 0.995) * q[A] + 0.2 * np.abs(x[A] * e)
            q[np.setdiff1d(np.arange(P), A)] *= hp.get("decay", 0.995)
    return float(err.mean()), float(rec.mean())


GRIDS = {"all_lms_tuned": [dict(lr=l) for l in [0.005, 0.01, 0.02, 0.04]],
         "all_rls": [dict(lam=l) for l in [0.9, 0.95, 0.98, 0.99]],
         "all_l1": [dict(lr=l, l1=c) for l in [0.01, 0.02, 0.04] for c in [1e-3, 3e-3, 1e-2]],
         "cur_honest": [dict(eps=e, decay=d) for e in [0.1, 0.3, 0.5] for d in [0.98, 0.995]]}
METHODS = ["all_orig", "fixed", "oracle", "cur_orig", "cur_random", "cur_nodecay", "cur_honest",
           "all_lms_tuned", "all_rls", "all_l1"]


def main():
    out = {}
    print(f"{'method':14s} | MSE mean ± 95% CI (20 seeds) | relevant recall | chosen hp")
    for m in METHODS:
        hp = {}
        if m in GRIDS:
            hp = min(GRIDS[m], key=lambda h: np.mean([run(m, 100 + s, h)[0] for s in range(5)]))
        r = [run(m, s, hp) for s in range(20)]; v = np.array([a for a, _ in r])
        ci = stats.t.ppf(.975, 19) * v.std(ddof=1) / np.sqrt(20)
        out[m] = dict(mse=float(v.mean()), ci=float(ci), recall=float(np.mean([b for _, b in r])), hp=hp, vals=v.tolist())
        print(f"{m:14s} | {v.mean():6.3f} ± {ci:.3f} | {out[m]['recall']:.2f} | {hp}", flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e33_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
