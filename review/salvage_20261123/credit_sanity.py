"""Spectral credit (spectral-credit-stsa docs/01) sanity: per-design step-size tuning.

Same 19-state random walk as src/spectral_credit.py, but
  * every design gets its own alpha chosen on validation seeds (0..19),
  * reported AUC uses disjoint test seeds (1000..1099),
  * offline compound-return update per episode (V frozen inside an episode; same for all designs),
  * also checks: w>=0, sum w=1  =>  h_k = sum_{n>=k} w_n is non-increasing with h_1 = 1.
numpy only. Writes credit_sanity.json next to this file.
"""
import json
from pathlib import Path
import numpy as np

N = 19
V_TRUE = np.array([(s - (N + 1) / 2) / ((N + 1) / 2) for s in range(1, N + 1)])
KMAX = 64
EPISODES = 100
VAL = range(0, 20)
TEST = range(1000, 1100)
ALPHAS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3)


def episode(rng):
    s = (N + 1) // 2
    st = [s]
    while True:
        s += 1 if rng.random() < 0.5 else -1
        st.append(s)
        if s == 0 or s == N + 1:
            return np.array(st)


def w_nstep(n):
    w = np.zeros(KMAX); w[n - 1] = 1; return w


def w_lambda(lam):
    n = np.arange(1, KMAX + 1)
    w = (1 - lam) * lam ** (n - 1); w[-1] += lam ** KMAX
    return w / w.sum()


def w_hann(K):
    n = np.arange(1, K + 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * (n - 0.5) / K)
    w = np.zeros(KMAX); w[:K] = win / win.sum(); return w


def run(w, seed, alpha):
    rng = np.random.default_rng(seed)
    V = np.zeros(N + 2)
    nn = np.arange(1, KMAX + 1)
    out = np.empty(EPISODES)
    for e in range(EPISODES):
        st = episode(rng)
        T = len(st) - 1                     # transitions; st[T] terminal
        R = 1.0 if st[-1] == N + 1 else -1.0
        t = np.arange(T)[:, None]
        idx = t + nn[None, :]               # (T,K) index of bootstrap state
        term = idx >= T
        boot = np.where(term, R, V[st[np.minimum(idx, T)]])   # gamma=1, reward only at the end
        G = boot @ w                        # compound return per t
        np.add.at(V, st[:T], alpha * (G - V[st[:T]]))
        V[0] = V[N + 1] = 0
        out[e] = np.sqrt(np.mean((V[1:N + 1] - V_TRUE) ** 2))
    return out


def auc(w, seeds, alpha):
    return float(np.mean([run(w, s, alpha).mean() for s in seeds]))


designs = {f"TD(lambda={l})": w_lambda(l) for l in (0.0, 0.4, 0.6, 0.8, 0.9, 0.95)}
designs.update({f"n-step n={n}": w_nstep(n) for n in (2, 4, 8, 16)})
designs.update({f"Hann K={k}": w_hann(k) for k in (4, 8, 12, 16, 24)})

# monotonicity check of h for random nonnegative weights
rng = np.random.default_rng(7)
mono = True
for _ in range(2000):
    w = rng.random(KMAX) * (rng.random(KMAX) < 0.3); w /= max(w.sum(), 1e-12)
    h = np.cumsum(w[::-1])[::-1]
    mono &= bool(np.all(np.diff(h) <= 1e-12)) and abs(h[0] - 1) < 1e-9

res = {}
for name, w in designs.items():
    val = {a: auc(w, VAL, a) for a in ALPHAS}
    a_best = min(val, key=val.get)
    seeds_test = [run(w, s, a_best).mean() for s in TEST]
    res[name] = dict(alpha=a_best, val_auc=val, test_auc=float(np.mean(seeds_test)),
                     test_se=float(np.std(seeds_test) / np.sqrt(len(TEST))), per_seed=seeds_test)
    print(f"{name:18s} alpha*={a_best:<6} test AUC={res[name]['test_auc']:.4f} +- {res[name]['test_se']:.4f}", flush=True)

# original setting reproduction (alpha=0.02 for all) on test seeds
orig = {n: auc(designs[n], TEST, 0.02) for n in ("TD(lambda=0.8)", "Hann K=12", "n-step n=4", "TD(lambda=0.0)")}
best_lam = min((n for n in res if n.startswith("TD")), key=lambda n: res[n]["test_auc"])
best_hann = min((n for n in res if n.startswith("Hann")), key=lambda n: res[n]["test_auc"])
d = np.array(res[best_hann]["per_seed"]) - np.array(res[best_lam]["per_seed"])
summary = dict(h_monotone_for_nonneg_w=bool(mono), orig_alpha002_test=orig, best_lambda=best_lam, best_hann=best_hann,
               paired_diff_hann_minus_lambda=float(d.mean()),
               paired_ci95=[float(d.mean() - 1.96 * d.std() / np.sqrt(len(d))), float(d.mean() + 1.96 * d.std() / np.sqrt(len(d)))])
print(json.dumps(summary, indent=1))
for v in res.values():
    v.pop("per_seed")
Path(__file__).with_name("credit_sanity.json").write_text(json.dumps(dict(results=res, summary=summary), indent=1), encoding="utf-8")
