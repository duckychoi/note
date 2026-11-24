"""Aggregate run JSONs: normalized return, MMD, SW2, kNN state-entropy gap vs expert.
usage: python aggregate.py RUNDIR [--posthoc]   (posthoc = roll out saved actors when feats npz missing)"""
import sys, os, json, glob, math
import numpy as np, torch
from scipy.special import digamma, gammaln
from common import FEAT_DIM, make_env, load_expert, Actor

RUN = sys.argv[1]; POST = "--posthoc" in sys.argv
RANDOM = {}


def random_return(env_id, n=10):
    if env_id not in RANDOM:
        e = make_env(env_id, False); R = []
        for i in range(n):
            e.reset(seed=i); d = False; s = 0.0
            while not d:
                _, r, te, tr, _ = e.step(e.action_space.sample()); s += r; d = te or tr
            R.append(s)
        RANDOM[env_id] = float(np.mean(R))
    return RANDOM[env_id]


def knn_entropy(X, k=5):
    """Kozachenko-Leonenko entropy estimate (nats)."""
    X = torch.tensor(X, dtype=torch.float64)
    n, d = X.shape
    D = torch.cdist(X, X); D.fill_diagonal_(float("inf"))
    eps = D.kthvalue(k, 1).values.numpy()
    logV = (d / 2) * math.log(math.pi) - gammaln(d / 2 + 1)
    return float(digamma(n) - digamma(k) + logV + d * np.mean(np.log(2 * eps + 1e-12)) - d * math.log(2))


def rollout(env_id, actor_path, n=10):
    env = make_env(env_id, False); k = FEAT_DIM[env_id]
    a = Actor(env.observation_space.shape[0], env.action_space.shape[0]); a.load_state_dict(torch.load(actor_path, map_location="cpu"))
    F = []
    for i in range(n):
        o, _ = env.reset(seed=10_000 + i); d = False; prev = o[:k]
        while not d:
            with torch.no_grad():
                act = a(torch.tensor(o, dtype=torch.float32)[None], det=True)[0][0].numpy()
            o, _, te, tr, _ = env.step(act); d = te or tr
            F.append(np.concatenate([prev, o[:k]])); prev = o[:k]
    return np.array(F, np.float32)


rows = []
for jf in sorted(glob.glob(os.path.join(RUN, "*.json"))):
    L = json.load(open(jf)); a = L["args"]
    if "final" not in L:
        continue
    env_id, k = a["env"], FEAT_DIM[a["env"]]
    rr = random_return(env_id)
    nret = (L["final"]["ret"] - rr) / (L["expert_return"] - rr)
    fe = jf.replace(".json", "_feats.npz")
    F = np.load(fe)["F"] if os.path.exists(fe) else (rollout(env_id, jf.replace(".json", "_actor.pt")) if POST else None)
    dH = float("nan")
    if F is not None and len(F) > 50:
        _, XH, mu, sd, _ = load_expert(env_id, a["ntraj"], seed=0)
        rng = np.random.default_rng(0)
        n = min(len(F), 3000)
        P = (F[rng.choice(len(F), n, replace=False), :k] - mu) / sd
        Q = (XH[rng.choice(len(XH), n, replace=False), :k] - mu) / sd
        dH = knn_entropy(P) - knn_entropy(Q)
    rows.append((env_id, a["method"], a["seed"], nret, L["final"]["mmd"], L["final"]["sw2"], dH))

print(f"{'env':15s} {'method':16s} {'seed':>4s} {'normRet':>8s} {'MMD':>7s} {'SW2':>6s} {'H-H_exp':>8s}")
for r in rows:
    print(f"{r[0]:15s} {r[1]:16s} {r[2]:4d} {r[3]:8.3f} {r[4]:7.4f} {r[5]:6.3f} {r[6]:8.2f}")
json.dump([dict(env=r[0], method=r[1], seed=r[2], norm_ret=r[3], mmd=r[4], sw2=r[5], dH=r[6]) for r in rows],
          open(os.path.join(RUN, "summary.json"), "w"), indent=1)
