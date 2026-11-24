"""#36 step 4a: retrieval re-tuning + retrieval-proposal PF with exact mixture importance weights. See PREREG.md (4a)."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402

import pf36_g3 as g  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPS = (.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)
ESS_T = (8, 32, 128, 512)
K, ALPHA = 64, .1
ZD = [1, 2, 4, 5, 6]
LOGV = float(np.sum(np.log(g.HI-g.LO))) + np.log(144)


def kern():
    k = np.zeros((12, 12))
    for i in range(12):
        k[i, i] += .7
        k[i, max(i-1, 0)] += .15
        k[i, min(i+1, 11)] += .15
    return k


KM = kern()


def score_from_lw(lib, lw, case, xthr, seed, topk=4096):
    pos, bnc = lib[0], lib[1]
    idx = np.argpartition(-lw, topk, 1)[:, :topk]
    kept = np.take_along_axis(softmax(lw, 1), idx, 1).sum(1)
    sub = np.take_along_axis(lw, idx, 1)
    b1, b2 = g.events(pos[idx], bnc[idx], xthr)
    res = g.score_weighted(pos[idx][:, :, g.CUT:], b1, b2, sub, case, xthr, np.random.default_rng(seed))
    res["kept_mass"] = kept
    return res


def adaptive_lw(lw1, ess_target):
    out = np.empty_like(lw1)
    chosen = np.empty(len(lw1))
    for c in range(len(lw1)):
        for t in TEMPS:
            w = softmax(lw1[c]/t)
            if 1/np.square(w).sum() >= ess_target or t == TEMPS[-1]:
                out[c], chosen[c] = lw1[c]/t, t
                break
    return out, chosen


def retrieval_variants(lib, case, xthr, seed):
    lw1 = g.lib_logw(lib[0], case, 1.)
    res = {}
    for t in TEMPS:
        res[("fixed", t)] = score_from_lw(lib, lw1/t, case, xthr, seed)
    for e in ESS_T:
        lw, ch = adaptive_lw(lw1, e)
        r = score_from_lw(lib, lw, case, xthr, seed)
        r["temp_median"] = np.full(len(lw), np.median(ch))
        res[("ess", e)] = r
    return res


def proposal_init(lib, L, case, temp_ret, N):
    """Mixture-of-Gaussians proposal around top-K library matches + defensive prior. Exact log(prior/q)."""
    pos, _, st0, im0, id0 = lib
    C = len(case["obs"])
    lw = g.lib_logw(pos[:L], case, temp_ret)
    top = np.argpartition(-lw, K, 1)[:, :K]
    pi = softmax(np.take_along_axis(lw, top, 1), 1)                     # (C,K)
    zj = st0[:L][top][..., ZD]                                           # (C,K,5)
    mj, dj = im0[:L][top], id0[:L][top]
    mean = np.einsum("ck,ckd->cd", pi, zj)
    dev = zj-mean[:, None]
    cov = np.einsum("ck,ckd,cke->cde", pi, dev, dev) + np.diag(((g.HI-g.LO)*.005)**2)
    Lc = np.linalg.cholesky(cov)
    Li = np.linalg.inv(Lc)
    logdet = 2*np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)          # (C,)

    def draw(rng):
        comp = np.stack([rng.choice(K, N, p=p) for p in pi])             # (C,N)
        z = np.take_along_axis(zj, comp[..., None], 1) + np.einsum("cde,cne->cnd", Lc, rng.standard_normal((C, N, 5)))
        imu = np.array([[rng.choice(12, p=KM[a]) for a in row] for row in np.take_along_axis(mj, comp, 1)])
        idd = np.array([[rng.choice(12, p=KM[a]) for a in row] for row in np.take_along_axis(dj, comp, 1)])
        pri = rng.uniform(size=(C, N)) < ALPHA
        zp = g.LO + (g.HI-g.LO)*rng.uniform(size=(C, N, 5))
        z = np.where(pri[..., None], zp, z)
        imu = np.where(pri, rng.integers(0, 12, (C, N)), imu)
        idd = np.where(pri, rng.integers(0, 12, (C, N)), idd)
        # exact mixture density
        diff = z[:, :, None, :]-zj[:, None, :, :]                        # (C,N,K,5)
        u = np.einsum("cde,cnke->cnkd", Li, diff)
        lg = -.5*np.square(u).sum(-1) - .5*logdet[:, None, None] - 2.5*np.log(2*np.pi)
        with np.errstate(divide='ignore'):
            ld = np.log(KM[mj[:, None, :], imu[:, :, None]]) + np.log(KM[dj[:, None, :], idd[:, :, None]])
        lmix = logsumexp(lg + ld + np.log(pi)[:, None, :], axis=2)
        inside = np.all((z >= g.LO) & (z <= g.HI), -1)
        lprior = np.where(inside, -LOGV, -np.inf)
        lq = np.logaddexp(np.log(1-ALPHA)+lmix, np.log(ALPHA)+np.where(inside, -LOGV, -np.inf))
        logw = lprior - lq
        st = np.zeros((C*N, 7))
        st[:, ZD] = z.reshape(-1, 5)
        return st, imu.ravel(), idd.ravel(), logw
    return draw


def summarize(r):
    return {k: float(np.mean(v)) for k, v in r.items()}


def main():
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    val = g.make_cases(51000, 48, test=False)
    tests = [g.make_cases(s, 48, test=True) for s in range(51200, 51204)]
    libs = {"train_196k": g.library(g.TRPOOL, 8, 8, 196_608, 52001), "broad_196k": g.library(g.POOL, 12, 12, 196_608, 52003)}
    xthr = float(np.median(libs["train_196k"][0][:, -1, 0]-libs["train_196k"][0][:, g.CUT-1, 0]))
    log("libs ready xthr", xthr)
    # (i) retrieval re-tuning on validation only
    valres = {n: retrieval_variants(l, val, xthr, 1) for n, l in libs.items()}
    val_energy = {f"{n}|{v[0]}|{v[1]}": float(np.mean(r["energy"])) for n, rr in valres.items() for v, r in rr.items()}
    best_fixed = {n: min((v for v in rr if v[0] == "fixed"), key=lambda v: np.mean(rr[v]["energy"])) for n, rr in valres.items()}
    best_ess = {n: min((v for v in rr if v[0] == "ess"), key=lambda v: np.mean(rr[v]["energy"])) for n, rr in valres.items()}
    cands = [(n, best_fixed[n]) for n in libs] + [(n, best_ess[n]) for n in libs]
    best = min(cands, key=lambda c: np.mean(valres[c[0]][c[1]]["energy"]))
    log("val best fixed", best_fixed, "best ess", best_ess, "overall", best)
    table = {}
    for n, v in cands:
        per = [summarize(retrieval_variants(libs[n], c, xthr, 7+i)[v]) for i, c in enumerate(tests)]
        key = f"retrieval_{n}_{v[0]}{v[1]}"
        table[key] = dict({k: float(np.mean([p[k] for p in per])) for k in per[0]}, energy_per_seed=[p["energy"] for p in per],
                          calls_total=196_608, calls_per_case=0)
        log(key, {k: round(x, 4) for k, x in table[key].items() if not isinstance(x, list)})
    best_key = f"retrieval_{best[0]}_{best[1][0]}{best[1][1]}"
    # (ii) retrieval-proposal PF on the broad library
    temp_ret = best_fixed["broad_196k"][1]
    lib = libs["broad_196k"]
    for L, N in ((98_304, 512), (147_456, 256), (184_320, 64), (196_608, 1024)):
        vt = {}
        for tp in (1, 2, 4):
            vt[tp] = float(np.mean(g.run_pf_full(val, N, tp, xthr, 3, init=proposal_init(lib, L, val, temp_ret, N), resample_after=g.CUT-1)["energy"]))
        tp = min(vt, key=vt.get)
        per = []
        for i, c in enumerate(tests):
            r = g.run_pf_full(c, N, tp, xthr, 300+N+i, init=proposal_init(lib, L, c, temp_ret, N), resample_after=g.CUT-1)
            per.append(summarize(r))
        key = f"rpf_L{L}_N{N}"
        table[key] = dict({k: float(np.mean([p[k] for p in per])) for k in per[0]}, energy_per_seed=[p["energy"] for p in per],
                          calls_total=L+N*192, calls_per_case=N, library=L, pf_temp=tp, val=vt,
                          within_budget=bool(L+N*192 <= 196_608))
        log(key, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in table[key].items() if not isinstance(x, (list, dict))})
    g3 = json.loads((HERE/"gate3.json").read_text(encoding="utf-8"))["table"]
    ref = g3["pf_N8192"]["energy"]
    ret = table[best_key]["energy"]
    ret_seeds = np.array(table[best_key]["energy_per_seed"])
    verdict = {}
    for k, v in table.items():
        if not k.startswith("rpf_"):
            continue
        e = v["energy"]
        verdict[k] = dict(vs_ref=e/ref-1, vs_retrieval=e/ret-1, all_seeds_beat_retrieval=bool(np.all(np.array(v["energy_per_seed"]) < ret_seeds)),
                          within_budget=v["within_budget"],
                          passed=bool(v["within_budget"] and e <= 1.05*ref and e <= .95*ret and np.all(np.array(v["energy_per_seed"]) < ret_seeds)))
    log("verdict", verdict)
    out = dict(best_retrieval=best_key, temp_ret_for_proposal=temp_ret, val_retrieval=val_energy, table=table,
               reference=dict(pf_N8192=g3["pf_N8192"], pf_N1024=g3["pf_N1024"]), verdict=verdict,
               passed=any(v["passed"] for v in verdict.values()), seconds=time.time()-t0)
    (HERE/"gate4a.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
