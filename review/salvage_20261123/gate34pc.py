"""#34 positive control (PREREG_34pc.md): two HalfCheetah gaits differing only in front/back footfall ORDER.
(a) do style features separate them (AUC); (b) can an oracle footfall-order reward reach each gait under the
same CEM budget (1x and 4x)?  Run with fm34 venv python.  argv[1]: "smoke" for a tiny run.
"""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gate34_20261123"))
import gate34 as G  # noqa: E402
G.NTHREAD = 6
from coordination34 import features  # noqa: E402  (gate34 put verification_20261123 on the path)

V, P = 1.0, G.P
BASES = (2, 4, 5)
FRONT = (3, 4, 5)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"


def circ_mean(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else float((np.angle(np.mean(np.exp(2j * np.pi * x))) / (2 * np.pi)) % 1)


def dcirc(a, b):
    if np.isnan(a) or np.isnan(b):
        return .5
    d = abs(a - b) % 1
    return min(d, 1 - d)


class Feet:
    def __init__(self, sim):
        self.m, self.d, self.sim = sim.m, mujoco.MjData(sim.m), sim

    def contacts(self, q):
        """q (F, nq) frame qpos -> (cb, cf) bool (F,) floor contact at frame resolution (as gate34, via fwdPosition)."""
        cb, cf = np.zeros(len(q), bool), np.zeros(len(q), bool)
        for t in range(len(q)):
            self.d.qpos[:] = q[t]
            mujoco.mj_fwdPosition(self.m, self.d)
            for k in range(self.d.ncon):
                g = (self.d.contact[k].geom1, self.d.contact[k].geom2)
                if self.sim.floor in g:
                    cb[t] |= self.sim.bfoot in g
                    cf[t] |= self.sim.ffoot in g
        return cb, cf

    def lag(self, q):
        cb, cf = self.contacts(q)
        tb = np.where(cb[1:] & ~cb[:-1])[0] + 1
        tf = np.where(cf[1:] & ~cf[:-1])[0] + 1
        ls = []
        for b in tb:
            nxt = tf[(tf > b) & (tf <= b + int(1.5 * P))]
            if len(nxt):
                ls.append(((nxt[0] - b) / P) % 1)
        return circ_mean(ls), len(ls)


def frames(sim, st):
    return st[:, sim.skip - 1::sim.skip, 1:1 + sim.m.nq]


def variant(pa, sign):
    p = pa.reshape(6, 5).copy(); p[G.FTHIGH, 2] = p[G.BTHIGH, 2]
    for j in FRONT:
        p[j, 2] += sign * np.pi / 2; p[j, 4] += np.pi
    return p


def fit_constants(sim, p, seed, gens=8, pop=64):
    """CEM over the 6 constant terms only (amplitudes/phases fixed), task reward."""
    rng = np.random.default_rng(seed); c0 = p[:, 0].copy()
    center, scale = c0.copy(), np.full(6, .3)
    def obj(C, s):
        Q = np.repeat(p[None], len(C), 0); Q[:, :, 0] = C
        start, st = sim.run(Q.reshape(len(C), 30), s, G.H_SEARCH, None)
        return sim.task(start, st, V, G.H_SEARCH)[0]
    cands = [c0]
    for g in range(gens):
        C = rng.normal(center, scale, (pop, 6))
        avg = np.mean([obj(C, seed + 2 * g + j) for j in range(2)], 0)
        el = C[np.argsort(avg)[-10:]]; center, scale = el.mean(0), np.maximum(el.std(0), .02)
        cands += [center.copy(), el[-1].copy()]
    cands = np.array(cands)
    val = np.mean([obj(cands, seed + 100 + j) for j in range(4)], 0)
    q = p.copy(); q[:, 0] = cands[np.argmax(val)]
    return q


def demos(sim, feet, p, resets):
    out = []
    for e in G.episodes(sim, p.ravel(), None, V, resets):
        start, st = sim.run(p.ravel()[None], e["reset"], G.H_EVAL, None)
        q = frames(sim, st)[0]
        l, n = feet.lag(q[-256:])
        out.append(dict(task_ok=e["task_ok"], v=e["v"], lag=l, nlag=n, J=q[-128:, 3:9]))
    return out


def auc(far, near):
    far, near = np.asarray(far), np.asarray(near)
    return float(((far[:, None] > near[None]).mean() + .5 * (far[:, None] == near[None]).mean()))


def main():
    t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    sim = G.Sim(); feet = Feet(sim)
    z = np.load(HERE.parent / "gate34_20261123" / "gaits_v1.0.npz")
    nd = 3 if SMOKE else 8
    bases = BASES[:1] if SMOKE else BASES
    res = dict(bases={}, cem=[])
    try:
        info = {}
        for i in bases:
            pa = z[f"A_inphase_s{i}"]
            gv = {}
            for name, sign, rs in (("F", +1, 341000), ("R", -1, 342000)):
                p = fit_constants(sim, variant(pa, sign), 343000 + 10 * i + (sign > 0))
                ds = [d for d in demos(sim, feet, p, [rs + 100 * i + j for j in range(nd)]) if d["task_ok"]]
                gv[name] = dict(p=p, demos=ds)
                log("base", i, name, "pass", len(ds), "/", nd, "v", [round(d["v"], 2) for d in ds], "lag", [round(d["lag"], 3) for d in ds])
            F, R = gv["F"]["demos"], gv["R"]["demos"]
            if len(F) < 3 or len(R) < 3:
                info[i] = dict(usable=False, nF=len(F), nR=len(R)); log("base", i, "unusable"); continue
            sdj = np.maximum(np.concatenate([d["J"] for d in F[:2] + R[:2]]).std(0), 1e-3)
            feat = lambda ds: features(np.array([d["J"] for d in ds]) / sdj)
            refs = {g: {k: v.mean(0) for k, v in feat(gv[g]["demos"][:2]).items()} for g in ("F", "R")}
            def dist(ds, g):
                f = feat(ds)
                return {k: np.linalg.norm((v - refs[g][k]).reshape(len(ds), -1), axis=1) / np.linalg.norm(refs[g][k]) for k, v in f.items()}
            dF, dR = dist(F[2:], "F"), dist(R[2:], "F")
            lF, lR = circ_mean([d["lag"] for d in F]), circ_mean([d["lag"] for d in R])
            psdF, psdR = feat(F)["psd"].mean(0), feat(R)["psd"].mean(0)
            info[i] = dict(usable=True, nF=len(F), nR=len(R), lag_F=lF, lag_R=lR,
                           lag_F_eps=[d["lag"] for d in F], lag_R_eps=[d["lag"] for d in R],
                           v_F=[d["v"] for d in F], v_R=[d["v"] for d in R],
                           psd_rel_diff=float(np.linalg.norm(psdF - psdR) / np.linalg.norm(psdF)),
                           auc={k: auc(dR[k], dF[k]) for k in dF},
                           auc_order=auc([dcirc(d["lag"], lF) for d in R[2:]], [dcirc(d["lag"], lF) for d in F[2:]]),
                           _dF={k: v.tolist() for k, v in dF.items()}, _dR={k: v.tolist() for k, v in dR.items()},
                           params={g: gv[g]["p"].tolist() for g in gv})
            info[i]["_refs"], info[i]["_sdj"] = refs, sdj
            log("base", i, "lagF", round(lF, 3), "lagR", round(lR, 3), "psd_rel_diff", round(info[i]["psd_rel_diff"], 3),
                "auc", {k: round(v, 3) for k, v in info[i]["auc"].items()}, "auc_order", round(info[i]["auc_order"], 3))
        # (b) CEM arms
        budgets = (16,) if SMOKE else (64, 256)
        seeds = (0,) if SMOKE else (0, 1, 2)
        for i in bases:
            if not info[i]["usable"]:
                continue
            refs, sdj = info[i]["_refs"], info[i]["_sdj"]
            tgt = {"F": info[i]["lag_F"], "R": info[i]["lag_R"]}
            def score(p, s, arm):
                start, st = sim.run(p, s, G.H_SEARCH, None)
                t = sim.task(start, st, V, G.H_SEARCH)[0]
                if arm == "task":
                    return t
                kind, g = arm.split("_")
                q = frames(sim, st)
                if kind == "oracle":
                    return t - 2 * np.array([dcirc(feet.lag(x[-128:])[0], tgt[g]) for x in q])
                f = features(q[:, -128:, 3:9] / sdj)["cross_spectrum"]; r = refs[g]["cross_spectrum"]
                return t - np.linalg.norm((f - r).reshape(len(q), -1), axis=1) / np.linalg.norm(r)
            for pop in budgets:
                for sd in seeds:
                    for arm in ("task", "oracle_F", "oracle_R", "cross_F", "cross_R"):
                        rng = np.random.default_rng(10000 * i + 100 * sd + pop)
                        center = np.zeros(30); center[1::5] = .6
                        scale = np.full(30, .55); scale[2::5] = np.pi; scale[4::5] = np.pi
                        base = 350000 + 10000 * i + 500 * sd
                        cands = []
                        for g in range(2 if SMOKE else 8):
                            pp = rng.normal(center, scale, (pop, 30))
                            avg = np.mean([score(pp, base + 2 * g + j, arm) for j in range(2)], 0)
                            ne = max(10, pop // 6 + 0) if pop > 64 else 10
                            el = pp[np.argsort(avg)[-ne:]]
                            center, scale = el.mean(0), np.maximum(el.std(0), .05)
                            cands += [center.copy(), el[-1].copy()]
                        cands = np.array(cands)
                        val = np.mean([score(cands, base + 100 + j, arm) for j in range(4)], 0)
                        best = cands[np.argmax(val)]
                        ev = [dict(task_ok=e["task_ok"], v=e["v"]) for e in G.episodes(sim, best, None, V, [360000 + 10000 * i + 100 * sd + j for j in range(4)])]
                        lags = []
                        for j in range(4):
                            start, st = sim.run(best[None], 360000 + 10000 * i + 100 * sd + j, G.H_EVAL, None)
                            lags.append(feet.lag(frames(sim, st)[0][-256:])[0])
                        l = circ_mean(lags); ok4 = all(e["task_ok"] for e in ev)
                        row = dict(base=i, seed=sd, pop=pop, arm=arm, task_ok=sum(e["task_ok"] for e in ev), lag=l, lags=lags,
                                   reach_F=bool(ok4 and dcirc(l, tgt["F"]) < .125), reach_R=bool(ok4 and dcirc(l, tgt["R"]) < .125),
                                   v=[e["v"] for e in ev])
                        res["cem"].append(row)
                        log(i, sd, pop, arm, "task", row["task_ok"], "lag", None if np.isnan(l) else round(l, 3), "reachF", row["reach_F"], "reachR", row["reach_R"])
        for i in info:
            info[i].pop("_refs", None); info[i].pop("_sdj", None)
        res["bases"] = {str(k): v for k, v in info.items()}
        summ = {}
        for pop in budgets:
            for arm, tgt_ in (("task", "F"), ("task", "R"), ("oracle_F", "F"), ("oracle_R", "R"), ("cross_F", "F"), ("cross_R", "R"), ("oracle_F", "R"), ("oracle_R", "F")):
                rs = [r for r in res["cem"] if r["pop"] == pop and r["arm"] == arm]
                summ[f"{arm}->{tgt_}@{pop}"] = f"{sum(r['reach_' + tgt_] for r in rs)}/{len(rs)}"
        res["summary"] = summ
        usable = [k for k, v in info.items() if v["usable"]]
        allF = np.concatenate([info[k]["_dF"]["cross_spectrum"] for k in usable]) if usable else []
        allR = np.concatenate([info[k]["_dR"]["cross_spectrum"] for k in usable]) if usable else []
        res["auc_cross_pooled_within_base"] = float(np.mean([info[k]["auc"]["cross_spectrum"] for k in usable])) if usable else None
        n9 = len(seeds) * len(usable)
        okF = max(sum(r["reach_F"] for r in res["cem"] if r["arm"] == "oracle_F" and r["pop"] == pop) for pop in budgets) if usable else 0
        okR = max(sum(r["reach_R"] for r in res["cem"] if r["arm"] == "oracle_R" and r["pop"] == pop) for pop in budgets) if usable else 0
        res["verdict"] = dict(a_pass=bool(usable) and all(info[k]["auc"]["cross_spectrum"] >= .9 for k in usable),
                              b_pass=bool(usable) and okF >= n9 - 1 and okR >= n9 - 1, okF=okF, okR=okR, n=n9, usable_bases=usable)
        log("SUMMARY", json.dumps(summ)); log("VERDICT", json.dumps(res["verdict"]))
        (HERE / ("gate34pc_smoke.json" if SMOKE else "gate34pc.json")).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
