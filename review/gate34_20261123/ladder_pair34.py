"""Stage 1: pair-level demo-label swap inside CEM (see PREREG.md, 단계 1)."""
import json
from pathlib import Path
import numpy as np
import gate34 as G
from coordination34 import features

HERE = Path(__file__).resolve().parent
V = 1.0
PAIRS = (2, 4, 5)
KINDS = ("cross_spectrum", "zero_lag", "psd")
WS = (0.3, 1.0)


def main():
    z = np.load(HERE/"gaits_v1.0.npz")
    pb_all = json.loads((HERE/"gate0b_v1.0.json").read_text())["paramsB"]
    sim = G.Sim(); nq = sim.m.nq; rows = []; pairinfo = {}
    try:
        for i in PAIRS:
            pa = z[f"A_inphase_s{i}"].reshape(6, 5).copy(); pa[G.FTHIGH, 2] = pa[G.BTHIGH, 2]
            EA = [e for e in G.episodes(sim, pa.ravel(), None, V, [170000+10*i+j for j in range(8)]) if e["task_ok"]]
            EB = [e for e in G.episodes(sim, np.array(pb_all[i]), None, V, [175000+10*i+j for j in range(8)]) if e["task_ok"]]
            CA = [G.cycles(e, sim.skip)[0] for e in EA]; CB = [G.cycles(e, sim.skip)[0] for e in EB]
            allc = np.concatenate(CA[:2]+CB[:2]); mu, sd = allc.mean(0), allc.std(0)+1e-6
            zf = lambda x, mu=mu, sd=sd: (x-mu)/sd
            dd = np.sqrt(((zf(allc)[:, None]-zf(allc)[None])**2).sum(-1)); h = float(np.median(dd[np.triu_indices(len(allc), 1)]))
            TA, TB = np.concatenate([zf(c) for c in CA[2:]]), np.concatenate([zf(c) for c in CB[2:]])
            gap = G.mmd2(TA, TB, h)
            fr = {"A": np.array([e["frames_joints"][-128:] for e in EA[2:]]), "B": np.array([e["frames_joints"][-128:] for e in EB[2:]])}
            sdj = np.maximum(np.concatenate([fr["A"].reshape(-1, 6), fr["B"].reshape(-1, 6)]).std(0), 1e-3)
            refs = {g: {k: v.mean(0) for k, v in features(fr[g]/sdj).items()} for g in fr}
            pairinfo[i] = dict(nA=len(EA), nB=len(EB), gap=gap, bandwidth=h)
            print("pair", i, len(EA), len(EB), "gap", round(gap, 4), flush=True)

            def sdist(st, g, kind):
                q = st[:, sim.skip-1::sim.skip, 1:1+nq][:, -128:, 3:9]
                f = features(q/sdj)[kind]; r = refs[g][kind]
                return np.linalg.norm((f-r).reshape(len(q), -1), axis=1)/np.linalg.norm(r)

            def cem(seed, lab, kind, w):
                rng = np.random.default_rng(1000*i+seed)
                center = np.zeros(30); center[1::5] = .6
                scale = np.full(30, .55); scale[2::5] = np.pi; scale[4::5] = np.pi
                base = 200000+10000*i+500*seed
                def obj(p, s):
                    start, st = sim.run(p, s, G.H_SEARCH, None)
                    t = sim.task(start, st, V, G.H_SEARCH)[0]
                    return t if lab is None else t-w*sdist(st, lab, kind)
                cands = []
                for g in range(8):
                    p = rng.normal(center, scale, (64, 30))
                    avg = np.mean([obj(p, base+2*g+j) for j in range(2)], axis=0)
                    el = p[np.argsort(avg)[-10:]]
                    center, scale = el.mean(0), np.maximum(el.std(0), .05)
                    cands += [center.copy(), el[-1].copy()]
                cands = np.array(cands)
                val = np.mean([obj(cands, base+100+j) for j in range(4)], axis=0)
                return cands[np.argmax(val)]

            arms = [(None, None, 0.0)]+[(lab, k, w) for k in KINDS for w in WS for lab in ("A", "B")]
            for seed in range(3):
                for lab, kind, w in arms:
                    p = cem(seed, lab, kind, w)
                    eps = G.episodes(sim, p, None, V, [260000+1000*i+10*seed+j for j in range(4)])
                    cs = [G.cycles(e, sim.skip)[0] for e in eps]
                    cs = np.concatenate(cs) if cs else np.zeros((0, 194))
                    s = float(G.mmd2(zf(cs), TA, h)-G.mmd2(zf(cs), TB, h)) if len(cs) >= 2 else None
                    rows.append(dict(pair=i, seed=seed, label=lab, kind=kind, w=w, task_ok=sum(e["task_ok"] for e in eps),
                                     v=[e["v"] for e in eps], signed=s, ncycles=len(cs)))
                    print(i, seed, lab, kind, w, rows[-1]["task_ok"], None if s is None else round(s, 4), flush=True)
        verdict = {}
        for kind in KINDS:
            for w in WS:
                per = []
                for i in PAIRS:
                    for seed in range(3):
                        ra = next(r for r in rows if (r["pair"], r["seed"], r["label"], r["kind"], r["w"]) == (i, seed, "A", kind, w))
                        rb = next(r for r in rows if (r["pair"], r["seed"], r["label"], r["kind"], r["w"]) == (i, seed, "B", kind, w))
                        mv = None if ra["signed"] is None or rb["signed"] is None else rb["signed"]-ra["signed"]
                        ok = mv is not None and mv >= pairinfo[i]["gap"]/2 and ra["task_ok"] == 4 and rb["task_ok"] == 4
                        per.append(dict(pair=i, seed=seed, move=mv, half_gap=pairinfo[i]["gap"]/2, task_both=ra["task_ok"] == 4 and rb["task_ok"] == 4, ok=ok))
                verdict[f"{kind}_w{w}"] = dict(n_ok=sum(x["ok"] for x in per), n=len(per), pass_=all(x["ok"] for x in per),
                                               n_move_positive=sum((x["move"] or 0) > 0 for x in per), per=per)
        out = dict(pairs=pairinfo, rows=rows, verdict=verdict)
        (HERE/"ladder_pair_v1.0.json").write_text(json.dumps(out, indent=1, allow_nan=False)+"\n", encoding="utf-8")
        for k, v in verdict.items():
            print("VERDICT", k, v["n_ok"], "/", v["n"], "pass" if v["pass_"] else "fail", "move>0:", v["n_move_positive"])
    finally:
        sim.close()


if __name__ == "__main__":
    main()
