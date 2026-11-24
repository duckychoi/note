"""#34 v3 rung 1 (only after gate 0 passes): demo-label swap inside CEM. Unconstrained 30-coef class,
reward = task - w * d_style(demo), d_style = Fourier relative distance (cross_spectrum or psd) to demo A or B.
Judged by held-out contact-aligned cycle MMD (not the reward). See PREREG.md."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import gate34 as G
from coordination34 import features

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vstar", type=float, default=1.5)
    a = ap.parse_args()
    vstar = a.vstar
    z = np.load(HERE/f"gaits_v{vstar}.npz")
    sim = G.Sim()
    try:
        names = ("A_inphase", "B_antiphase")
        deltas = {"A_inphase": 0.0, "B_antiphase": np.pi}
        # regenerate demo episodes (deterministic) from saved gait params
        demo = {}
        for g in names:
            demo[g] = []
            for i in range(6):
                p = z[f"{g}_s{i}"]
                eps = G.episodes(sim, p, deltas[g], vstar,
                                 [130000+10*i+j+(0 if g == "A_inphase" else 5000) for j in range(4)])
                demo[g].append([e for e in eps if e["task_ok"]])
        sub = sim.skip
        cyc = {g: [np.concatenate([G.cycles(e, sub)[0] for e in s]) for s in demo[g] if s] for g in names}
        train = np.concatenate([cyc[g][0] for g in names])
        mu, sd = train.mean(0), train.std(0)+1e-6
        zf = lambda x: (x-mu)/sd
        dtr = np.sqrt(((zf(train)[:, None]-zf(train)[None])**2).sum(-1))
        h = float(np.median(dtr[np.triu_indices(len(train), 1)]))
        test = {g: np.concatenate([zf(c) for c in cyc[g][1:]]) for g in names}
        gap = G.mmd2(test[names[0]], test[names[1]], h)
        # Fourier style references from demo frames (all demo seeds; style reward side)
        fr = {g: np.array([e["frames_joints"][-128:] for s in demo[g] for e in s]) for g in names}
        sdj = np.maximum(np.concatenate([fr[g].reshape(-1, 6) for g in names]).std(0), 1e-3)
        refs = {g: {k: v.mean(0) for k, v in features(fr[g]/sdj).items()} for g in names}
        nq = sim.m.nq

        def style_dist(states, g, kind):
            q = states[:, sim.skip-1::sim.skip, 1:1+nq][:, -128:, 3:9]
            f = features(q/sdj)[kind]
            r = refs[g][kind]
            return np.linalg.norm((f-r).reshape(len(q), -1), axis=1)/np.linalg.norm(r)

        def cem_style(seed, label, kind, w):
            rng = np.random.default_rng(seed)
            center = np.zeros(30); center[1::5] = .6
            scale = np.full(30, .55); scale[2::5] = np.pi; scale[4::5] = np.pi
            cands = []
            base = 150000+97*seed
            def obj(params, s):
                start, st = sim.run(params, s, G.H_SEARCH, None)
                t = sim.task(start, st, vstar, G.H_SEARCH)[0]
                return t if label is None else t - w*style_dist(st, label, kind)
            for gen in range(8):
                params = rng.normal(center, scale, (64, 30))
                avg = np.mean([obj(params, base+2*gen+j) for j in range(2)], axis=0)
                elite = params[np.argsort(avg)[-10:]]
                center, scale = elite.mean(0), np.maximum(elite.std(0), .05)
                cands += [center.copy(), elite[-1].copy()]
            cands = np.array(cands)
            val = np.mean([obj(cands, base+100+j) for j in range(4)], axis=0)
            return cands[np.argmax(val)]

        rows = []
        arms = [(None, None, 0.0)] + [(lab, k, w) for k in ("cross_spectrum", "psd") for w in (0.3, 1.0) for lab in names]
        for seed in range(5):
            for lab, kind, w in arms:
                p = cem_style(seed, lab, kind, w)
                eps = G.episodes(sim, p, None, vstar, [160000+10*seed+j for j in range(4)])
                ok = [e for e in eps if e["task_ok"]]
                cs = np.concatenate([zf(G.cycles(e, sub)[0]) for e in eps])
                s = (G.mmd2(cs, test[names[0]], h) - G.mmd2(cs, test[names[1]], h)) if len(cs) >= 2 else None
                rows.append(dict(seed=seed, label=lab, kind=kind, w=w, task_ok=len(ok), n=len(eps),
                                 v=[e["v"] for e in eps], upright=[e["upright"] for e in eps],
                                 signed=s, ncycles=len(cs)))
                print(seed, lab, kind, w, len(ok), None if s is None else round(s, 4), flush=True)
        # verdict per (kind,w): swap moves signed distance by >= gap/2 in every seed with task pass
        verdict = {}
        for kind in ("cross_spectrum", "psd"):
            for w in (0.3, 1.0):
                per = []
                for seed in range(5):
                    ra = next(r for r in rows if r["seed"] == seed and r["label"] == names[0] and r["kind"] == kind and r["w"] == w)
                    rb = next(r for r in rows if r["seed"] == seed and r["label"] == names[1] and r["kind"] == kind and r["w"] == w)
                    move = None if ra["signed"] is None or rb["signed"] is None else rb["signed"]-ra["signed"]
                    per.append(dict(seed=seed, move=move, task_ok=(ra["task_ok"] == 4 and rb["task_ok"] == 4)))
                verdict[f"{kind}_w{w}"] = dict(per_seed=per, gap=gap, pass_=all(
                    x["move"] is not None and x["move"] >= gap/2 and x["task_ok"] for x in per))
        out = dict(vstar=vstar, gap_AB=gap, bandwidth=h, rows=rows, verdict=verdict)
        (HERE/f"ladder_v{vstar}.json").write_text(json.dumps(out, indent=1, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(verdict, indent=1))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
