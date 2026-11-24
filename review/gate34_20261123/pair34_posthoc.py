"""POST-HOC exploratory (not a gate): per-demonstrator pair A_i vs B_i (gate0b params) with episode-split floor
within the same controller, using 8 eval episodes per controller. Contacts shifted together in the common-shift control."""
import json, itertools
from pathlib import Path
import numpy as np
import gate34 as G
HERE = Path(__file__).resolve().parent
d = json.loads((HERE/"gate0b_v1.0.json").read_text())
z = np.load(HERE/"gaits_v1.0.npz")
sim = G.Sim(); out = []
try:
    for i in range(6):
        pa = z[f"A_inphase_s{i}"].reshape(6, 5).copy(); pa[G.FTHIGH, 2] = pa[G.BTHIGH, 2]
        pb = np.array(d["paramsB"][i])
        EA = [e for e in G.episodes(sim, pa.ravel(), None, 1.0, [170000+10*i+j for j in range(8)]) if e["task_ok"]]
        EB = [e for e in G.episodes(sim, pb, None, 1.0, [175000+10*i+j for j in range(8)]) if e["task_ok"]]
        if len(EA) < 6 or len(EB) < 6:
            out.append(dict(seed=i, skipped=True, nA=len(EA), nB=len(EB))); print(i, "skip", len(EA), len(EB)); continue
        CA = [G.cycles(e, sim.skip)[0] for e in EA]; CB = [G.cycles(e, sim.skip)[0] for e in EB]
        allc = np.concatenate(CA[:2]+CB[:2]); mu, sd = allc.mean(0), allc.std(0)+1e-6   # first 2 eps each: scaling only
        zf = lambda x: (x-mu)/sd
        dd = np.sqrt(((zf(allc)[:, None]-zf(allc)[None])**2).sum(-1)); h = float(np.median(dd[np.triu_indices(len(allc), 1)]))
        A, B = [zf(c) for c in CA[2:]], [zf(c) for c in CB[2:]]
        k = min(len(A), len(B))//2
        floor = [G.mmd2(np.concatenate([X[j] for j in s]), np.concatenate([X[j] for j in range(len(X)) if j not in s][:k] and [X[j] for j in range(len(X)) if j not in s][:k]), h)
                 for X in (A, B) for s in itertools.combinations(range(len(X)), k)]
        betw = [G.mmd2(np.concatenate([A[j] for j in sa]), np.concatenate([B[j] for j in sb]), h)
                for sa in itertools.combinations(range(len(A)), k) for sb in itertools.combinations(range(len(B)), k)]
        f95, bm = float(np.percentile(floor, 95)), float(np.median(betw))
        out.append(dict(seed=i, nA=len(EA), nB=len(EB), floor95=f95, between_median=bm, ratio=bm/max(f95, 1e-12)))
        print(i, len(EA), len(EB), round(f95, 4), round(bm, 4), round(bm/max(f95, 1e-12), 1), flush=True)
finally:
    sim.close()
(HERE/"pair34_posthoc.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
