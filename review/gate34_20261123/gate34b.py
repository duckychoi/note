"""Gate 0b: B_i = A_i with front-leg phase +pi, re-optimize only const/amplitudes for the same speed."""
import json
from pathlib import Path
import numpy as np
import gate34 as G

HERE = Path(__file__).resolve().parent
VSTAR = 1.0


def main():
    z = np.load(HERE/f"gaits_v{VSTAR}.npz")
    sim = G.Sim()
    try:
        gaits = {"A_inphase": dict(seeds=[]), "B_frontshift": dict(seeds=[])}
        for i in range(6):
            pa = z[f"A_inphase_s{i}"].reshape(6, 5).copy()
            pa[G.FTHIGH, 2] = pa[G.BTHIGH, 2]          # materialize delta=0 constraint
            epsA = G.episodes(sim, pa.ravel(), None, VSTAR, [130000+10*i+j for j in range(4)])
            gaits["A_inphase"]["seeds"].append(dict(params=pa.ravel(), eps=epsA))
            base = pa.copy(); base[3:6, 2] += np.pi
            free = np.r_[[0, 1, 3]]                     # c, a1, a2 per joint
            rng = np.random.default_rng(9100+i)
            center = base[:, free].ravel(); scale = np.full(18, .15)
            def make(x):
                p = np.repeat(base[None], len(x), 0); p[:, :, free] = x.reshape(-1, 6, 3); return p.reshape(len(x), 30)
            cands = []
            for g in range(8):
                x = rng.normal(center, scale, (64, 18))
                avg = np.mean([sim.task(*sim.run(make(x), 140000+1000*i+2*g+j, G.H_SEARCH, None), VSTAR, G.H_SEARCH)[0]
                               for j in range(2)], axis=0)
                el = x[np.argsort(avg)[-10:]]
                center, scale = el.mean(0), np.maximum(el.std(0), .02)
                cands += [center.copy(), el[-1].copy()]
            cands = np.array(cands)
            val = np.mean([sim.task(*sim.run(make(cands), 140000+1000*i+100+j, G.H_SEARCH, None), VSTAR, G.H_SEARCH)[0]
                           for j in range(4)], axis=0)
            pb = make(cands[np.argmax(val)][None])[0]
            epsB = G.episodes(sim, pb, None, VSTAR, [135000+10*i+j for j in range(4)])
            gaits["B_frontshift"]["seeds"].append(dict(params=pb, eps=epsB))
            print(i, [(round(e["v"], 2), e["task_ok"]) for e in epsA], [(round(e["v"], 2), e["task_ok"]) for e in epsB], flush=True)
        res = G.evaluate(sim, gaits)
        res.update(vstar=VSTAR, design="B_i = A_i front-leg phase +pi, amplitudes re-optimized",
                   paramsB=[s["params"].tolist() for s in gaits["B_frontshift"]["seeds"]])
        (HERE/"gate0b_v1.0.json").write_text(json.dumps(res, indent=1, allow_nan=False)+"\n", encoding="utf-8")
        for k in ['task_pass_episodes', 'missing_cycles', 'floor95', 'between_median', 'between_min', 'ratio', 'gate0_pass',
                  'controls_vs_A_test', 'fourier_auc_B_farther_than_A', 'speed', 'realized_thigh_phase_rad']:
            print(k, res[k])
    finally:
        sim.close()


if __name__ == "__main__":
    main()
