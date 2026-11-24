"""Paired simulator diagnostic. Intervene on friction with launch, damping and noise fixed."""
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("PF_NTH", "2")
import pf36_g8 as gate


def observe_effect(launch, damping, mu_pair, seed):
    paths = []
    for mu in mu_pair:
        models = [gate.g.mk(mu, d) for d in damping]
        position, _ = gate.g.simulate(models, launch.copy(),
                                      np.random.default_rng(seed), gate.g.T)
        paths.append(position)
    x_diff_mm = 1000 * np.abs(paths[0][:, :, 0] - paths[1][:, :, 0])
    endpoint = x_diff_mm[:, -1]
    return {
        "frictions": mu_pair,
        "n": len(endpoint),
        "early_peak_abs_x_median_mm": float(np.median(x_diff_mm[:, :gate.g.CUT].max(axis=1))),
        "endpoint_abs_x_median_mm": float(np.median(endpoint)),
        "endpoint_abs_x_p90_mm": float(np.quantile(endpoint, .9)),
        "endpoint_above_15mm_count": int((endpoint > 1000 * gate.g.SIG).sum()),
        "endpoint_abs_x_median_by_damping_band_mm": {
            "low": float(np.median(endpoint[damping < .4])),
            "high": float(np.median(endpoint[damping > .4])),
        },
    }


def run():
    seed = 67238
    rng = np.random.default_rng(seed)
    n = 256
    damping = np.r_[rng.uniform(.07, .17, n//2), rng.uniform(.55, .8, n//2)]
    output = {}
    # Separate seeds for launch and kicks. Gate 8's two launch types share all other launch coordinates.
    for kind in ("drop", "oblique"):
        start = gate.launch(np.random.default_rng(seed+1), n, kind)
        output[kind] = [observe_effect(start, damping, pair, seed+2)
                        for pair in ([.12, 1.05], [.10, .20], [.90, 1.20])]
    return {"development_seed": seed, "frame": gate.g.T-1, "observation_sd_mm": 1000*gate.g.SIG,
            "paired_interventions": output}


if __name__ == "__main__":
    data = run()
    result_path = HERE / "gate8_friction_effect.json"
    if "--check" in sys.argv:
        assert json.loads(result_path.read_text(encoding="utf-8")) == data
    else:
        result_path.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(data["paired_interventions"]))
