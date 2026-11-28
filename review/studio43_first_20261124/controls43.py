"""Post-gate diagnostic controls. Do not use these to select or retrain any arm."""
import json
from collections import defaultdict

import numpy as np

from data import DEV, HERE
from evaluate import metric_setup
from studio import base_depth, real_frame


def main():
    rows = json.loads((HERE / "inputs.json").read_text())["rows"]
    offsets, distances = [], []
    for row in rows:
        if row["scene"] not in DEV:
            continue
        rgb, gt, fx = real_frame(row)
        h, w = gt.shape
        iy, ix = np.indices((4, 8))
        y, x = ((iy + .5) * h / 4).astype(int), ((ix + .5) * w / 8).astype(int)
        observed = gt[y, x]
        keep = ((ix + iy) % 2 == 0) & (observed >= .2) & (observed <= 10)
        offsets.extend(np.log(observed[keep] / base_depth(row["key"], rgb, fx)[y, x][keep]))
        distances.extend(observed[keep])
    scale = float(np.exp(np.median(offsets)))
    constant = float(np.median(distances))
    by_method = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row["scene"] in DEV:
            continue
        rgb, gt, fx = real_frame(row)
        metric, _ = metric_setup(row, rgb, gt, fx)
        focus = row.get("focus", 1.5)
        for name, z in (
                ("H+_sparse_scale", scale * base_depth(row["key"], rgb, fx)),
                ("development_constant", np.full(rgb.shape[:2], constant, np.float32)),
                ("sharp_input", np.full(rgb.shape[:2], focus, np.float32))):
            for key, value in metric(z).items():
                by_method[name][key][row["scene"]].append(value)
    output = {"post_hoc": True, "n_calibration": len(offsets), "H+_multiplier": scale,
              "constant_m": constant, "means": {}, "by_unit": {}}
    for name, metrics in by_method.items():
        by_unit = {key: {scene: float(np.mean(values)) for scene, values in units.items()}
                   for key, units in metrics.items()}
        output["by_unit"][name] = by_unit
        output["means"][name] = {key: float(np.mean(list(units.values())))
                                 for key, units in by_unit.items()}
    (HERE / "controls43.json").write_text(json.dumps(output, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
