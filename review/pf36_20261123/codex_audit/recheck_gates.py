"""Reaggregate saved gates and test whether changing only a later photo affects the current score."""
import json
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("PF_NTH", "2")
import pf36_g6b as experiment


def saved_scores():
    six = json.loads((SOURCE / "gate6b.json").read_text(encoding="utf-8"))["rows"]
    seven = json.loads((SOURCE / "gate7.json").read_text(encoding="utf-8"))["rows"]
    summaries = {}
    for key in six:
        setting, init, kick, move, particles, model = key.split("|")
        group = "|".join([setting, init, kick, move, particles])
        if group in summaries:
            continue
        members = [v for k, v in six.items() if k.rsplit("|", 1)[0] == group]
        summaries[group] = {metric: mean(v[metric] for v in members)
                            for metric in ("forecast_energy", "filter_energy", "x_cover", "z_cover",
                                           "test_wall_sec") if metric in members[0]}
        summaries[group]["ms_per_case"] = summaries[group]["test_wall_sec"] * 1000 / 192
    pairs = {}
    for setting in ("S1", "S2"):
        keys = ["forecast_energy"] if setting == "S1" else ["forecast_energy", "filter_energy"]
        for metric in keys:
            engines = np.array([six[f"{setting}|ours|prior|none|N256|{s}"]["per_seed"][metric]
                                for s in (8100, 8200, 8300)])
            kicks = np.array([six[f"{setting}|ours|kick3|none|N256|{s}"]["per_seed"][metric]
                              for s in (8100, 8200, 8300)])
            pairs[f"{setting}|{metric}"] = dict(wins=int((kicks < engines).sum()),
                    checkpoint_averaged_diff_by_block=(engines-kicks).mean(0).tolist())
    repeat = {}
    for row in seven:
        arm = row["arm"], row["throws"], row["N"]
        members = [r for r in seven if (r["arm"], r["throws"], r["N"]) == arm]
        repeat[str(arm)] = {m: mean(r[m] for r in members)
                            for m in ("energy", "x_cover", "z_cover", "sec", "logmu_abs_err", "d_abs_err")
                            if m in row}
    return dict(six=summaries, six_pairs=pairs, seven=repeat)


def future_photo_check():
    exp = experiment
    exp.torch.set_num_threads(2)
    model, props = exp.a6.load_models()[8100], exp.load_props()
    case = exp.cases_full(62681, 48, "shift")
    # A single observation in the 25..29 scoring bin, followed by a later observation at frame 30.
    case["mask"][:, 25:] = 0
    case["mask"][:, [25, 30]] = 1
    changed = dict(case, obs=case["obs"].copy())
    changed["obs"][:, 30, 1] += .10
    scores = {}
    for arm in ("prior", "gap", "kick3"):
        def get_scores(c):
            result = exp.run_filter(c, "ours", arm, 256, 40, 62900, model, props, resample_from=24)
            return result["filt_rmse_25"]
        original = get_scores(case)
        modified = get_scores(changed)
        delta = np.abs(original-modified)
        scores[arm] = dict(case_count=len(delta), changed_above_1e_minus_8_m=int((delta > 1e-8).sum()),
                           max_absolute_rmse_difference_m=float(delta.max()),
                           original_at_max_m=float(original[delta.argmax()]),
                           modified_at_max_m=float(modified[delta.argmax()]))
    return dict(seed=62681, particles=256, current_frame=25, changed_frame=30,
                change_in_height_m=.10, scores=scores)


if __name__ == "__main__":
    result = saved_scores()
    masks = [experiment.cases_full(seed, 48, "shift")["mask"][:, 25:40]
             for seed in range(56000, 56004)]
    mask = np.concatenate(masks)
    result["observations_25_to_39"] = dict(total=int(mask.sum()), cases=len(mask),
                                          cases_with_any=int(mask.any(axis=1).sum()))
    result["future_photo_check"] = future_photo_check()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (HERE / "recheck_gates.json").write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(result["future_photo_check"]))
