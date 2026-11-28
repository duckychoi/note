"""Compute paired iBims-1 bootstrap intervals from the saved result rows."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FILES = ("encoder_phase_expanded_results.json", "encoder_phase_dev_selected_results.json")


def summarize(file):
    scores = json.loads(file.read_text(encoding="utf8"))["scores"]
    if set(scores) != {"frozen","supervised","cosine","phase","decoder"}:
        raise ValueError(f"Incomplete run: {file.name}")
    names = sorted(scores["phase"]["ibims_rows"])
    assert len(names) == 100
    boot = np.random.default_rng(4243).integers(100,size=(20000,100))
    phase = np.array([scores["phase"]["ibims_rows"][name] for name in names])
    out = {"arms":{}, "phase_comparisons":{}}
    for arm,data in scores.items():
        base = np.array([data["ibims_rows"][name] for name in names])
        assert np.allclose(base.mean(axis=0),data["ibims"])
        out["arms"][arm] = dict(ibims=data["ibims"], middlebury=data["real"],
                                synthetic_absrel=data["synthetic_absrel"],
                                selected_step=data.get("selected_step"))
        if arm == "phase":
            continue
        change = 100*(1-phase[boot,0].mean(1)/base[boot,0].mean(1))
        out["phase_comparisons"][arm] = dict(absrel_relative_reduction_pct=
            100*(1-phase[:,0].mean()/base[:,0].mean()),
            interval_95_pct=np.quantile(change,[.025,.975]).tolist(),
            delta1_change_pp=100*(phase[:,1]-base[:,1]).mean())
    return out


if __name__ == "__main__":
    reports = {name:summarize(ROOT/name) for name in FILES if (ROOT/name).exists()}
    (ROOT/"encoder_phase_summary.json").write_text(json.dumps(reports,indent=2),encoding="utf8")
    print(json.dumps(reports,indent=2))
