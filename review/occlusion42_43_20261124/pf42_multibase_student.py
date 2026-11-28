"""Run equal-budget RGB-only heads after the multi-baseline teacher gate."""
import json
import sys

import numpy as np
import torch

import pf42_b_teacher as b
from pf42_b_report import interval


OUT = b.prior.OUT / "multibase-b"


def run():
    torch.set_num_threads(2)
    gate = json.loads((OUT/"results.json").read_text(encoding="utf8"))
    if not gate["legacy_gate"]:
        print("No student training: the teacher did not beat the historical continuous median.")
        return
    data = {k:b.prior.load(k) for k in ("synthetic","fit",*b.prior.DATASETS)}
    with np.load(OUT/"student-targets.npz", allow_pickle=False) as a:
        direct, direct_long, depth, p = a["direct"], a["direct_long"], a["depth"], a["probability"]
    assert len(direct) == len(data["fit"]["base"]) and np.isfinite(direct).all()
    np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
    data["fit"]["y"] = direct[:,None]
    seeds = b.train_student(data,(np.log(depth),p),output=OUT,extra_direct=direct_long)
    summary = {}
    for name in b.prior.DATASETS:
        groups = sorted(seeds[0]["scores"]["direct"][name])
        direct_all, proposed, long_only = np.array([[[r["scores"][a][name][g] for g in groups] for r in seeds]
                                                  for a in ("direct","separated","direct_long")]).mean(1)
        gain_all, ci_all = interval(proposed,direct_all)
        gain_long, ci_long = interval(proposed,long_only)
        summary[name] = dict(direct=float(direct_all.mean()),direct_long=float(long_only.mean()),
                             proposed=float(proposed.mean()),
                             relative=float(gain_all),interval=ci_all.tolist(),
                             relative_long=float(gain_long),interval_long=ci_long.tolist())
    (OUT/"student-results.json").write_text(json.dumps(dict(seeds=seeds,metrics=summary),indent=2),encoding="utf8")
    print(summary,flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    run()
