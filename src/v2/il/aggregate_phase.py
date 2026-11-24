"""Phase experiment summary: mean±sd over seeds per (env, method) for each test condition.
usage: python aggregate_phase.py E:/0.Project/fm34_env/phase"""
import sys, os, json, glob
import numpy as np
from collections import defaultdict

RUN = sys.argv[1]
D = defaultdict(list)
for jf in sorted(glob.glob(os.path.join(RUN, "*.json"))):
    L = json.load(open(jf))
    if "final" in L:
        D[(L["args"]["env"], L["args"]["method"])].append(L)


def ms(v):
    v = [x for x in v if x is not None]
    return (f"{np.mean(v):.2f}±{np.std(v):.2f}" if v else "  n/a  ")


out = {}
print(f"{'env':12s} {'method':18s} {'n':>2s} | {'ret/exp':>10s} {'PLV':>10s} {'|off|':>10s} | "
      f"{'jump PLVpost':>12s} {'off_post':>10s} | {'PLV x0.85':>10s} {'PLV x1.15':>10s}")
for (env, m), Ls in sorted(D.items()):
    f = [L["final"] for L in Ls]; er = Ls[0]["expert_return"]
    row = dict(ret=[x["nominal"]["ret"] / er for x in f], plv=[x["nominal"]["plv"] for x in f],
               off=[x["nominal"]["abs_off"] for x in f], plv_post=[x["jump"]["plv_post"] for x in f],
               off_post=[abs(x["jump"]["off_post"]) if x["jump"]["off_post"] is not None else None for x in f],
               plv_slow=[x["tempo0.85"]["plv"] for x in f], plv_fast=[x["tempo1.15"]["plv"] for x in f])
    out[f"{env}/{m}"] = row
    print(f"{env[:12]:12s} {m:18s} {len(f):2d} | {ms(row['ret']):>10s} {ms(row['plv']):>10s} {ms(row['off']):>10s} | "
          f"{ms(row['plv_post']):>12s} {ms(row['off_post']):>10s} | {ms(row['plv_slow']):>10s} {ms(row['plv_fast']):>10s}")
json.dump(out, open(os.path.join(RUN, "summary_phase.json"), "w"), indent=1)
