"""Retest published solver and policy comparisons on independent sampling seeds."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
NOTE = HERE.parents[1]
sys.path.insert(0, str(NOTE/"src/v2/pf"))
import e35_audit as solver
import e35p_gate as policy


def ed(x, y):
    return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())


def main():
    solver.K = 256
    cfg = json.loads((NOTE/"results/v2/e35p_gate.json").read_text())
    out = {"particles":solver.K, "evaluation_seeds":list(range(8120,8132)), "generation":{}, "policy":{}}
    for budget in (4,8):
        for method, run in [("euler",solver.euler),("heun",solver.heun),("smc",solver.smc)]:
            values = []
            for seed in out["evaluation_seeds"]:
                ref = solver.G.sample(solver.K, np.random.default_rng(seed+10000))
                x, cost = run(budget, np.random.default_rng(seed))
                assert cost == budget
                values.append(ed(x,ref))
            out["generation"][f"{budget}/{method}"] = values
    for budget in (64,256):
        for method in ("end","fk","guid","tds"):
            key = f"s3.0/C{budget}/{method}"
            params = cfg[key]["cfg"]
            values = []
            for seed in out["evaluation_seeds"]:
                q,gq = policy.make_critic(3.0,seed)
                x,cost = policy.run(method,**params,q=q,gq=gq,Dn=256,rng=np.random.default_rng(seed+20000))
                assert cost == cfg[key]["cost"] and cost <= budget
                values.append(float(policy.E.Q(policy.P,x).mean()))
            out["policy"][key] = dict(config=params,cost=cost,mean_true_q_by_seed=values)
    return out


if __name__ == "__main__":
    (HERE/"particle35.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
