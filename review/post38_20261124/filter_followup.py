"""Post-hoc design, fresh seeds. Uses the original estimator with an optional sensor mixture."""
import json
import numpy as np
from filter_gate import CONDITIONS, METHODS, OUT, fit_map, generate, run


def main():
    _,z,q = generate(72331,50)
    table = fit_map(z,q["aligned"])
    dev = [generate(72332+offset,60,coupled) for offset,coupled in ((0,True),(100,False))]
    methods = [(name,rich) for name in METHODS for rich in (False,True)]
    scales = (.125,.25,.5,1.,2.,4.,8.,16.)
    chosen, scores = {}, {}
    for name,rich in methods:
        tag = name + ("_sensor" if rich else "")
        grid = {s:float(np.mean([run(x,z,q["aligned"],table,name,s,False,rich)["rmse"].mean()
                                for x,z,q in dev])) for s in scales}
        chosen[tag] = min(grid,key=grid.get)
        scores[tag] = grid
    boot = np.random.default_rng(183).integers(240,size=(10000,240))
    result = dict(chosen=chosen,dev=scores,table=table.tolist(),summary={},contrasts={},trajectories={})
    for offset,coupled in ((0,True),(100,False)):
        x,z,q = generate(72333+offset,240,coupled)
        for c in CONDITIONS:
            group = ("coupled/" if coupled else "uncoupled/")+c
            samples = {}
            result["summary"][group] = {}
            for name,rich in methods:
                tag = name + ("_sensor" if rich else "")
                data = run(x,z,q[c],table,name,chosen[tag],False,rich)
                samples[tag] = {k:v.tolist() for k,v in data.items() if k in ("rmse","postjump")}
                result["summary"][group][tag] = {k:float(np.mean(v)) for k,v in samples[tag].items()}
            result["trajectories"][group] = samples
            matches = {}
            for a,b in (("jump_fixed_sensor","jump_quality_sensor"),("jump_fixed_sensor","adaptive_sensor"),
                        ("jump_quality_sensor","adaptive_sensor"),("fixed","jump_fixed_sensor"),
                        ("jump_fixed","jump_fixed_sensor"),("fixed_sensor","jump_fixed_sensor")):
                left = np.array(samples[a]["rmse"])
                delta = left - samples[b]["rmse"]
                matches[f"{a} minus {b}"] = dict(absolute=float(delta.mean()), relative=float(delta.mean()/left.mean()),
                                                  ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist())
            result["contrasts"][group] = matches
    (OUT/"filter_followup_results.json").write_text(json.dumps(result,separators=(",",":")),encoding="utf-8")
    print(chosen)
    for group,s in result["summary"].items():
        print(group, {k:round(v["rmse"],3) for k,v in s.items()})


if __name__ == "__main__":
    main()
