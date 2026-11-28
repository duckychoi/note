"""Equal-weight averages over poses, then scenes; preserve each held-out block."""
from pathlib import Path
import json
from statistics import mean
import numpy as np

ROOT = Path(__file__).resolve().parent


def metrics(row):
    result = {k:row[k] for k in ("median_response_mm","response_after_scale_pct","pattern_over_ordinary")}
    for image in ("ordinary","pattern"):
        result.update({image+"_"+k:v for k,v in row[image].items()})
    for policy,(mae,valid) in row["anchors"].items():
        result[policy+"_mae_mm"] = mae
        result[policy+"_returns"] = valid
    return result


def average(rows):
    return {k:mean(r[k] for r in rows) for k in rows[0]}


if __name__=="__main__":
    out = {}
    for path in sorted(ROOT.glob("*.json")):
        if path.name=="summary.json":
            continue
        log = json.loads(path.read_text())
        by_scene = {s:average([metrics(r) for r in log["rows"] if r["scene"]==s]) for s in (4,5,8,9)}
        assert len(log["rows"])==8 and log["rows"][0]["control_max_mm"]==0
        out[path.stem] = dict(by_scene=by_scene,mean=average(list(by_scene.values())))
        suffix = "" if log["preset"]=="depthpro2-s" else "_h"
        train = json.loads((ROOT/f"chrome_screw{suffix}.json").read_text())
        if all("center_valid_log_ratios" in r for r in train["rows"]):
            ratios = [a for r in train["rows"] if r["scene"] in (4,8) for a in r["center_valid_log_ratios"]]
            factor = float(np.exp(np.median(ratios)))
            cache = ROOT.parents[2]/"research-sources"/"physical43_20261124"
            if log["object"]=="Gear":
                cache = cache/"Gear"
            trials = []
            for r in log["rows"]:
                if log["object"]!="Gear" and r["scene"] in (4,8):
                    continue
                coords = cache/"depthpro2_probe"/f"{log['preset']}_{r['scene']}_{r['view']}.npz"
                with np.load(coords) as d:
                    trials.append(float(np.abs(factor*d["pred"]-d["gt"]).mean()))
            out[path.stem]["fixed_scale"] = dict(factor=factor,n_dev_success=len(ratios),
                        test_views=len(trials),test_mae_mm=mean(trials),per_view=trials)
        print(path.stem,{k:round(v,2) for k,v in out[path.stem]["mean"].items()})
    (ROOT/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf8")
