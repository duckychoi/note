"""Verdicts recomputed from saved per-member JSON (PREREG.md).  python agg34.py s1 <dir>  |  python agg34.py s3 <dir> <s2.json> [<s1 dir>]"""
import glob, json, os, sys
import numpy as np


def load(d, stage):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, f"{stage}_*.json")))]


def member_stats(L):
    E = L["final"]["episodes"]
    return dict(n=len(E), task_ok=sum(e["task_ok"] for e in E), okF=sum(e["order_ok_F"] for e in E), okR=sum(e["order_ok_R"] for e in E),
                undefined=sum(not np.isfinite(e["lag"]) if e["lag"] is not None else 1 for e in E),
                v=float(np.mean([e["v"] for e in E])), verr=float(np.mean([abs(e["v"] - 1.0) for e in E])),
                up=float(np.mean([e["upright"] for e in E])), ctrl=float(np.mean([e["ctrl"] for e in E])),
                s=float(np.mean([e["s"] for e in E])), lag=[None if not np.isfinite(e["lag"]) else round(e["lag"], 3) for e in E],
                Tb=float(np.nanmedian([e["Tb_med"] if e["Tb_med"] is not None else np.nan for e in E])))


def s1(d):
    rows = [(L["member"], member_stats(L)) for L in load(d, "s1")]
    out = dict(members=[dict(**m, **st) for m, st in rows], verdict={})
    for w in (0.5, 1.0):
        seeds = {}
        for m, st in rows:
            if m["arm"] == "oracle" and m["w"] == w:
                okk = st["okF"] if m["label"] == "F" else st["okR"]
                seeds[f"{m['label']}{m['seed']}"] = dict(task_ok=st["task_ok"], order_ok=okk, pass_=bool(st["task_ok"] >= 9 and okk >= 8))
        out["verdict"][str(w)] = dict(seeds=seeds, S1_pass=bool(len(seeds) == 6 and all(v["pass_"] for v in seeds.values())),
                                     task_ok_total=sum(v["task_ok"] for v in seeds.values()))
    passing = [w for w in ("0.5", "1.0") if out["verdict"][w]["S1_pass"]]
    if len(passing) == 2:
        a, b = out["verdict"]["0.5"]["task_ok_total"], out["verdict"]["1.0"]["task_ok_total"]
        lock = 1.0 if b > a else 0.5
    else:
        lock = float(passing[0]) if passing else None
    out["S1_pass"] = bool(passing); out["w_lock"] = lock
    return out


def s3(d, s2path, s1dir=None):
    s2 = json.load(open(s2path)); gap = s2["gap"]
    rows = [(L["member"], member_stats(L)) for L in load(d, "s3")]
    out = dict(gap=gap, members=[dict(**m, **st) for m, st in rows], arms={})
    for arm in sorted({m["arm"] for m, _ in rows}):
        per = {}
        for k in sorted({m["seed"] for m, _ in rows}):
            F = next(st for m, st in rows if m["arm"] == arm and m["label"] == "F" and m["seed"] == k)
            R = next(st for m, st in rows if m["arm"] == arm and m["label"] == "R" and m["seed"] == k)
            shift = F["s"] - R["s"]
            per[k] = dict(sF=F["s"], sR=R["s"], shift=shift, shift_over_gap=shift / gap, taskF=F["task_ok"], taskR=R["task_ok"],
                          pass_=bool(shift >= gap / 2 and F["task_ok"] >= 9 and R["task_ok"] >= 9))
        sts = [st for m, st in rows if m["arm"] == arm]
        out["arms"][arm] = dict(seeds=per, pass_=all(v["pass_"] for v in per.values()), n_seed_pass=sum(v["pass_"] for v in per.values()),
                                task_rate=float(np.mean([st["task_ok"] / st["n"] for st in sts])), verr=float(np.mean([st["verr"] for st in sts])),
                                ctrl=float(np.mean([st["ctrl"] for st in sts])))
    if s1dir:
        t = [member_stats(L) for L in load(s1dir, "s1") if L["member"]["arm"] == "task"]
        out["task_only_ref"] = dict(task_rate=float(np.mean([st["task_ok"] / st["n"] for st in t])), verr=float(np.mean([st["verr"] for st in t])),
                                    ctrl=float(np.mean([st["ctrl"] for st in t])), s=[st["s"] for st in t])
    return out


def s4prep(d):
    out = dict(members=[], verdict={})
    for L in load(d, "s4prep"):
        m = L["member"]; E = L["final"]["episodes"]
        okM = sum(bool(np.isfinite(e["lag"]) and min(abs(e["lag"] - .5) % 1, 1 - abs(e["lag"] - .5) % 1) < .125) for e in E)
        st = member_stats(L); st.update(okM=okM, pass_=bool(st["task_ok"] >= 9 and okM >= 8))
        out["members"].append(dict(**m, **st))
    for w in (0.5, 1.0):
        out["verdict"][str(w)] = {f"s{x['seed']}": x["pass_"] for x in out["members"] if x["w"] == w}
    return out


if __name__ == "__main__":
    st, d = sys.argv[1], sys.argv[2]
    r = s1(d) if st == "s1" else (s4prep(d) if st == "s4prep" else s3(d, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
    json.dump(r, open(os.path.join(d, f"verdict_{st}.json"), "w"), indent=1)
    for m in r["members"]:
        print({k: m.get(k) for k in ("arm", "label", "w", "seed", "task_ok", "okF", "okR", "okM", "undefined", "v", "up", "s", "Tb")})
    print(json.dumps({k: v for k, v in r.items() if k != "members"}, indent=1))
