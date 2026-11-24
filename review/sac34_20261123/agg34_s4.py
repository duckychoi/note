"""S4 verdict from saved JSON (PREREG 'S4 수정 1').  python agg34_s4.py <s4 dir> <s2.json> <s2M.json>"""
import glob, json, math, os, sys
import numpy as np

STY = ["F", "R", "M"]
T95_ONESIDED_DF2 = 2.920


def dcirc(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1 - d)


def classify(segs, centers, task_ok=True):
    """style only if all 4 segments: R>=.5, same nearest centre within .125, and >=70% of individual touchdown lags
    within .125 of that centre (codex r2: blocks alternating .34/.66 being counted as M)."""
    if not task_ok:
        return "fail"
    labs = []
    for seg in segs:
        m, R, n = seg[0], seg[1], seg[2]
        if m is None or not np.isfinite(m) or R < 0.5:
            return "none"
        d = [dcirc(m, c) for c in centers]; k = int(np.argmin(d))
        if d[k] >= 0.125:
            return "none"
        ls = seg[3] if len(seg) > 3 else []
        if not ls or np.mean([dcirc(x, centers[k]) < 0.125 for x in ls]) < 0.7:
            return "none"
        labs.append(k)
    return STY[labs[0]] if len(set(labs)) == 1 else "none"


def switches(segs, centers):
    ks = [int(np.argmin([dcirc(sg[0], c) for c in centers])) for sg in segs if sg[0] is not None and np.isfinite(sg[0])]
    return sum(a != b for a, b in zip(ks, ks[1:])) / max(len(ks) - 1, 1)


def summarize(E, centers):
    lab = [classify(e["segs"], centers, e["task_ok"]) for e in E]
    cnt = {k: lab.count(k) for k in STY + ["none", "fail"]}
    h = np.array([cnt[k] for k in STY], float) + 0.5; h /= h.sum()
    nst = sum(cnt[k] for k in STY)
    share = [cnt[k] / nst if nst else 0.0 for k in STY]
    hw = np.array([len(E), 0, 0], float) + 0.5; hw /= hw.sum(); KLworst = float(np.sum(hw * np.log(hw * 3)))
    KLraw = float(np.sum(h * np.log(h * 3)))
    styled_share = nst / len(E); trate = float(np.mean([e["task_ok"] for e in E]))
    valid = styled_share >= 0.5 and trate >= 0.9
    out = dict(counts=cnt, KL=KLraw if valid else KLworst, KL_raw=KLraw, KL_valid=valid, styled_share=styled_share, entropy=float(-np.sum(h * np.log(h))),
               n_styles_ge10=int(sum(s >= 0.10 for s in share)) if nst else 0, task_rate=float(np.mean([e["task_ok"] for e in E])),
               switch_rate=float(np.mean([switches(e["segs"], centers) for e in E])))
    z = [e.get("z", -1) for e in E]
    if any(zz >= 0 for zz in z):
        out["follow_rate"] = float(np.mean([l == STY[zz] for l, zz in zip(lab, z) if zz >= 0]))
    return out


def qualify(s2p, s2mp, centers):
    per = {k: [0, 0] for k in STY}
    for path in (s2p, s2mp):
        S2 = json.load(open(path)); ho = str(S2["ho_seed"])
        for key, v in S2["demonstrators"].items():
            if not key.endswith(ho):
                continue
            for segs, ok in zip(v["segs"], v["task_ok"]):
                if ok:
                    per[key[0]][0] += 1; per[key[0]][1] += classify(segs, centers) == key[0]
    n = sum(a for a, b in per.values()); k = sum(b for a, b in per.values())
    ok = bool(n and k / n >= 0.95 and all(a >= 10 and b / a >= 0.9 for a, b in per.values()))
    return ok, k / max(n, 1), n, per


def main(d, s2p, s2mp):
    L = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "s4_*.json")))]
    centers = L[0]["centers"]
    # held-out classifier accuracy on demos (demonstrator seed 2, task-ok episodes)
    cls_ok, acc, acc_n, per = qualify(s2p, s2mp, centers)
    out = dict(centers=centers, classifier_heldout_acc=acc, classifier_n=acc_n, classifier_per_style=per, members={}, arms={})
    for x in L:
        m = x["member"]; key = f"{m['arm']}_s{m['seed']}"
        out["members"][key] = {mode: summarize(x["final"][mode], centers) for mode in ("stoch", "det")}
    for arm in sorted({x["member"]["arm"] for x in L}):
        out["arms"][arm] = {mode: {k: [out["members"][f"{arm}_s{s}"][mode][k] for s in (0, 1, 2)] for k in ("KL", "KL_valid", "styled_share", "n_styles_ge10", "task_rate", "switch_rate")}
                            for mode in ("stoch", "det")}
    ver = {}
    for mode in ("stoch", "det"):
        if "amp_phase" in out["arms"] and "fm_phase" in out["arms"]:
            diff = np.array(out["arms"]["amp_phase"][mode]["KL"]) - np.array(out["arms"]["fm_phase"][mode]["KL"])
            lb = float(diff.mean() - T95_ONESIDED_DF2 * diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None
            dtask = float(np.mean(out["arms"]["fm_phase"][mode]["task_rate"]) - np.mean(out["arms"]["amp_phase"][mode]["task_rate"]))
            amp_collapse = sum(n <= 2 for n in out["arms"]["amp_phase"][mode]["n_styles_ge10"]) >= 2
            ver[mode] = dict(KL_diff_amp_minus_fm=diff.tolist(), mean=float(diff.mean()), lower95=lb, task_fm_minus_amp=dtask,
                             amp_collapsed=amp_collapse, pass_=bool(lb is not None and lb > 0 and dtask >= -0.05))
    out["classifier_ok"] = cls_ok
    # positive control (same latent-free class): all 3 styles each >=10% of styled episodes, stochastic eval, in >=2 of 3 seeds
    oc = [out["members"][f"oracle_mix_s{s_}"]["stoch"] for s_ in (0, 1, 2)] if "oracle_mix" in out["arms"] else []
    out["oracle_mix_ok"] = bool(oc and sum(o["n_styles_ge10"] == 3 and o["KL_valid"] for o in oc) >= 2)   # also task>=.9 and styled>=.5
    for mode, v in ver.items():
        v["pass_"] = bool(v["pass_"] and v["mean"] >= 0.10)          # minimal practical KL improvement 0.10 nats
        decidable = out["classifier_ok"] and out["oracle_mix_ok"]
        v["pass_"] = bool(v["pass_"] and decidable)
        v["status"] = "undecidable" if not decidable else ("pass" if v["pass_"] else "fail")
    out["verdict"] = ver
    if "fm_contrast" in out["arms"] and "amp_phase" in out["arms"]:      # auxiliary, exploratory (PREREG S4 수정 4)
        aux = {}
        for mode in ("stoch", "det"):
            diff = np.array(out["arms"]["amp_phase"][mode]["KL"]) - np.array(out["arms"]["fm_contrast"][mode]["KL"])
            aux[mode] = dict(KL_diff_amp_minus_fmc=diff.tolist(), mean=float(diff.mean()),
                             lower95=float(diff.mean() - T95_ONESIDED_DF2 * diff.std(ddof=1) / math.sqrt(len(diff))),
                             task_fmc_minus_amp=float(np.mean(out["arms"]["fm_contrast"][mode]["task_rate"]) - np.mean(out["arms"]["amp_phase"][mode]["task_rate"])))
        out["aux_fm_contrast"] = aux
    json.dump(out, open(os.path.join(d, "verdict_s4.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "members"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
