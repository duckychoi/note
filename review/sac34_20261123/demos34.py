"""S2 (PREREG.md): roll out the S1 oracle policies (locked w) to build demo sets D_F, D_R.
Demonstrator-level split: seeds 0,1 = train demos (12 eps each), seed 2 = held-out (20 eps).  Resets 600000+e.
  python demos34.py --s1 <s1 dir> --w 0.5 --out <dir>
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sac34 as C  # noqa: E402
from vec import BSAC  # noqa: E402


def seg_lags(ev):
    """4 segments of 200 steps over steps 200-1000: (circular mean lag, R, n) per segment"""
    out = []
    for j in range(4):
        a, b = 200 + 200 * j, 400 + 200 * j
        ls = [l for t_, l in ev.lags if a <= t_ < b]
        m, R = C.circ_mean(ls) if len(ls) >= 3 else (float("nan"), 0.0)
        out.append((m, R, len(ls), [float(x) for x in ls]))
    return out

p = argparse.ArgumentParser()
p.add_argument("--s1", required=True); p.add_argument("--w", type=float, required=True); p.add_argument("--out", required=True)
p.add_argument("--ckpt", default="s1_actors.pt"); p.add_argument("--train_seeds", default="0,1"); p.add_argument("--ho_seed", type=int, default=2); p.add_argument("--labels", default="F,R"); p.add_argument("--tag", default=""); p.add_argument("--qual", default=None)
p.add_argument("--n_train", type=int, default=12); p.add_argument("--n_ho", type=int, default=20); p.add_argument("--skip", type=int, default=100)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda")
ck = torch.load(os.path.join(args.s1, args.ckpt)); LABS = args.labels.split(",")
members = ck["members"]; S = len(members)
agent = BSAC(S, C.OD, 6, dev); agent.actor.load_state_dict(ck["actor"])
TRS = [int(x) for x in args.train_seeds.split(",")]
if "M" in LABS:   # PREREG S4 수정 2: M demonstrators chosen by rule, not by hand
    assert args.qual, "--qual verdict_s4prep.json is required for M demos"
    Q = json.load(open(args.qual))["verdict"][str(args.w)]
    passing = sorted(int(k[1:]) for k, v in Q.items() if v)
    assert len(passing) >= 2, f"S4 stopped: fewer than 2 qualified M demonstrators {passing}"
    args.ho_seed = 2 if 2 in passing else passing[-1]
    TRS = [x for x in passing if x != args.ho_seed]
    print("M demonstrators: train", TRS, "held-out", args.ho_seed, flush=True)
use = {i: m for i, m in enumerate(members) if m["arm"] == "oracle" and m["w"] == args.w and m["label"] in LABS and m["seed"] in TRS + [args.ho_seed]}
assert len(use) == (len(TRS) + 1) * len(LABS), use
envs = {i: C.Cheetah() for i in use}
n_ep = max(args.n_train, args.n_ho)
eps = {i: [] for i in use}
for e in range(n_ep):
    obs = np.zeros((S, C.OD), np.float32)
    for i, ev in envs.items():
        obs[i] = ev.reset(600_000 + e)
    O = {i: [obs[i].copy()] for i in use}; PH = {i: [] for i in use}
    XS = {i: [float(envs[i].d.qpos[0])] for i in use}; UP = {i: [] for i in use}; CT = {i: [] for i in use}
    for t in range(C.H):
        A = agent.act(torch.tensor(obs, device=dev), det=True).cpu().numpy()
        for i, ev in envs.items():
            PH[i].append(ev.phase())
            o, _, inf = ev.step(A[i]); obs[i] = o; O[i].append(o.copy())
            XS[i].append(float(ev.d.qpos[0])); UP[i].append(inf["up"]); CT[i].append(inf["ctrl"])
    for i, ev in envs.items():
        rec = C.episode_record(ev, XS[i], UP[i], CT[i])
        Oi = np.array(O[i])[:, :C.K]
        rec["X"] = np.concatenate([Oi[:-1], Oi[1:]], 1)[args.skip:].astype(np.float32)
        rec["P"] = np.array(PH[i], np.float32)[args.skip:]
        rec["joints"] = Oi[200:, 2:8]
        rec["lags_all"] = [l for t_, l in ev.lags if t_ >= 200]
        rec["segs"] = seg_lags(ev)
        eps[i].append(rec)
    print("ep", e, {f"{use[i]['label']}{use[i]['seed']}": (round(eps[i][-1]["v"], 2), eps[i][-1]["task_ok"], round(eps[i][-1]["s"], 2)) for i in use}, flush=True)

out, rep = {}, dict(w=args.w, counts={}, demonstrators={}, train_seeds=TRS, ho_seed=args.ho_seed)
psd = {}
for lab in LABS:
    tr, ho = [], []
    for i, m in use.items():
        if m["label"] != lab:
            continue
        L = eps[i][: (args.n_ho if m["seed"] == args.ho_seed else args.n_train)]
        ok = [r for r in L if r["task_ok"]]
        rep["demonstrators"][f"{lab}{m['seed']}"] = dict(n=len(L), n_task_ok=len(ok), s=[r["s"] for r in L], lag=[r["lag"] for r in L],
                                                       segs=[r["segs"] for r in L], task_ok=[r["task_ok"] for r in L],
                                                       v=[r["v"] for r in L], Tb_med=[r["Tb_med"] for r in L])
        (ho if m["seed"] == args.ho_seed else tr).extend(ok)
        psd[(lab, m["seed"])] = np.mean([np.stack([welch(r["joints"][:, j], fs=20, nperseg=128)[1] for j in range(6)]) for r in ok], 0) if ok else None
    rep["counts"][lab] = dict(train=len(tr), heldout=len(ho))
    out[f"X_{lab}"] = np.concatenate([r["X"] for r in tr]) if tr else np.zeros((0, 34), np.float32)
    out[f"P_{lab}"] = np.concatenate([r["P"] for r in tr]) if tr else np.zeros(0, np.float32)
    out[f"Xho_{lab}"] = np.concatenate([r["X"] for r in ho]) if ho else np.zeros((0, 34), np.float32)
    out[f"Pho_{lab}"] = np.concatenate([r["P"] for r in ho]) if ho else np.zeros(0, np.float32)
    out[f"lagD_{lab}"] = C.circ_mean(np.concatenate([r["lags_all"] for r in tr]))[0] if tr else np.nan
    out[f"sho_{lab}"] = np.array([r["s"] for r in ho])
    rep[f"lagD_{lab}"] = float(out[f"lagD_{lab}"])

sF, sR = out.get("sho_F", np.zeros(0)), out.get("sho_R", np.zeros(0))
rep["auc_s_heldout"] = float(((sF[:, None] > sR[None]).mean() + .5 * (sF[:, None] == sR[None]).mean())) if len(sF) and len(sR) else None
rep["gap"] = float(sF.mean() - sR.mean()) if len(sF) and len(sR) else None
rep["s_heldout_mean"] = dict(F=float(sF.mean()) if len(sF) else None, R=float(sR.mean()) if len(sR) else None)
enough = all(rep["counts"][l]["train"] >= 20 and rep["counts"][l]["heldout"] >= 20 for l in LABS)
rep["enough_demos"] = enough
rep["S2_pass"] = bool(enough and rep["auc_s_heldout"] is not None and rep["auc_s_heldout"] >= 0.9)


def rel(a, b):   # per-joint relative PSD difference
    return [float(np.linalg.norm(a[j] - b[j]) / (0.5 * (np.linalg.norm(a[j]) + np.linalg.norm(b[j])) + 1e-12)) for j in range(6)]


pF = [psd[("F", s)] for s in (0, 1, 2) if psd.get(("F", s)) is not None] if "F" in LABS else []
pR = [psd[("R", s)] for s in (0, 1, 2) if psd.get(("R", s)) is not None] if "R" in LABS else []
if pF and pR:
    rep["psd_rel_F_vs_R"] = rel(np.mean(pF, 0), np.mean(pR, 0))
    within = [rel(a, b) for P_ in (pF, pR) for x_, a in enumerate(P_) for b in P_[x_ + 1:]]
    rep["psd_rel_within_label_demonstrators"] = [float(np.mean([w[j] for w in within])) for j in range(6)] if within else None
np.savez(os.path.join(args.out, f"demos{args.tag}.npz"), **out)
json.dump(rep, open(os.path.join(args.out, f"s2{args.tag}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "demonstrators"}, indent=1))
