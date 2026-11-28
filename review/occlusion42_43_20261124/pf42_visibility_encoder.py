"""Matched Depth Pro 2-S encoder adaptation after the held teacher comparison."""
import argparse
from collections import defaultdict
from functools import lru_cache
import copy
import json
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from light_encoder_study import Model, atomic_save, yield_memory
from lightview42 import frames
from pf42_crossview import comparison, entries, load
from pf42_crossview_data import DEPTH, FIT, OUT, scene
from pf42_visibility import consensus_map, predictions
from pf42_trials import DATASETS, SEEDS
import sequential_pf42 as synthetic
from run_stereo_students import capacity, gpu_slot


RUN = OUT / "visibility-encoder"
WIDTH, WARM, JOINT, MAX_POINTS = 384, 360, 300, 192
ARMS = ("direct", "shared")
SEED = 421351


def rescale(rgb, xy):
    h, w = rgb.shape[:2]
    height = round(h * WIDTH / w)
    rgb = cv2.resize(rgb, (WIDTH, height), interpolation=cv2.INTER_AREA)
    xy = np.rint((xy + .5) * [WIDTH / w, height / h] - .5).astype(int)
    return rgb, xy.clip([0, 0], [WIDTH - 1, height - 1])


def prepare():
    for name in FIT:
        files = list(entries("fit", [name]))
        cams = scene(name, with_truth=False)
        by_path = {str(c.path): c for c in cams}
        anchors = [by_path[str(load(p)["ref"])] for p in files]
        maps = {}
        for path in files:
            dest = RUN / path.name
            if dest.is_file():
                continue
            obs = load(path)
            sources = [by_path[str(c)] for c in obs["sources"]]
            for s in sources:
                if s not in maps:
                    maps[s] = consensus_map(s, cams, anchors)
            keep = (np.isfinite(obs["peaks"]).sum(1)>=3) & (np.isfinite(obs["cost"]).sum(1)>=2).any(1)
            use = {k: obs[k][keep] if k in ("peaks", "cost", "tally") else obs[k]
                   for k in ("peaks", "cost", "tally", "groups")}
            prob = predictions(use, by_path[str(obs["ref"])], sources, maps, posterior=True)
            direct = np.nanmedian(use["peaks"], axis=1)
            rgb, xy = rescale(obs["rgb"], use["tally"])
            temp = dest.with_suffix(".partial.npz")
            np.savez_compressed(temp, rgb=rgb, xy=xy, direct=np.log(direct),
                                probability=prob.astype(np.float32), group=obs["scene"])
            temp.replace(dest)
            print("real targets", path.stem, len(xy), flush=True)


def real_data():
    return [load(RUN / path.name) for path in entries("fit", FIT)]


def syn_rows():
    rows = json.loads((synthetic.OUT / "manifest.json").read_text(encoding="utf8"))["synthetic"]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[0]].append(i)
    rng = np.random.default_rng(SEED)
    chosen = np.array([rng.choice(ids) for _, ids in sorted(groups.items())])
    alternatives = {int(i): tuple(j for j in groups[rows[i][0]] if j!=i) for i in chosen}
    eligible = set(load(OUT.parent/"synthetic.npz")["group"])
    replacements = [int(i) for i in chosen if rows[i][0] in eligible]
    return rows, chosen, alternatives, replacements


@lru_cache(maxsize=370)
def syn_data(index):
    candidates = (index,*SYN_ALTERNATIVES[index])
    sample = None
    for alternative in candidates:
        rgb, z, valid, _ = synthetic.read_training("s", int(alternative), SYN, None)
        y, x = np.nonzero(valid & np.isfinite(z) & (z>=.35) & (z<=25))
        if len(x):
            break
        if sample is None:
            yl, xl = np.nonzero(valid & np.isfinite(z) & (z>=.25) & (z<=30))
            if len(xl):
                sample = rgb, z, yl, xl, alternative
    else:
        if sample is None:
            if index in SYN_REPLACEMENTS:
                raise ValueError(f"Previously scored Hypersim scene {SYN[index][0]} has no valid target")
            replacement = SYN_REPLACEMENTS[index % len(SYN_REPLACEMENTS)]
            print("Hypersim depth empty in entire scene",index,replacement,flush=True)
            return syn_data(replacement)
        rgb, z, y, x, alternative = sample
    if alternative != index:
        print("Hypersim replacement within the same scene", index, alternative, flush=True)
    use = np.random.default_rng(SEED+int(index)).choice(len(x), min(512, len(x)), replace=False)
    y, x = y[use], x[use]
    rgb, xy = rescale(rgb, np.column_stack((x, y)))
    return dict(rgb=rgb, xy=xy, direct=np.log(z[y,x]).astype(np.float32))


class Trainer(Model):
    def __init__(self):
        super().__init__()
        self.arms = {a: copy.deepcopy(self.encoder.blocks[self.start:]).requires_grad_(True) for a in ARMS}
        self.rope_by_grid = {}
        self.checkpoint = RUN

    @torch.no_grad()
    def encode(self, rgb):
        yield_memory()
        if self.check is None:
            image, state, _ = self.baseline(rgb)
        else:
            image, state = self.input(rgb)
        tokens, maps, grid, rope, h, w = state
        self.rope_by_grid[grid] = rope
        scale = float((self.net.fov.focal_length_px(image) / w).log().item())
        return tokens.cpu(), [m.cpu() for m in maps], grid, h, w, scale

    def depth(self, encoded, arm):
        tokens, maps, grid, h, w, scale = encoded
        state = tokens.cuda(), [m.cuda() for m in maps], grid, self.rope_by_grid[grid], h, w
        return scale - self.q(state, arm).log()


def train(model, real, pool, *, real_objective=None):
    grid_log = torch.from_numpy(np.log(DEPTH)).cuda()

    @lru_cache(maxsize=370)
    def saved(index):
        rows = real[-index-1] if index<0 else syn_data(index)
        return model.encode(rows["rgb"])

    def loss(index, row, points, arm, real_target=False):
        xy = row["xy"][points]
        x, y = torch.from_numpy(xy.T.copy()).cuda()
        depth = model.depth(saved(index), arm)[y, x]
        point_target = row.get("arm_targets", {}).get(arm, row["direct"]) if real_target else row["direct"]
        target = torch.from_numpy(point_target[points].astype(np.float32)).cuda()
        if real_target and real_objective is not None:
            probability = torch.from_numpy(row["probability"][points]).cuda()
            return real_objective(arm, depth, target, probability, grid_log)
        if real_target and arm=="shared":
            probability = torch.from_numpy(row["probability"][points]).cuda()
            diff = F.smooth_l1_loss(depth[:,None].expand_as(probability), grid_log[None].expand_as(probability),
                                    beta=.06, reduction="none")
            return (diff*probability).sum(1).mean()
        return F.smooth_l1_loss(depth, target, beta=.06)

    def point_ids(row, rng):
        return rng.choice(len(row["xy"]), min(MAX_POINTS, len(row["xy"])), replace=False)

    def update(opt, arm, obj):
        opt.zero_grad(set_to_none=True)
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.arms[arm].parameters(), 1., error_if_nonfinite=True)
        opt.step()

    nparams = sum(p.numel() for p in model.arms["direct"].parameters())
    print("trainable Depth Pro 2-S encoder parameters", nparams, flush=True)
    for seed in SEEDS:
        dest = RUN / f"seed-{seed}.pt"
        if dest.is_file():
            continue
        resume = RUN / f"seed-{seed}-latest.pt"
        rng = np.random.default_rng(seed)
        order = np.concatenate([rng.permutation(pool)[:WARM], rng.choice(pool, JOINT)])
        real_groups = [np.flatnonzero([str(r["group"])==g for r in real]) for g in FIT]
        opt = {a: torch.optim.AdamW(model.arms[a].parameters(), lr=3e-6, weight_decay=.01) for a in ARMS}
        start = 0
        if resume.is_file():
            state = torch.load(resume, weights_only=True, map_location="cpu")
            start = state["step"]
            for a in ARMS:
                model.arms[a].load_state_dict(state["weights"][a])
                opt[a].load_state_dict(state["optimizers"][a])
        else:
            base = model.encoder.blocks[model.start:].state_dict()
            for a in ARMS:
                model.arms[a].load_state_dict(base)
        for step, index in enumerate(order):
            index = int(index)
            warm = step < WARM
            rng_step = np.random.default_rng(seed*1000 + step)
            if step<start:
                continue
            s = syn_data(index)
            si = point_ids(s, rng_step)
            yield_memory()
            if warm:
                update(opt["direct"], "direct", loss(index, s, si, "direct"))
            else:
                j = int(rng_step.choice(real_groups[rng_step.integers(len(real_groups))]))
                r = real[j]
                ri = point_ids(r, rng_step)
                for a in ARMS:
                    update(opt[a], a, .5*(loss(index, s, si, a) + loss(-j-1, r, ri, a, True)))
            if step+1==WARM:
                for a in ARMS:
                    if a!="direct":
                        model.arms[a].load_state_dict(model.arms["direct"].state_dict())
                opt = {a: torch.optim.AdamW(model.arms[a].parameters(), lr=3e-6, weight_decay=.01) for a in ARMS}
            if (step+1)%30==0:
                weights = {a: m.state_dict() for a,m in model.arms.items()}
                atomic_save(resume,dict(step=step+1,weights=weights,optimizers={a:o.state_dict() for a,o in opt.items()}))
                print("encoder",seed,step+1,WARM+JOINT,flush=True)
        atomic_save(dest,dict(step=WARM+JOINT, weights={a:m.state_dict() for a,m in model.arms.items()}, trainable=nparams))


@torch.no_grad()
def evaluate(model):
    models = {}
    for seed in SEEDS:
        params = torch.load(RUN/f"seed-{seed}.pt",weights_only=True,map_location="cpu")
        for arm, weights in params["weights"].items():
            key = f"{seed}-{arm}"
            models[key] = copy.deepcopy(model.arms[arm]).requires_grad_(False)
            models[key].load_state_dict(weights)
    model.arms = models
    out = {}
    for dataset in DATASETS:
        old = load(OUT.parent / f"{dataset}.npz")
        chosen = set()
        for group in np.unique(old["group"]):
            keys = np.unique(old["key"][old["group"]==group])
            chosen.update(keys[np.linspace(0,len(keys)-1,min(8,len(keys)),dtype=int)])
        errors = defaultdict(list)
        for key, group, rgb, z, valid in frames(dataset):
            if key not in chosen:
                continue
            y, x = np.meshgrid(np.linspace(0,z.shape[0]-1,16,dtype=int),
                               np.linspace(0,z.shape[1]-1,24,dtype=int),indexing="ij")
            keep = valid[y,x] & np.isfinite(z[y,x]) & (z[y,x]>=.25) & (z[y,x]<=30)
            if not keep.any():
                continue
            y,x = y[keep],x[keep]
            rgb, xy = rescale(rgb,np.column_stack((x,y)))
            state = model.encode(rgb)
            x0,y0 = xy.T
            methods = [None,*models]
            errs = {a or "apple": float(abs(model.depth(state,a)[y0,x0].exp().cpu().numpy()/z[y,x]-1).mean()) for a in methods}
            errors[group].append(errs)
        if not errors:
            raise ValueError(f"No external RGB evaluation files: {dataset}")
        groups = {g:{a:float(np.mean([r[a] for r in rows])) for a in rows[0]} for g,rows in errors.items()}
        paired = {a:{g:float(np.mean([r[f"{s}-{a}"] for s in SEEDS])) for g,r in groups.items()} for a in ARMS}
        point = {a:float(np.mean(list(vals.values()))) for a,vals in paired.items()}
        point["apple"] = float(np.mean([r["apple"] for r in groups.values()]))
        out[dataset] = dict(absrel=point, relative=comparison(paired["shared"],paired["direct"]),
                            n=sum(map(len,errors.values())),groups=groups)
        print(dataset,out[dataset]["absrel"],out[dataset]["relative"],flush=True)
    (RUN/"results.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train"))
    args = parser.parse_args()
    gate = json.loads((OUT/"visibility-results.json").read_text(encoding="utf8"))
    if not gate["encoder_gate"] or len(gate["rows"])!=9:
        raise ValueError("Complete the registered teacher gate on the full holdout first")
    RUN.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    if args.stage=="prepare":
        prepare()
    else:
        SYN, INDICES, SYN_ALTERNATIVES, SYN_REPLACEMENTS = syn_rows()
        with gpu_slot():
            while not capacity():
                time.sleep(5)
            model = Trainer()
            train(model, real_data(), INDICES)
            evaluate(model)
