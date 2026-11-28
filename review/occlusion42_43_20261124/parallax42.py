"""Train and evaluate matched RGB-only Depth Pro 2-S metric scale adapters."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from light_encoder_score import summary
from light_encoder_study import CKPT, Model, atomic_save, sha, write, yield_memory
from preserve42 import native_mask
from run_stereo_students import gpu_slot
import sequential_pf42 as previous
from parallax42_data import (EPS, FOCAL_BASELINE, GRID, LIMIT, OUT, REAL, SEED, SOURCE_SCALE,
                             grid_sample, point_multiplier, read_rgb, robust_multiplier)


ARMS = ("synthetic_replay", "stereo_point", "stereo_robust")
DATASETS = ("hammer", "diode", "ibims")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


class Scale(Model):
    def __init__(self, training):
        super().__init__()
        self.arms = {arm: deepcopy(self.encoder.blocks[self.start:]).requires_grad_(training) for arm in ARMS}
        dim = self.encoder.blocks[-1].norm1.normalized_shape[0]
        head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1)).cuda()
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        self.heads = {arm: deepcopy(head).requires_grad_(training) for arm in ARMS}

    def scale(self, state, arm):
        tokens, _, _, rope, _, _ = state
        with self.cast(tokens.device, self.net.precision)():
            for i, layer in enumerate(self.arms[arm], self.start):
                tokens = self.block(i, layer, tokens, rope)
        embedding = tokens[:, self.encoder.num_prefix_tokens:].float().mean(1)
        return LIMIT * self.heads[arm](embedding).reshape(()).tanh()

    @torch.no_grad()
    def metric(self, rgb):
        x, state, q = self.baseline(rgb)
        focal = self.net.fov.focal_length_px(x).float()
        return focal / (q * q.shape[1]), state


def observations():
    with np.load(OUT / "stereo.npz", allow_pickle=False) as cache:
        return {k: cache[k] for k in cache.files}


def unpack(stereo, i):
    n = stereo["n"][i]
    y, x = stereo["y"][i, :n], stereo["x"][i, :n]
    q = stereo["disparity"][i, :n].astype(np.float64) + stereo["dmin"][i]
    return y, x, q


def signature():
    files = [CKPT, OUT / "stereo.npz", previous.OUT / "manifest.json", Path(__file__),
             Path(__file__).with_name("parallax42_data.py"), Path(__file__).with_name("PARALLAX42_PROTOCOL.md")]
    return [sha(p) for p in files]


def schedule(stereo, manifest):
    rng = np.random.default_rng(SEED)
    ids = np.flatnonzero(~stereo["held"] & (stereo["n"] > 0))
    n_real = sum(kind == "r" for kind, _ in manifest["order"])
    tiles = np.concatenate([rng.permutation(ids) for _ in range((n_real + len(ids)-1) // len(ids))])[:n_real]
    replay = rng.choice(len(manifest["synthetic"]), n_real, replace=False)
    assert len(manifest["order"]) == 2190 and n_real == 730
    return tiles, replay


@torch.inference_mode()
def real_targets(model, stereo):
    path = OUT / "real-teachers.json"
    if path.is_file() and read(path)["signature"] == signature():
        return read(path)
    rows = []
    for i, name in enumerate(stereo["ref"]):
        if stereo["held"][i] or not stereo["n"][i]:
            continue
        yield_memory()
        rgb = read_rgb(REAL / name)
        depth, _ = model.metric(rgb)
        y, x, q = unpack(stereo, i)
        d = depth[y, x].double().cpu().numpy()
        robust, bound = robust_multiplier(d, q)
        point = point_multiplier(d, FOCAL_BASELINE / q)
        risky = d * q / FOCAL_BASELINE
        rows.append(dict(i=i, group=str(stereo["family"][i]), name=str(name), n=len(q), point=point,
                         robust=robust, bound=bound,
                         risk=[float(abs(k*risky-1).mean()) for k in GRID]))
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row["risk"])
    risk = np.mean([np.mean(rs, axis=0) for rs in groups.values()], axis=0)
    obj = dict(signature=signature(), fixed=float(GRID[risk.argmin()]), rows=rows)
    write(path, obj)
    return obj


def synthetic_input(model, index, manifest):
    rgb, truth, valid, _ = previous.read_training("s", index, manifest["synthetic"], None)
    depth, state = model.metric(rgb)
    valid = valid & (truth >= .25) & (truth <= 30)
    y, x = grid_sample(valid, SEED + index)
    if not len(y):
        return state, 1., 0
    target = point_multiplier(depth[y, x].double().cpu().numpy(), truth[y, x].astype(np.float64))
    return state, target, len(y)


def learn():
    torch.manual_seed(SEED)
    model = Scale(True)
    stereo = observations()
    data = real_targets(model, stereo)
    teachers = {r["i"]: r for r in data["rows"]}
    manifest = read(previous.OUT / "manifest.json")
    tiles, replay = schedule(stereo, manifest)
    lock = signature()
    parameters = {a: [*model.arms[a].parameters(), *model.heads[a].parameters()] for a in ARMS}
    opt = {a: torch.optim.AdamW([dict(params=model.arms[a].parameters(), lr=3e-6),
                                 dict(params=model.heads[a].parameters(), lr=5e-4)], weight_decay=.01) for a in ARMS}
    history = []
    checkpoint = OUT / "latest.train.pt"
    if checkpoint.is_file():
        saved = torch.load(checkpoint, weights_only=True, map_location="cuda")
        if lock != saved["signature"]:
            raise ValueError("Training inputs have changed since the checkpoint")
        history = saved["history"]
        for arm in ARMS:
            model.arms[arm].load_state_dict(saved["arms"][arm])
            model.heads[arm].load_state_dict(saved["heads"][arm])
            opt[arm].load_state_dict(saved["optimizers"][arm])
    real_slot = sum(row["kind"] == "r" for row in history)
    order = manifest["order"]
    for step, (kind, index) in enumerate(order[len(history):], len(history)+1):
        yield_memory()
        record = dict(kind=kind, synthetic=index)
        if kind == "s":
            state, t, count = synthetic_input(model, index, manifest)
            inputs = {arm: (state, t) for arm in ARMS}
            record["synthetic_points"] = count
        else:
            replay_index = int(replay[real_slot])
            stereo_index = int(tiles[real_slot])
            real_slot += 1
            t = teachers[stereo_index]
            synthetic, target, count = synthetic_input(model, replay_index, manifest)
            _, state = model.input(read_rgb(REAL / stereo["ref"][stereo_index]))
            inputs = {ARMS[0]: (synthetic, target), ARMS[1]: (state, t["point"]), ARMS[2]: (state, t["robust"])}
            record.update(synthetic=replay_index, synthetic_points=count, stereo=stereo_index,
                          stereo_points=t["n"], robust_changed=t["bound"] > 0)
        for arm, (state, target) in inputs.items():
            pred = model.scale(state, arm)
            loss = F.smooth_l1_loss(pred, pred.new_tensor(np.log(target)), beta=.05)
            opt[arm].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters[arm], 1., error_if_nonfinite=True)
            opt[arm].step()
            record[arm] = float(loss.detach())
        history.append(record)
        if step % 64 == 0 or step == len(order):
            payload = dict(signature=lock, arms={a: m.state_dict() for a, m in model.arms.items()},
                           heads={a: m.state_dict() for a, m in model.heads.items()})
            atomic_save(checkpoint, dict(**payload, history=history,
                                         optimizers={a: o.state_dict() for a, o in opt.items()}))
            print("parallax training", step, len(order), record, flush=True)
    atomic_save(OUT / "student.pt", dict(signature=lock,
                                       arms={a: m.state_dict() for a, m in model.arms.items()},
                                       heads={a: m.state_dict() for a, m in model.heads.items()}))
    write(OUT / "training.json", dict(signature=lock, history=history))


@torch.inference_mode()
def audit():
    output = OUT / "held-stereo.json"
    if output.is_file() and read(output)["signature"] == signature():
        return
    model = Model()
    stereo = observations()
    rows = []
    for i in np.flatnonzero(stereo["held"] & (stereo["n"] > 0)):
        yield_memory()
        path = REAL / stereo["ref"][i]
        depth, _ = Scale.metric(model, read_rgb(path))
        truth_path = path.parents[2] / f"disp{path.stem[-1]}.png"
        disparity = cv2.imdecode(np.fromfile(truth_path, np.uint8), cv2.IMREAD_UNCHANGED)
        total = disparity.astype(np.float64) / SOURCE_SCALE + stereo["dmin"][i]
        truth = FOCAL_BASELINE / np.maximum(total, 1e-8)
        valid = (disparity > 0) & (truth >= .25) & (truth <= 30)
        y, x, q = unpack(stereo, i)
        d = depth[y, x].double().cpu().numpy()
        robust, lower = robust_multiplier(d, q)
        point = point_multiplier(d, FOCAL_BASELINE / q)
        use = valid[y, x]
        coverage = abs(total[y[use], x[use]] - q[use]) <= EPS
        truths = truth[y[use], x[use]]
        support = d[use] / truths
        full = depth.cpu().numpy()[valid] / truth[valid]
        oracle = point_multiplier(full, np.ones_like(full))
        def delta(t, k):
            return float((abs(t-1) - abs(k*t-1)).mean()) if t.size else None
        rows.append(dict(key=str(stereo["ref"][i]), group=str(stereo["family"][i]), n=int(len(x)),
                         valid=int(use.sum()), inside=int(coverage.sum()), area=float(stereo["area"][i]),
                         all_inside=bool(use.all() and coverage.all()), bound=lower,
                         robust=robust, point=point, full_oracle=oracle,
                         baseline=float(abs(full-1).mean()), support_robust=delta(support, robust),
                         full_robust=delta(full, robust), support_point=delta(support, point),
                         full_point=delta(full, point)))
    write(output, dict(signature=signature(), rows=rows))
    print("parallax held", len(rows), flush=True)


@torch.inference_mode()
def evaluate():
    path = OUT / "student.pt"
    model = Scale(False)
    state = torch.load(path, weights_only=True, map_location="cuda")
    if state["signature"] != signature():
        raise ValueError("Student and experiment inputs differ")
    for arm in ARMS:
        model.arms[arm].load_state_dict(state["arms"][arm])
        model.heads[arm].load_state_dict(state["heads"][arm])
    lock = [sha(CKPT), sha(path)]
    fixed = read(OUT / "real-teachers.json")["fixed"]
    for dataset in DATASETS:
        output = OUT / f"{dataset}.json"
        if output.is_file() and read(output)["lock"] == lock:
            continue
        reference = read(previous.OUT / f"dense-{dataset}.json")
        rows = []
        for (name, group, rgb, z, valid), old in zip(previous.frames(dataset), reference["rows"], strict=True):
            if (name, group) != (old["key"], old["group"]):
                raise ValueError("Baseline frame key changed")
            yield_memory()
            base, features = model.metric(rgb)
            truth, keep = native_mask(z, valid)
            t = base[keep] / truth[keep]
            scales = {"frozen": 1., "fixed_stereo": fixed, **{a: float(model.scale(features, a).exp()) for a in ARMS}}
            scores = {a: float((t*k-1).abs().mean()) for a, k in scales.items()}
            np.testing.assert_allclose(scores["frozen"], old["scores"]["frozen"], atol=2e-6)
            scores.update({k: old["scores"][k] for k in ("dense", "nyu_full")})
            oracle = point_multiplier(t.double().cpu().numpy(), np.ones(t.numel()))
            rows.append(dict(key=name, group=group, scores=scores, scales=scales, oracle=oracle,
                             oracle_absrel=float((t*oracle-1).abs().mean())))
            if len(rows) % 200 == 0:
                print("parallax eval", dataset, len(rows), flush=True)
        result = summary(rows)
        comparisons = {a: {b: previous.reduction(result["groups"], a, b)
                           for b in ("frozen", "fixed_stereo", "stereo_point", "nyu_full") if a != b}
                       for a in ARMS}
        write(output, dict(**result, lock=lock, rows=rows, comparisons=comparisons))
        print("parallax eval", dataset, result["means"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "train", "eval", "all"))
    stage = parser.parse_args().stage
    with gpu_slot():
        if stage in ("audit", "all"):
            audit()
        if stage in ("train", "all") and not (OUT / "student.pt").is_file():
            learn()
        if stage in ("eval", "all"):
            evaluate()
