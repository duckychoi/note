"""Decompose the old PF's errors and train a bounded correction to frozen D2."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from depth_full_data import NyuDepth
from depth_full_train import supervised
from light_encoder_score import summary
from light_encoder_study import CKPT, Model, atomic_save, sha, write, yield_memory
from run_stereo_students import gpu_slot
import sequential_pf42 as previous


OUT = previous.OUT.parent / "preserve42"
SEED = 421129
STRIDE = previous.STRIDE
ARMS = ("bounded_direct", "bounded_particle")
DATASETS = ("hammer", "diode", "ibims", "nyu")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


@torch.inference_mode()
def error_components(depth, truth, valid):
    residual = torch.where(valid, depth.log() - truth.clamp_min(1e-8).log(), 0.)
    error = residual[valid]
    scale = error.double().mean()
    shape = error.double() - scale
    result = dict(absrel=float(error.expm1().abs().mean()), scale=float(scale), scale2=float(scale.square()),
                  shape2=float(shape.square().mean()), log2=float(error.double().square().mean()))
    if abs(result["log2"] - result["scale2"] - result["shape2"]) > 1e-9:
        raise ValueError("Log squared-error decomposition failed")
    squared, n = 0., 0
    for err, z, ok in ((residual, truth, valid), (residual.T, truth.T, valid.T)):
        near, far = z[:, :-8], z[:, 8:]
        keep = ok[:, :-8] & ok[:, 8:] & ((near - far).abs() <= .05 * torch.minimum(near, far))
        delta = (err[:, 8:] - err[:, :-8])[::4, ::4][keep[::4, ::4]]
        squared += float(delta.double().square().sum())
        n += delta.numel()
    if not n:
        raise ValueError("No adjacent surface pairs")
    result["surface2"] = squared / n
    return result, shape


def compare_components(base, candidate, truth, valid):
    left, s0 = error_components(base, truth, valid)
    right, s1 = error_components(candidate, truth, valid)
    a10 = float((s0 + right["scale"]).expm1().abs().mean())
    a01 = float((s1 + left["scale"]).expm1().abs().mean())
    d_scale = .5 * (a10 - left["absrel"] + right["absrel"] - a01)
    d_shape = .5 * (a01 - left["absrel"] + right["absrel"] - a10)
    if abs(d_scale + d_shape - (right["absrel"] - left["absrel"])) > 1e-12:
        raise ValueError("Attribution identity failed")
    return {**{f"frozen_{k}": v for k, v in left.items()},
            **{f"old_{k}": v for k, v in right.items()}, "frozen": left["absrel"], "old": right["absrel"],
            "scale_from_old": a10, "shape_from_old": a01, "delta_scale": d_scale, "delta_shape": d_shape}


def native_mask(truth, valid):
    z = torch.as_tensor(truth, device="cuda")
    keep = torch.as_tensor(valid, device="cuda") & z.isfinite() & (z >= .25) & (z <= 30)
    return z, keep


@torch.inference_mode()
def diagnose():
    path = previous.OUT / "student.pt"
    state = torch.load(path, weights_only=True, map_location="cuda")
    model = previous.Predictor(False)
    model.arms["particle"].load_state_dict(state["weights"]["particle"])
    model.heads["particle"].load_state_dict(state["heads"]["particle"])
    lock = [sha(CKPT), sha(path)]
    for dataset in DATASETS:
        dst = OUT / f"diagnose-{dataset}.json"
        if dst.is_file() and read(dst)["lock"] == lock:
            continue
        rows = []
        original = read(previous.OUT / f"{dataset}.json")
        for (name, group, rgb, z, valid), old in zip(previous.frames(dataset), original["rows"], strict=True):
            yield_memory()
            if (name, group) != (old["key"], old["group"]):
                raise ValueError("Previous and current frame differ")
            image, feat, q = model.baseline(rgb)
            focal = model.net.fov.focal_length_px(image).float()
            base = model.full_depth(image, q, "frozen", focal)
            candidate = model.full_depth(image, model.q(feat, "particle"), "particle", focal)
            truth, mask = native_mask(z, valid)
            scores = compare_components(base, candidate, truth, mask)
            np.testing.assert_allclose([scores["frozen"], scores["old"]],
                                       [old["scores"]["frozen"], old["scores"]["particle"]], atol=2e-6)
            rows.append(dict(key=name, group=group, scores=scores))
            if len(rows) % 200 == 0:
                print("diagnose", dataset, len(rows), flush=True)
        result = summary(rows)
        write(dst, dict(**result, lock=lock, rows=rows))
        print("diagnose", dataset, result["means"], flush=True)


class Bounded(Model):
    def __init__(self, training):
        super().__init__(False)
        self.arms = {arm: deepcopy(self.encoder.blocks[self.start:]).requires_grad_(training) for arm in ARMS}
        head = nn.Sequential(nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 9)).cuda()
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        self.heads = {arm: deepcopy(head).requires_grad_(training) for arm in ARMS}
        levels = torch.tensor([-1., 0., 1.], device="cuda")
        self.grid = torch.cartesian_prod(levels, levels)
        self.transition = 1.75 * torch.cdist(self.grid, self.grid).square()

    def correction(self, image, inv, baseline, arm):
        _, logits, offsets = previous.Predictor.fields(self, image, inv, arm)
        _, priors = previous.scan(logits, self.transition, torch.zeros_like(logits), "exact")
        proposed = previous.join([(weight.exp()[:, None, None] * delta).sum(0)
                                  for weight, delta in zip(priors, offsets)])
        raw = baseline.log() - inv.log() + proposed
        mean = raw.mean()
        scale = .10 * (mean / .10).tanh()
        shape = .06 * ((raw - mean) / .10).tanh()
        return scale, shape - shape.mean(), logits, offsets

    @staticmethod
    def mixture(scale, shape):
        return .5 * (1 + (scale + shape).exp())

    def evaluations(self, image, q, feat, focal):
        base = focal / (q * q.shape[1])
        output = {"frozen": base}
        for arm in ARMS:
            inv = self.q(feat, arm)[::STRIDE, ::STRIDE]
            scale, shape, _, _ = self.correction(image, inv, q[::STRIDE, ::STRIDE], arm)
            shape = F.interpolate(shape[None, None], size=q.shape, mode="bilinear", align_corners=False)[0, 0]
            shape = (shape - shape.mean()).clamp(-.12, .12)
            output[arm] = base * self.mixture(scale, shape)
            if arm == "bounded_particle":
                output["scale_only"] = base * self.mixture(scale, 0.)
                multiplier = self.mixture(0., shape).log()
                output["shape_only"] = base * (multiplier - multiplier.mean()).exp()
        return output


def learn():
    manifest = read(previous.OUT / "manifest.json")
    signature = [sha(CKPT), sha(previous.OUT / "manifest.json"), sha(Path(__file__)),
                 sha(Path(__file__).with_name("PRESERVE42_PROTOCOL.md"))]
    nyu = NyuDepth(previous.NYU, "train", 1)
    torch.manual_seed(SEED)
    model = Bounded(True)
    rng = torch.Generator(device="cuda").manual_seed(SEED)
    params = {a: [*model.arms[a].parameters(), *model.heads[a].parameters()] for a in ARMS}
    opt = {a: torch.optim.AdamW([dict(params=model.arms[a].parameters(), lr=3e-6),
                                 dict(params=model.heads[a].parameters(), lr=5e-4)], weight_decay=.01) for a in ARMS}
    history = []
    checkpoint = OUT / "latest.train.pt"
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
        if signature != saved["signature"]:
            raise ValueError("Locked training inputs changed")
        history = saved["history"]
        rng.set_state(saved["rng"].cpu())
        for arm in ARMS:
            model.arms[arm].load_state_dict(saved["arms"][arm])
            model.heads[arm].load_state_dict(saved["heads"][arm])
            opt[arm].load_state_dict(saved["optimizers"][arm])
    order = manifest["order"]
    for step, (kind, index) in enumerate(order[len(history):], len(history)+1):
        yield_memory()
        rgb, z, valid, k = previous.read_training(kind, index, manifest["synthetic"], nyu)
        image, feat, q0 = model.baseline(rgb)
        q0 = q0[::STRIDE, ::STRIDE]
        valid = valid[::STRIDE, ::STRIDE]
        z = z[::STRIDE, ::STRIDE]
        ids = previous.observations(z, valid, index)
        count = sum(x.numel() for x in ids)
        depth = torch.as_tensor(np.where(valid, z, 1.), device="cuda")
        mask = torch.as_tensor(valid, device="cuda")
        focal = q0.new_tensor(float(k[0, 0]))
        record = dict(kind=kind, index=index, samples=count)
        for arm in ARMS:
            q1 = model.q(feat, arm)[::STRIDE, ::STRIDE]
            a, s, logits, offsets = model.correction(image, q1, q0, arm)
            q = q0 / model.mixture(a, s)
            fit = .25 * supervised(q, depth, mask, focal / STRIDE, True) if kind == "s" else 0.
            loss = q.sum() * 0 + fit
            if count:
                center = -q.log() + torch.log(focal / rgb.shape[1])
                squared = [((c.reshape(-1)[ix] - target.reshape(-1)[ix].log()) / previous.SIGMA).square().sum()
                           for c, target, ix in zip(previous.regions(center), previous.regions(depth), ids)]
                direct = .5 * torch.stack(squared).sum() / count
                if arm == "bounded_particle":
                    # Candidate states are offsets about the actual student prediction.
                    likelihood, _, _ = previous.observation_likelihood(-q.log(), offsets, depth, ids,
                                                                       focal, rgb.shape[1])
                    nll, _ = previous.scan(logits, model.transition, likelihood, "particle", rng)
                    loss = loss + .5 * direct + .25 * nll / count
                else:
                    loss = loss + .5 * direct
            opt[arm].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params[arm], 1., error_if_nonfinite=True)
            opt[arm].step()
            record[arm] = float(loss.detach())
        history.append(record)
        if step % 64 == 0 or step == len(order):
            payload = dict(signature=signature, arms={k: v.state_dict() for k, v in model.arms.items()},
                           heads={k: v.state_dict() for k, v in model.heads.items()})
            atomic_save(checkpoint, dict(**payload, history=history, rng=rng.get_state(),
                                         optimizers={k: v.state_dict() for k, v in opt.items()}))
            print("preserve training", step, len(order), record, flush=True)
    final = dict(signature=signature, arms={k: v.state_dict() for k, v in model.arms.items()},
                 heads={k: v.state_dict() for k, v in model.heads.items()})
    atomic_save(OUT / "bounded.pt", final)
    write(OUT / "training.json", dict(signature=signature, history=history))


@torch.inference_mode()
def assess():
    path = OUT / "bounded.pt"
    state = torch.load(path, weights_only=True, map_location="cuda")
    model = Bounded(False)
    for arm in ARMS:
        model.arms[arm].load_state_dict(state["arms"][arm])
        model.heads[arm].load_state_dict(state["heads"][arm])
    lock = [sha(CKPT), sha(path)]
    for dataset in DATASETS:
        dst = OUT / f"evaluate-{dataset}.json"
        if dst.is_file() and read(dst)["lock"] == lock:
            continue
        rows = []
        old_path = previous.OUT / f"dense-{dataset}.json"
        reference = read(old_path)
        for (name, group, rgb, z, valid), old in zip(previous.frames(dataset), reference["rows"], strict=True):
            yield_memory()
            if (name, group) != (old["key"], old["group"]):
                raise ValueError("Previous and current frame differ")
            image, feat, q = model.baseline(rgb)
            focal = model.net.fov.focal_length_px(image).float()
            preds = model.evaluations(image, q, feat, focal)
            truth, mask = native_mask(z, valid)
            scores = {key: float((value[mask] / truth[mask] - 1).abs().mean()) for key, value in preds.items()}
            np.testing.assert_allclose(scores["frozen"], old["scores"]["frozen"], atol=2e-6)
            scores.update(old_pf=old["scores"]["particle"], dense=old["scores"]["dense"], nyu_full=old["scores"]["nyu_full"])
            for key in ("frozen", "bounded_particle"):
                d, _ = error_components(preds[key], truth, mask)
                scores.update({f"{key}_{k}": d[k] for k in ("scale", "scale2", "shape2", "surface2")})
            if not np.isfinite(list(scores.values())).all():
                raise ValueError(name)
            rows.append(dict(key=name, group=group, scores=scores))
            if len(rows) % 200 == 0:
                print("preserve eval", dataset, len(rows), flush=True)
        result = summary(rows)
        comparison = {ref: previous.reduction(result["groups"], "bounded_particle", ref)
                      for ref in ("frozen", "old_pf", "bounded_direct", "dense", "nyu_full")}
        write(dst, dict(**result, lock=lock, reference=sha(old_path), comparison=comparison, rows=rows))
        print("preserve eval", dataset, result["means"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("diagnose", "train", "eval", "all"))
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    with gpu_slot():
        if stage in ("diagnose", "all"):
            diagnose()
        if stage in ("train", "all") and not (OUT / "bounded.pt").is_file():
            learn()
        if stage in ("eval", "all"):
            assess()
