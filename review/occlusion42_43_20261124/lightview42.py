"""Compare absolute and relative synthetic supervision at a real-measured scale anchor."""
import argparse
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from depth_full_eval import load_sample, sample_names
from light_encoder_score import summary
from light_encoder_study import CKPT, Model, atomic_save, sha, write, yield_memory
import parallax42 as old
from parallax42_data import EPS, FOCAL_BASELINE, OUT as PRIOR, REAL, grid_sample, read_rgb
from run_stereo_students import gpu_slot


OUT = PRIOR.parent / "lightview42"
SEED = 421131
ARMS = ("mixed_metric", "lightview")
DATASETS = ("hammer", "diode", "ibims", "diode_outdoor")
K0 = math.exp(-.20)
LOG_MIN, LOG_MAX = -.40, .20


def lock():
    return [sha(p) for p in (CKPT, old.previous.OUT / "manifest.json", PRIOR / "stereo.npz",
                            PRIOR / "student.pt", Path(__file__), Path(__file__).with_name("LIGHTVIEW42_PROTOCOL.md"))]


def exposure(rgb, index):
    """A geometry-preserving RGB perturbation of an existing photorealistic render."""
    rng = np.random.default_rng(SEED + index)
    h, w = rgb.shape[:2]
    gx = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(-1, 1, h, dtype=np.float32)[:, None, None]
    a, b = rng.uniform(-.25, .25, size=2)
    scale = np.exp(rng.uniform(-.50, .50) + a*gx + b*gy)
    tint = np.exp(rng.uniform(-.09, .09, size=3)).astype(np.float32)
    return np.rint((rgb / np.float32(255.0) * scale * tint).clip(0, 1)*255).astype(np.uint8)


def median_scale(depth, truth, low, high):
    r = truth/depth
    i = r.argsort()
    w = 1/r[i]
    return float(np.clip(r[i[np.searchsorted(w.cumsum(), .5*w.sum())]], low, high))


def worst_excess(k, candidates):
    return (abs(k*candidates-1) - abs(K0*candidates-1)).max(axis=-1)


def support(d, q):
    lo, hi = d*(q-EPS)/FOCAL_BASELINE, d*(q+EPS)/FOCAL_BASELINE
    return np.column_stack((lo, hi, np.clip(1/K0, lo, hi)))


def read_syn(index, manifest):
    return old.previous.read_training("s", index, manifest["synthetic"], None)[:3]


@torch.inference_mode()
def prepare():
    path = OUT / "teachers.json"
    binding = lock()
    if path.is_file() and old.read(path)["lock"] == binding:
        return old.read(path)
    model = old.Scale(False)
    initial = torch.load(PRIOR / "student.pt", map_location="cuda", weights_only=True)
    arm = "stereo_robust"
    model.arms[arm].load_state_dict(initial["arms"][arm])
    model.heads[arm].load_state_dict(initial["heads"][arm])
    manifest = old.read(old.previous.OUT / "manifest.json")
    stereo = old.observations()
    previous_real = {r["i"]: r for r in old.read(PRIOR / "real-teachers.json")["rows"]}
    real = {}
    fit = np.flatnonzero(~stereo["held"] & (stereo["n"] > 0))
    for i in fit:
        yield_memory()
        depth, features = model.metric(read_rgb(REAL / stereo["ref"][i]))
        y, x, q = old.unpack(stereo, i)
        d = depth[y, x].double().cpu().numpy()
        real[str(i)] = dict(candidates=support(d, q).tolist(), group=str(stereo["family"][i]),
                            old_target=previous_real[i]["robust"], old_pred=float(model.scale(features, arm).exp()))

    syn = []
    for i in range(len(manifest["synthetic"])):
        yield_memory()
        rgb, z, valid = read_syn(i, manifest)
        d1, features = model.metric(rgb)
        d2, _ = model.metric(exposure(rgb, i))
        y, x = grid_sample(valid & (z >= .25) & (z <= 30), old.SEED + i)
        old_target, one, two = 1., K0, K0
        relative = float((d1[::8, ::8].log()-d2[::8, ::8].log()).median())
        if len(x):
            base = d1[y, x].double().cpu().numpy()
            altered = d2[y, x].double().cpu().numpy()
            truth = z[y, x].astype(np.float64)
            old_target = old.point_multiplier(base, truth)
            one = median_scale(base, truth, math.exp(LOG_MIN), math.exp(LOG_MAX))
            two = median_scale(altered, truth, math.exp(LOG_MIN), math.exp(LOG_MAX))
            relative = float(np.median(np.log(base/altered)))
        syn.append(dict(old_target=old_target, old_pred=float(model.scale(features, arm).exp()) if i % 16 == 0 else None,
                        one=one, two=two, relative=relative, n=len(x)))
        if (i+1) % 250 == 0:
            print("lightview teacher prep", i+1, flush=True)

    groups = {}
    for row in real.values():
        groups.setdefault(row["group"], []).append(np.asarray(row["candidates"]))
    k_grid = np.exp(np.linspace(LOG_MIN, LOG_MAX, 601))
    fixed_surface = [np.mean([np.mean([worst_excess(k, c).mean() for c in rows])
                              for rows in groups.values()]) for k in k_grid]
    fixed = float(k_grid[np.argmin(fixed_surface)])

    # Reconstruct the original student objective, its shared log-scale slope at unity, and real family support.
    train = old.read(PRIOR / "training.json")["history"]
    original = {}
    for kind in ("s", "r"):
        targets = np.array([syn[r["synthetic"]]["old_target"] if kind == "s" else real[str(r["stereo"])]["old_target"]
                            for r in train if r["kind"] == kind])
        grad = np.clip(-np.log(targets)/.05, -1, 1)
        original[kind] = dict(n=len(targets), up=int((targets > 1).sum()), down=int((targets < 1).sum()),
                              median=float(np.median(targets)), mean_gradient=float(grad.mean()),
                              gradient_sum=float(grad.sum()))
    payload = dict(lock=binding, real=real, synthetic=syn, fixed=fixed, original=original)
    write(path, payload)
    print("lightview old objective", original, "anchored fit constant", fixed, flush=True)
    return payload


class Anchored(old.Scale):
    def scale_log(self, state, arm):
        tokens, _, _, rope, _, _ = state
        with self.cast(tokens.device, self.net.precision)():
            for i, layer in enumerate(self.arms[arm], self.start):
                tokens = self.block(i, layer, tokens, rope)
        embedding = tokens[:, self.encoder.num_prefix_tokens:].float().mean(1)
        value = self.heads[arm](embedding).reshape(())
        center, half = (LOG_MIN+LOG_MAX)/2, (LOG_MAX-LOG_MIN)/2
        bias = math.atanh((math.log(K0)-center)/half)
        return center + half*(value+bias).tanh()

    def __init__(self, training):
        super().__init__(training)
        template = self.arms["stereo_robust"]
        head = self.heads["stereo_robust"]
        from copy import deepcopy
        self.arms = {a: deepcopy(template) for a in ARMS}
        self.heads = {a: deepcopy(head) for a in ARMS}


def learn(teachers):
    torch.manual_seed(SEED)
    model = Anchored(True)
    manifest = old.read(old.previous.OUT / "manifest.json")
    stereo = old.observations()
    ordering = old.read(PRIOR / "training.json")["history"]
    save = OUT / "latest.train.pt"
    params = {a: [*model.arms[a].parameters(), *model.heads[a].parameters()] for a in ARMS}
    opt = {a: torch.optim.AdamW([dict(params=model.arms[a].parameters(), lr=3e-6),
                                 dict(params=model.heads[a].parameters(), lr=5e-4)], weight_decay=.01)
           for a in ARMS}
    history = []
    if save.is_file():
        checkpoint = torch.load(save, map_location="cuda", weights_only=True)
        if checkpoint["lock"] != teachers["lock"]:
            raise ValueError("The experiment lock changed")
        history = checkpoint["history"]
        for arm in ARMS:
            model.arms[arm].load_state_dict(checkpoint["arms"][arm])
            model.heads[arm].load_state_dict(checkpoint["heads"][arm])
            opt[arm].load_state_dict(checkpoint["optimizers"][arm])

    def weights():
        return dict(lock=teachers["lock"], arms={a: m.state_dict() for a, m in model.arms.items()},
                    heads={a: m.state_dict() for a, m in model.heads.items()})

    for step, sample in enumerate(ordering[len(history):], len(history)+1):
        yield_memory()
        kind = sample["kind"]
        index = sample["synthetic"] if kind == "s" else sample["stereo"]
        if kind == "s":
            rgb, _, _ = read_syn(index, manifest)
            _, one = model.input(rgb)
            _, two = model.input(exposure(rgb, index))
            teacher = teachers["synthetic"][index]
        else:
            _, one = model.input(read_rgb(REAL / stereo["ref"][index]))
            candidates = torch.tensor(teachers["real"][str(index)]["candidates"], device="cuda", dtype=torch.float32)
            fixed_risk = (K0*candidates-1).abs()
        row = dict(kind=kind, index=index)
        for arm in ARMS:
            a1 = model.scale_log(one, arm)
            opt[arm].zero_grad(set_to_none=True)
            if kind == "s":
                a2 = model.scale_log(two, arm)
                relative = F.smooth_l1_loss(a2-a1, a1.new_tensor(teacher["relative"]), beta=.05)
                targets = a1.new_tensor([math.log(teacher[k]) for k in ("one", "two")])
                metric = F.smooth_l1_loss(torch.stack((a1, a2)), targets, beta=.05)
                if not teacher["n"]:
                    metric = metric * 0
                loss = relative + metric if arm == "mixed_metric" else relative
                row[f"{arm}_relative"] = float(relative.detach())
                row[f"{arm}_metric"] = float(metric.detach()) if arm == "mixed_metric" else 0.
            else:
                loss = ((a1.exp()*candidates-1).abs() - fixed_risk).amax(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params[arm], 1., error_if_nonfinite=True)
            opt[arm].step()
            row[arm] = float(loss.detach())
            row[f"{arm}_scale"] = float(a1.detach().exp())
        history.append(row)
        if step % 64 == 0 or step == len(ordering):
            atomic_save(save, dict(**weights(), history=history, optimizers={a: o.state_dict() for a, o in opt.items()}))
            print("lightview", step, len(ordering), row, flush=True)
    atomic_save(OUT / "student.pt", weights())
    write(OUT / "training.json", dict(lock=teachers["lock"], history=history))


def frames(dataset):
    if dataset != "diode_outdoor":
        yield from old.previous.frames(dataset)
        return
    with ZipFile(PRIOR.parent / "DIODE.zip") as archive:
        for name in sample_names(archive, False):
            if "/val/outdoor/" in name:
                rgb, z, valid, _ = load_sample(archive, name, 1)
                yield name, name.split("/")[3], rgb, z, valid


@torch.inference_mode()
def evaluate():
    head_path = OUT / "student.pt"
    binding = [sha(CKPT), sha(head_path)]
    model = Anchored(False)
    payload = torch.load(head_path, map_location="cuda", weights_only=True)
    if payload["lock"] != lock():
        raise ValueError("The trained model has a different experiment lock")
    for arm in ARMS:
        model.arms[arm].load_state_dict(payload["arms"][arm])
        model.heads[arm].load_state_dict(payload["heads"][arm])
    new_fixed = old.read(OUT / "teachers.json")["fixed"]
    public_model, transform = None, None
    skipped = []
    for dataset in DATASETS:
        path = OUT / f"{dataset}.json"
        if path.is_file() and old.read(path)["checkpoint"] == binding:
            continue
        new_set = dataset == "diode_outdoor"
        if not new_set:
            previous = old.read(PRIOR / f"{dataset}.json")["rows"]
            original = old.read(PRIOR / f"official-{dataset}.json")["rows"]
        else:
            from depth_pro_2 import create_model_and_transforms
            public_model, transform = create_model_and_transforms("depthpro2-s", device="cuda")
        rows = []
        for i, (key, group, rgb, z, valid) in enumerate(frames(dataset)):
            yield_memory()
            base, features = model.metric(rgb)
            truth, mask = old.native_mask(z, valid)
            if not mask.any():
                if not new_set:
                    raise ValueError("Previously evaluated frame now has no valid pixels")
                skipped.append(key)
                continue
            t = base[mask]/truth[mask]
            scales = {"hypersim": 1., "fixed_0819": K0, "train_fixed": new_fixed,
                      **{arm: float(model.scale_log(features, arm).exp()) for arm in ARMS}}
            scores = {a: float((k*t-1).abs().mean()) for a, k in scales.items()}
            if new_set:
                pred = public_model.infer(transform(rgb).cuda())
                inv = pred["inverse_depth"].clamp_min(1e-5)
                d = pred["focallength_px"]/(rgb.shape[1]*inv)
                scores["apple_s"] = float((d[mask]/truth[mask]-1).abs().mean())
            else:
                if key != previous[i]["key"] or key != original[i]["key"]:
                    raise ValueError(f"Reference order mismatch: {key}")
                original_scores = previous[i]["scores"]
                np.testing.assert_allclose(scores["hypersim"], original_scores["frozen"], atol=2e-6)
                np.testing.assert_allclose(scores["fixed_0819"], original_scores["fixed_stereo"], atol=2e-6)
                scores.update(apple_s=original[i]["scores"]["public_s"],
                              previous_lightview=original_scores["stereo_robust"], nyu_full=original_scores["nyu_full"])
            rows.append(dict(key=key, group=group, scales=scales, scores=scores))
        result = summary(rows)
        comparisons = {a: {ref: old.previous.reduction(result["groups"], a, ref)
                           for ref in ("hypersim", "fixed_0819", "train_fixed", "apple_s", "mixed_metric") if ref != a}
                       for a in ("lightview", "mixed_metric", "train_fixed", "fixed_0819")}
        write(path, dict(**result, rows=rows, comparisons=comparisons, checkpoint=binding, skipped=skipped))
        print("lightview scores", dataset, result["means"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "train", "eval", "all"))
    stage = p.parse_args().stage
    OUT.mkdir(exist_ok=True)
    with gpu_slot():
        if stage in ("prepare", "train", "all"):
            teachers = prepare()
            if stage != "prepare" and not (OUT / "student.pt").is_file():
                learn(teachers)
        if stage in ("eval", "all"):
            evaluate()
