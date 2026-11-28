"""Same-backbone synthetic and sparse-real training for a spatial particle filter."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from depth_full_data import Hypersim, NyuDepth
from depth_full_train import supervised
from encoder_expand_data import selection_for
from light_encoder_score import images, summary
from light_encoder_study import CKPT, Model, atomic_save, sha, write, yield_memory
from native_direction_eval import dataframes
from run_stereo_students import gpu_slot


OUT = CKPT.parents[3] / "sequential-pf42"
REAL_REFERENCE = OUT.parent / "full-comparison-4243/real/best.pt"
NYU = OUT.parent / "nyu-depthv2-wds"
ARMS = ("synthetic", "direct", "static", "exact", "particle")
SEED = 421127
STRIDE = 4
REAL_COUNT = 730
PARTICLES = 64
SIGMA = .20


def regions(x):
    h, w = x.shape[-2:]
    return (x[..., :h//2, :w//2], x[..., :h//2, w//2:],
            x[..., h//2:, w//2:], x[..., h//2:, :w//2])


def join(xs):
    return torch.cat((torch.cat(xs[:2], dim=-1), torch.cat((xs[3], xs[2]), dim=-1)), dim=-2)


def scan(logits, transition, log_likelihood, arm, generator=None, count=PARTICLES):
    """Use observation-free calls separately from the teacher's filtered calls."""
    prior = logits.mean(0).log_softmax(0)
    total, priors = prior.new_zeros(()), []
    for t in range(len(logits)):
        if t and arm not in ("static", "direct"):
            log_k = (logits[t][None, :] - transition).log_softmax(1)
            if arm == "particle":
                ids = torch.multinomial(posterior.detach().exp(), count, replacement=True, generator=generator)
                prior = torch.logsumexp(log_k[ids], dim=0) - float(np.log(count))
            else:
                prior = torch.logsumexp(posterior[:, None] + log_k, dim=0)
        elif t:
            prior = posterior
        priors.append(prior)
        unnormalized = prior + log_likelihood[t]
        evidence = torch.logsumexp(unnormalized, dim=0)
        posterior = unnormalized - evidence
        total = total - evidence
    return total, torch.stack(priors)


def verify():
    torch.manual_seed(SEED)
    a = torch.randn(2, 3, dtype=torch.float64)
    transition = torch.Tensor([[0, 1, 3], [1, 0, 1], [3, 1, 0]]).double()
    g = torch.tensor([[.98, .01, .01], [.01, .01, .98]], dtype=a.dtype)
    w = a.mean(0).softmax(0)
    kernel = (a[1][None, :] - transition).softmax(1)
    fixed = (w * g.prod(0)).sum()
    exact = ((w * g[0]) @ kernel * g[1]).sum()
    np.testing.assert_allclose(scan(a, transition, g.log(), "static")[0], -fixed.log(), atol=1e-12)
    np.testing.assert_allclose(scan(a, transition, g.log(), "exact")[0], -exact.log(), atol=1e-12)
    estimate = scan(a, transition, g.log(), "particle", count=50000)[0]
    np.testing.assert_allclose(estimate, -exact.log(), atol=.025)
    np.testing.assert_allclose(scan(a, transition, g.flip(0).log(), "static")[0], -fixed.log())
    assert abs(float(exact - fixed)) > .001
    write(OUT / "numerical.json", dict(static=float(fixed), exact=float(exact), particle=float((-estimate).exp())))


class Predictor(Model):
    def __init__(self, training):
        super().__init__(False)
        from copy import deepcopy
        self.arms = {arm: deepcopy(self.encoder.blocks[self.start:]).requires_grad_(training)
                     for arm in ARMS}
        self.heads = {arm: nn.Sequential(nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 9)).cuda()
                      for arm in ARMS if arm != "synthetic"}
        initial = next(iter(self.heads.values())).state_dict()
        for head in self.heads.values():
            head.load_state_dict(initial)
            head.requires_grad_(training)
        levels = torch.tensor([-1., 0., 1.], device="cuda")
        self.grid = torch.cartesian_prod(levels, levels)
        self.transition = 1.75 * torch.cdist(self.grid, self.grid).square()

    def fields(self, image, inverse, arm):
        q = inverse.clamp_min(1e-5)
        logz = -q.log()
        pooled = image[0, :, ::STRIDE, ::STRIDE]
        columns, offsets = [], []
        for tile, color in zip(regions(logz), regions(pooled)):
            location = tile.mean()
            centered = (tile - location).tanh()
            columns.append(torch.cat((color.mean((1, 2)), color.std((1, 2), unbiased=False),
                                      torch.stack((location, tile.std(unbiased=False),
                                                   tile.amin(), tile.amax())))))
            offsets.append(.22 * self.grid[:, 0, None, None] + .16 * self.grid[:, 1, None, None] * centered)
        inputs = torch.stack(columns).float()
        return logz, self.heads[arm](inputs), offsets

    def full_depth(self, image, inverse, name, focal):
        if name in ("synthetic", "frozen"):
            return focal / (inverse * inverse.shape[1])
        _, logits, offsets = self.fields(image, inverse[::STRIDE, ::STRIDE], name)
        _, prior = scan(logits, self.transition, torch.zeros_like(logits),
                        "exact" if name == "particle" else name)
        tiles = []
        for log_weights, delta, qi in zip(prior, offsets, regions(inverse)):
            delta = F.interpolate(delta[None], size=qi.shape, mode="bilinear", align_corners=False)[0]
            depths = (focal / (inverse.shape[1] * qi))[None] * delta.exp()
            values, ids = depths.sort(dim=0)
            weights = log_weights.exp()[ids] / values
            middle = (weights.cumsum(0) < .5 * weights.sum(0)).sum(0).clamp_max(len(log_weights)-1)
            tiles.append(values.gather(0, middle[None])[0])
        return join(tiles)


def training_manifest():
    selection = selection_for("train_full")
    rng = np.random.default_rng(SEED)
    synthetic = []
    for name, captures in sorted(selection.items()):
        order = rng.choice(len(captures), size=min(4, len(captures)), replace=False)
        synthetic.extend((name, captures[i][0], int(captures[i][1])) for i in order)
    if len({row[0] for row in synthetic}) != 365:
        raise ValueError("Expected all 365 official Hypersim training scenes")
    nyu = NyuDepth(NYU, "train", 1)
    real = rng.choice(len(nyu), REAL_COUNT, replace=False).tolist()
    combination = [("s", i) for i in range(len(synthetic))] + [("r", i) for i in real]
    order = [combination[i] for i in rng.permutation(len(combination))]
    return synthetic, nyu, order


def read_training(kind, index, synthetic, nyu):
    if kind == "r":
        return nyu[index][0]
    scene, cam, frame = synthetic[index]
    # Reuse the sample helper without indexing 59,543 files again.
    obj = Hypersim.__new__(Hypersim)
    obj.stride = 1
    return obj.read_frame(scene, cam, frame)


def observations(z, valid, index):
    selected = []
    rng = np.random.default_rng(SEED + index)
    for patch in regions(valid):
        axis = np.flatnonzero(patch.reshape(-1))
        selected.append(rng.choice(axis, min(8, len(axis)), replace=False))
    return [torch.as_tensor(x, device="cuda", dtype=torch.long) for x in selected]


def observation_likelihood(logz, offsets, depth, selected, focal, width):
    # Shapes are K by number of selected independent-depth pixels, per spatial stage.
    baseline = logz + torch.log(focal / width)
    likelihood, targets, pred = [], [], []
    for patch, delta, z, ids in zip(regions(baseline), offsets, regions(depth), selected):
        observed = z.reshape(-1)[ids].log()
        candidate = patch.reshape(-1)[ids][None] + delta.flatten(1)[:, ids]
        likelihood.append(-.5 * ((candidate - observed[None]) / SIGMA).square().sum(1))
        targets.append(observed)
        pred.append(candidate)
    return torch.stack(likelihood), targets, pred


def train():
    synthetic, nyu, order = training_manifest()
    signature = [sha(CKPT), sha(Path(__file__)), sha(Path(__file__).with_name("SEQUENTIAL_PF42_PROTOCOL.md")),
                 SEED, len(synthetic), REAL_COUNT, PARTICLES]
    write(OUT / "manifest.json", dict(synthetic=synthetic, order=order, signature=signature))
    torch.manual_seed(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    model = Predictor(True)
    pars = {arm: list(model.arms[arm].parameters()) +
            (list(model.heads[arm].parameters()) if arm in model.heads else []) for arm in ARMS}
    optimizers = {arm: torch.optim.AdamW(params, lr=3e-6, weight_decay=.01) for arm, params in pars.items()}
    # The lightweight probability heads need their own learning-rate group.
    for arm, head in model.heads.items():
        optimizers[arm] = torch.optim.AdamW([dict(params=model.arms[arm].parameters(), lr=3e-6),
                                            dict(params=head.parameters(), lr=5e-4)], weight_decay=.01)
    latest = OUT / "latest.train.pt"
    history, start = [], 0
    if latest.exists():
        saved = torch.load(latest, weights_only=True, map_location="cuda")
        if saved["signature"] != signature or saved["order"] != order:
            raise ValueError("Checkpoint schedule differs")
        start, history = saved["step"], saved["history"]
        generator.set_state(saved["random"].cpu())
        for arm in ARMS:
            model.arms[arm].load_state_dict(saved["weights"][arm])
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
        for arm in model.heads:
            model.heads[arm].load_state_dict(saved["heads"][arm])
    for step, (kind, index) in enumerate(order[start:], start + 1):
        yield_memory()
        rgb, z, valid, k = read_training(kind, index, synthetic, nyu)
        image, state = model.input(rgb)
        valid = valid[::STRIDE, ::STRIDE]
        ids = observations(z[::STRIDE, ::STRIDE], valid, index)
        count = sum(a.numel() for a in ids)
        z = torch.as_tensor(np.where(valid, z[::STRIDE, ::STRIDE], 1.), device="cuda")
        mask = torch.as_tensor(valid, device="cuda")
        focal = torch.tensor(float(k[0, 0]), device="cuda")
        log = dict(step=step, kind=kind, index=index, samples=count)
        for arm in ARMS:
            if kind == "r" and (arm == "synthetic" or not count):
                continue
            inv = model.q(state, arm)[::STRIDE, ::STRIDE]
            if kind == "s":
                dense = supervised(inv, z, mask, focal / STRIDE, True)
            else:
                dense = inv.new_zeros(())
            loss = dense
            if arm != "synthetic" and count:
                baseline, a, offsets = model.fields(image, inv, arm)
                likelihood, targets, predictions = observation_likelihood(baseline, offsets, z, ids, focal, rgb.shape[1])
                if arm == "direct":
                    weights = a.mean(0).softmax(0)
                    terms = [(estimate * weights[:, None]).sum(0) - y for estimate, y in zip(predictions, targets)]
                    likelihood_loss = .5 * torch.cat(terms).square().mean() / SIGMA**2
                else:
                    likelihood_loss, _ = scan(a, model.transition, likelihood, arm, generator)
                    likelihood_loss = likelihood_loss / count
                loss = dense + .5 * likelihood_loss
                log[arm + "_likelihood"] = float(likelihood_loss.detach())
            optimizers[arm].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pars[arm], 1., error_if_nonfinite=True)
            optimizers[arm].step()
            log[arm] = float(loss.detach())
        history.append(log)
        if step in (1, len(order)) or step % 64 == 0:
            record = dict(step=step, signature=signature,
                          weights={a: v.state_dict() for a, v in model.arms.items()},
                          heads={a: v.state_dict() for a, v in model.heads.items()})
            atomic_save(latest, dict(**record, random=generator.get_state(), order=order, history=history,
                                     optimizers={a: v.state_dict() for a, v in optimizers.items()}))
            print("train", step, "/", len(order), log, flush=True)
    final = dict(step=len(order), signature=signature,
                 weights={a: v.state_dict() for a, v in model.arms.items()},
                 heads={a: v.state_dict() for a, v in model.heads.items()})
    atomic_save(OUT / "student.pt", final)
    write(OUT / "training.json", dict(signature=signature, history=history))


def frames(dataset):
    if dataset == "hammer":
        yield from dataframes(dataset)
    elif dataset == "nyu":
        data = NyuDepth(NYU, "test", 1)
        for i in range(len(data)):
            rgb, z, valid, _ = data[i][0]
            yield str(i), str(i), rgb, z, valid
    else:
        yield from images(dataset)


def reduction(groups, arm, reference):
    base = np.array([v[reference] for v in groups.values()])
    result = np.array([v[arm] for v in groups.values()])
    choice = np.random.default_rng(SEED).integers(len(base), size=(10000, len(base)))
    diff = 100 * (1 - result[choice].mean(1) / base[choice].mean(1))
    return dict(relative=float(100 * (1 - result.mean() / base.mean())),
                interval=np.quantile(diff, [.025, .975]).tolist())


@torch.inference_mode()
def evaluate():
    path = OUT / "student.pt"
    data = torch.load(path, weights_only=True, map_location="cuda")
    models = Predictor(False)
    full_real = Model()
    missing = full_real.net.load_state_dict(torch.load(REAL_REFERENCE, weights_only=True, map_location="cpu"), strict=False)
    if missing.unexpected_keys or any(not k.startswith("fov.") for k in missing.missing_keys):
        raise ValueError(missing)
    for arm in ARMS:
        models.arms[arm].load_state_dict(data["weights"][arm])
    for arm in models.heads:
        models.heads[arm].load_state_dict(data["heads"][arm])
    locked = [sha(path), sha(REAL_REFERENCE), sha(CKPT)]
    for dataset in ("hammer", "diode", "ibims", "nyu"):
        result = OUT / f"{dataset}.json"
        if result.exists() and json.loads(result.read_text(encoding="utf8"))["sha"] == locked:
            continue
        rows = []
        for name, group, image, z, valid in frames(dataset):
            yield_memory()
            inp, features, q = models.baseline(image)
            focal = models.net.fov.focal_length_px(inp).float()
            y = torch.as_tensor(z, device="cuda")
            mask = torch.as_tensor(valid, device="cuda") & y.isfinite() & (y >= .25) & (y <= 30)
            if not mask.any():
                raise ValueError(name)
            scores = {}
            for arm in ("frozen", *ARMS, "nyu_full"):
                if arm == "nyu_full":
                    real_q = full_real.net.predict_inverse_depth(inp)[0].float().clamp_min(1e-5)
                    estimate = focal / (real_q * real_q.shape[1])
                else:
                    inverse = q if arm == "frozen" else models.q(features, arm)
                    estimate = models.full_depth(inp, inverse, arm, focal)
                scores[arm] = float((estimate[mask] / y[mask] - 1).abs().mean())
            if not all(np.isfinite(x) for x in scores.values()):
                raise ValueError(name)
            rows.append(dict(key=name, group=group, scores=scores))
            if len(rows) % 100 == 0:
                print("eval", dataset, len(rows), flush=True)
        totals = summary(rows)
        write(result, dict(**totals, rows=rows, sha=locked, primary="particle",
                           compare={k: reduction(totals["groups"], "particle", k)
                                    for k in ("frozen", "synthetic", "direct", "static", "exact", "nyu_full")}))
        print("eval", dataset, len(rows), totals["means"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "train", "eval", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("verify", "all"):
        verify()
    if args.stage != "verify":
        with gpu_slot():
            if args.stage in ("train", "all"):
                train()
            if args.stage in ("eval", "all"):
                evaluate()
