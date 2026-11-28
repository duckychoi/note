"""Train monocular shape corrections in Depth Pro 2-S with paired synthetic exposures."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from depth_full_data import camera_ray_lengths, h5_array, hypersim_intrinsics
from depth_full_train import supervised
from hypersim_fetch import frame_assets
from native_direction_data import CKPT, Native, OUT as PREVIOUS, manifest, read_rgb, sha, write


OUT = PREVIOUS.parent / "light-encoder-native"
ARMS = ("d2", "fixed", "conditional")
SEED = 4246
K_SCALE = 1.1129722251429508
DEVICE = "cuda"


def yield_memory():
    while psutil.virtual_memory().available < 8 * 2**30 or torch.cuda.mem_get_info()[0] < 4 * 2**30:
        time.sleep(4)


class Model:
    def __init__(self, trainable=False):
        native = Native()
        self.net, self.transform, self.cast = native.model, native.transform, native.cast
        self.net.requires_grad_(False)
        self.encoder = self.net.encoder
        self.start = len(self.encoder.blocks) - 4
        self.arms = {}
        if trainable:
            self.arms = {arm: copy.deepcopy(self.encoder.blocks[self.start:]).requires_grad_(True)
                         for arm in ARMS}
        self.check = None

    def block(self, i, layer, tokens, rope):
        if i in self.encoder.cross_frame_blocks:
            return self.encoder.cross_frame_attention(layer, tokens, rope)
        return layer(tokens, rope=rope)

    @torch.no_grad()
    def input(self, rgb):
        x = self.transform(rgb).unsqueeze(0).to(DEVICE)
        h, w = rgb.shape[:2]
        padded = F.pad(x, (0, -w % 32, 0, -h % 32), mode="replicate")
        grid = tuple(s // self.encoder.patch_size for s in padded.shape[-2:])
        with self.cast(x.device, self.net.precision)():
            rope = self.encoder.rope.get_embed(shape=grid)
            tokens = self.encoder.embed(padded)
            maps = []
            for i in range(self.start):
                tokens = self.block(i, self.encoder.blocks[i], tokens, rope)
                if i in self.encoder.reassembled_blocks:
                    maps.append(tokens[:, self.encoder.num_prefix_tokens:])
        return x, (tokens, maps, grid, rope, h, w)

    def q(self, state, arm=None):
        tokens, maps, grid, rope, h, w = state
        maps = list(maps)
        with self.cast(tokens.device, self.net.precision)():
            for i in range(self.start, len(self.encoder.blocks)):
                layer = self.encoder.blocks[i] if arm is None else self.arms[arm][i-self.start]
                tokens = self.block(i, layer, tokens, rope)
                if i in self.encoder.reassembled_blocks:
                    maps.append(tokens[:, self.encoder.num_prefix_tokens:])
        return self.net.decode_frame(maps, grid)[0, 0, :h, :w].float().clamp_min(1e-5)

    @torch.no_grad()
    def baseline(self, rgb):
        x, state = self.input(rgb)
        q = self.q(state)
        if self.check is None:
            exact = self.net.predict_inverse_depth(x)[0].clamp_min(1e-5)
            self.check = float((q - exact).abs().max())
            if self.check > 1e-6:
                raise ValueError(f"Split encoder output disagrees with the official path: {self.check}")
        return x, state, q


def load_scene(row):
    _, scene, cam, frame, _ = row
    rgb = read_rgb(scene, cam, frame)
    h, w = rgb.shape[:2]
    k = hypersim_intrinsics(scene, w, h)
    _, (_, depth) = frame_assets(scene, cam, frame)
    z = h5_array(depth) / camera_ray_lengths((h, w), tuple(k.ravel()))
    z = torch.tensor(z.astype(np.float32), device=DEVICE)
    valid = z.isfinite() & (z >= .25) & (z <= 30)
    return rgb, torch.where(valid, z, 1.), valid, k[0, 0]


def alternate(rgb, row):
    key = hashlib.sha256(repr(tuple(row[1:4])).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
    h, w = rgb.shape[:2]
    y, x = np.ogrid[:h, :w]
    radius = ((x/w-rng.uniform(.15,.85))**2+(y/h-rng.uniform(.15,.85))**2)
    factor = (.42 + .83*np.exp(-radius/.14)).astype(np.float32)
    image = np.rint(rgb.astype(np.float32)*factor[:, :, None]).clip(0,255).astype(np.uint8)
    return (image, rgb) if key[-1] % 2 else (rgb, image)


def differences(values):
    return torch.cat(((values[:, 1:]-values[:, :-1]).reshape(-1),
                      (values[1:]-values[:-1]).reshape(-1)))


def edge_means(values):
    return torch.cat(((.5*(values[:, 1:]+values[:, :-1])).reshape(-1),
                      (.5*(values[1:]+values[:-1])).reshape(-1)))


def edges(first, other, q, other_q, z, valid):
    def luma(rgb):
        return torch.tensor(rgb[::16, ::16].astype(np.float32).mean(-1)/255, device=DEVICE)
    light, alt = edge_means(luma(first)), edge_means(luma(other))
    bins = 2*(light < .28).long() + (alt > light).long()
    keep = edge_means(valid[::16, ::16].float()) == 1.
    b = -q[::16, ::16].log()
    d = differences(-other_q[::16, ::16].log() - b)
    r = differences(z[::16, ::16].log() - b)
    return b, d[keep], r[keep], bins[keep], keep


def sample(model, row):
    rgb, z, valid, focal = load_scene(row)
    first, other = alternate(rgb, row)
    x, state, q = model.baseline(first)
    _, _, other_q = model.baseline(other)
    edge = edges(first, other, q, other_q, z, valid)
    return state, q, z, valid, focal, edge


def moments(d, r, bins):
    return torch.stack([torch.bincount(bins, weights=values.double(), minlength=4)
                        for values in (torch.ones_like(r), r*d, d*d, r*r)]).cpu().numpy()


def signature(rows):
    return hashlib.sha256(repr(rows).encode()).hexdigest()


@torch.no_grad()
def calibrate():
    model = Model()
    rows = [r for r in manifest() if r[0] == "cal"]
    per_scene = {}
    for i, row in enumerate(rows):
        yield_memory()
        *_, (_, d, r, bins, _) = sample(model, row)
        per_scene.setdefault(row[1], np.zeros((4,4)))[:] += moments(d, r, bins)
        if (i+1) % 100 == 0:
            print("calibrate", i+1, len(rows), flush=True)
    total = np.sum(list(per_scene.values()), axis=0)
    gain = np.clip(np.divide(total[1], total[2], out=np.zeros(4), where=total[2]>0), 0, 1)
    fixed = float(np.clip(total[1].sum()/total[2].sum(), 0, 1))
    write(OUT/"calibration.json", dict(fixed=fixed, conditional=gain.tolist(), table=total.tolist(),
          scenes=len(per_scene), frames=len(rows), official_max_error=model.check,
          per_scene={key: value.tolist() for key,value in per_scene.items()}, ckpt=sha(CKPT),
          rows=signature(rows), code=sha(Path(__file__))))


def atomic_save(path, payload):
    tmp = path.with_suffix(".partial")
    torch.save(payload, tmp)
    tmp.replace(path)


def centered(q, student_q):
    delta = q.log() - student_q.log()
    return delta - delta.mean()


def training():
    rows = [r for r in manifest() if r[0]=="fit"]
    order = [rows[i] for i in np.random.default_rng(SEED).permutation(len(rows))]
    cal = json.loads((OUT/"calibration.json").read_text())
    calrows = [r for r in manifest() if r[0]=="cal"]
    if cal["ckpt"] != sha(CKPT) or cal["rows"] != signature(calrows) or cal["code"] != sha(Path(__file__)):
        raise ValueError("Calibration provenance changed")
    target = torch.tensor(cal["conditional"], device=DEVICE)
    model = Model(True)
    opts = {arm: torch.optim.AdamW(model.arms[arm].parameters(), lr=3e-6, weight_decay=.01)
            for arm in ARMS}
    mid = len(rows)//2
    path = OUT/"latest.train.pt"
    config = (signature(order), sha(OUT/"calibration.json"), sha(Path(__file__)))
    start, updates, losses = 0, 0, {a: [] for a in ARMS}
    if path.is_file():
        data = torch.load(path, weights_only=True, map_location=DEVICE)
        if data["config"] != config:
            raise ValueError("Resume configuration mismatch")
        start, updates = data["seen"], data["updates"]
        for arm in ARMS:
            model.arms[arm].load_state_dict(data["arms"][arm])
            opts[arm].load_state_dict(data["opts"][arm])
        losses = data["losses"]
    nparams = sum(p.numel() for p in model.arms["d2"].parameters())
    print("train", start, len(order), nparams, flush=True)
    for seen, row in enumerate(order[start:], start+1):
        yield_memory()
        state, q, z, valid, focal, (b, d, _, bins, keep) = sample(model, row)
        mask = valid[::4, ::4]
        if mask.any():
            for arm in ARMS:
                student_q = model.q(state, arm)
                h = centered(q, student_q)
                # Keep the same geometric mean at training and inference. Known synthetic
                # focal defines the supervised target, predicted focal is used at test.
                shaped_q = q[::4, ::4] * (-h[::4, ::4]).exp() / K_SCALE
                loss = supervised(shaped_q, z[::4, ::4], mask, focal/4, True)
                if arm != "d2" and d.numel():
                    correction = differences(h[::16, ::16])[keep]
                    weight = cal["fixed"] if arm=="fixed" else target[bins]
                    loss = loss + .5*(correction - weight*d).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError((row, arm, float(loss)))
                opts[arm].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.arms[arm].parameters(), 1., error_if_nonfinite=True)
                opts[arm].step()
                losses[arm].append(float(loss.detach()))
            updates += 1
        if seen % 100 == 0 or seen in (mid, len(rows)):
            weights = {a: model.arms[a].state_dict() for a in ARMS}
            atomic_save(path, dict(arms=weights, opts={a:o.state_dict() for a,o in opts.items()},
                        seen=seen, updates=updates, config=config, losses=losses))
            if seen in (mid, len(rows)):
                atomic_save(OUT/f"encoder-{seen}.pt", dict(arms=weights, seen=seen, updates=updates,
                            config=config, trainable=nparams, scenes=len(set(r[1] for r in order[:seen]))))
            print("trained", seen, updates, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibrate","train","both"))
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage in ("calibrate", "both"):
        calibrate()
    if stage in ("train", "both"):
        training()
