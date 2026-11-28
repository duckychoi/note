"""Dense real-label comparison locked to the PF trial's image and update schedule."""
import json
from pathlib import Path

import numpy as np
import torch

from depth_full_data import NyuDepth
from depth_full_train import supervised
from light_encoder_score import summary
from light_encoder_study import Model, atomic_save, sha, write, yield_memory
from run_stereo_students import gpu_slot
from sequential_pf42 import CKPT, NYU, OUT, SEED, STRIDE, frames, read_training, reduction


def fit():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    signature = [sha(CKPT), sha(manifest_path), sha(Path(__file__))]
    model = Model(True)
    weights = model.arms["d2"]
    opt = torch.optim.AdamW(weights.parameters(), lr=3e-6, weight_decay=.01)
    data = NyuDepth(NYU, "train", 1)
    records = []
    checkpoint = OUT / "dense.latest.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True, map_location="cuda")
        if state["signature"] != signature:
            raise ValueError("Dense schedule changed")
        weights.load_state_dict(state["weights"])
        opt.load_state_dict(state["optimizer"])
        records = state["records"]
    for step, (kind, index) in enumerate(manifest["order"][len(records):], len(records)+1):
        yield_memory()
        rgb, depth, valid, k = read_training(kind, index, manifest["synthetic"], data)
        _, feat = model.input(rgb)
        mask = valid[::STRIDE, ::STRIDE]
        z = torch.as_tensor(np.where(mask, depth[::STRIDE, ::STRIDE], 1.), device="cuda")
        q = model.q(feat, "d2")[::STRIDE, ::STRIDE]
        loss = supervised(q, z, torch.as_tensor(mask, device="cuda"), float(k[0, 0]) / STRIDE, kind == "s")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weights.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        records.append(dict(kind=kind, index=index, valid=int(mask.sum()), loss=float(loss.detach())))
        if step % 128 == 0 or step == len(manifest["order"]):
            atomic_save(checkpoint, dict(weights=weights.state_dict(), optimizer=opt.state_dict(),
                                         signature=signature, records=records))
            print("dense train", step, "/", len(manifest["order"]), flush=True)
    atomic_save(OUT / "dense.pt", dict(weights=weights.state_dict(), signature=signature))
    write(OUT / "dense-training.json", dict(signature=signature, records=records))


@torch.inference_mode()
def evaluate():
    p = OUT / "dense.pt"
    model = Model(True)
    state = torch.load(p, weights_only=True, map_location="cuda")
    model.arms["d2"].load_state_dict(state["weights"])
    fingerprint = sha(p)
    for dataset in ("hammer", "diode", "ibims", "nyu"):
        out = OUT / f"dense-{dataset}.json"
        if out.exists() and json.loads(out.read_text(encoding="utf8"))["sha"] == fingerprint:
            continue
        original = json.loads((OUT / f"{dataset}.json").read_text(encoding="utf8"))
        rows = []
        for (name, group, image, z, valid), paired in zip(frames(dataset), original["rows"], strict=True):
            yield_memory()
            if (name, group) != (paired["key"], paired["group"]):
                raise ValueError("Different real test sample")
            x, feat, _ = model.baseline(image)
            q = model.q(feat, "d2")
            focal = model.net.fov.focal_length_px(x).float()
            depth = focal / (q.shape[1] * q)
            y = torch.as_tensor(z, device="cuda")
            keep = torch.as_tensor(valid, device="cuda") & y.isfinite() & (y >= .25) & (y <= 30)
            score = float((depth[keep] / y[keep] - 1).abs().mean())
            if not np.isfinite(score):
                raise ValueError(name)
            rows.append(dict(key=name, group=group, scores={**paired["scores"], "dense": score}))
            if len(rows) % 200 == 0:
                print("dense eval", dataset, len(rows), flush=True)
        result = summary(rows)
        write(out, dict(**result, rows=rows, sha=fingerprint,
                        comparison=reduction(result["groups"], "particle", "dense")))
        print("dense eval", dataset, result["means"]["dense"], result["means"]["particle"], flush=True)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    with gpu_slot():
        if not (OUT / "dense.pt").exists():
            fit()
        evaluate()
