"""Run isolated native RGB inference with the same frozen Depth Pro 2-S checkpoint."""
import json
import time

import numpy as np
from PIL import Image
import torch

from light_encoder_study import CKPT, Model, sha, yield_memory
from material_light_assets import OUT, save_json


@torch.inference_mode()
def depth(model,rgb):
    yield_memory()
    result = model.net.infer(model.transform(rgb))
    z = result["focallength_px"]/(rgb.shape[1]*result["inverse_depth"].float().clamp_min(1e-5))
    if not torch.isfinite(z).all() or z.min()<=0:
        raise ValueError("Invalid monocular prediction")
    return z.cpu().numpy()


def predict_synthetic():
    model = Model()
    rows = json.loads((OUT/"scenes.json").read_text())
    for row in rows:
        folder = OUT/"renders"/row["scene"]
        while not (folder/"render.json").exists():
            time.sleep(2)
        path = folder/"predictions.npz"
        if path.exists():
            continue
        predictions = {}
        for material in ("matte","coat"):
            for light in range(3):
                key = f"{material}-{light}"
                rgb = np.asarray(Image.open(folder/(key+".png")).convert("RGB"))
                predictions[key] = depth(model,rgb)
        tmp = folder/"predictions.partial.npz"
        np.savez(tmp,**predictions)
        tmp.replace(path)
        print("depth",row["scene"],flush=True)
    save_json(OUT/"depth-run.json",dict(model_sha256=sha(CKPT),scene_count=len(rows),
                                        inputs="Each native RGB is processed separately with the model's own focal estimate."))


if __name__=="__main__":
    predict_synthetic()
