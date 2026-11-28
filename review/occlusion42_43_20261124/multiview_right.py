"""Right-view predictions for the existing rectified Middlebury evaluation."""
from dataclasses import replace
import sys

import numpy as np

from common import ROOT, STORE, TEST


OUTPUT = STORE / "multiview_right"


def prepare():
    import cv2

    OUTPUT.mkdir(exist_ok=True)
    for scene in TEST:
        raw = STORE / "raw" / scene
        bgr = cv2.imdecode(np.fromfile(raw/"im1.png", np.uint8), cv2.IMREAD_COLOR)
        with np.load(STORE/f"{scene}.npz") as f:
            h, w = f["rgb"].shape[:2]
        calib = dict(line.split("=", 1) for line in (raw/"calib.txt").read_text().splitlines())
        right = np.fromstring(calib["cam1"].strip("[]").replace(";", " "), sep=" ").reshape(3,3)
        left = np.fromstring(calib["cam0"].strip("[]").replace(";", " "), sep=" ").reshape(3,3)
        scale = np.diag([w/bgr.shape[1], h/bgr.shape[0], 1])
        pixel = cv2.cvtColor(cv2.resize(bgr,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        np.savez_compressed(OUTPUT/f"{scene}.npz", rgb=pixel, right=scale@right, left=scale@left,
                            baseline=float(calib["baseline"])/1000)


def infer():
    print("Loading PyTorch", flush=True)
    import torch

    sys.path.insert(0, str(ROOT.parents[2]/"research-sources"/"ml-depth-pro-2"/"src"))
    from depth_pro_2 import PRESETS, create_model_and_transforms

    torch.set_num_threads(2)
    preset = replace(PRESETS["depthpro2-s"], fov_checkpoint_uri=None)
    model, transform = create_model_and_transforms(preset, device="cuda")
    depth = {}
    with torch.inference_mode():
        for scene in TEST:
            with np.load(OUTPUT/f"{scene}.npz") as f:
                d = model.infer(transform(f["rgb"]), f_px=float(f["right"][0,0]))["depth"]
            depth[scene] = d.float().cpu().numpy()
            print("Inferred",scene,flush=True)
    np.savez_compressed(OUTPUT/"predictions.npz", **depth)


if __name__ == "__main__":
    {"prepare":prepare,"infer":infer}[sys.argv[1]]()
