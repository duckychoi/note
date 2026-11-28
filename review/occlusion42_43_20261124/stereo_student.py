"""Auxiliary stereo targets for a student that predicts from a single RGB image."""
from collections import defaultdict
import hashlib
import json

import cv2
import numpy as np
import torch

from controlled_observations import RUN, free_gib
from stereo_controlled_observations import rectification, stereo_pair


class StereoTargets:
    def __init__(self, mode, stride, calibration=None):
        if stride not in (1, 2, 4):
            raise ValueError("Student stride must divide the calibrated native teacher stride of four")
        self.stride = stride
        self.mode = mode
        cv2.setNumThreads(1)
        source = (calibration or RUN/"stereo_results.json").read_bytes()
        result = json.loads(source)
        if not result["student_exploration"]:
            raise ValueError("Stereo observations did not pass the exploration gate")
        self.digest = hashlib.sha256(source).hexdigest()
        self.fixed = np.asarray(result["fixed_gain"])
        self.pixel_gain = np.asarray(result["pixel_gain"])
        self.lut = np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)

    def loss(self, samples, pose, lights, predicted):
        factor = 4//self.stride
        pair = []
        for rgb, depth, valid, k in samples:
            mat = k.copy()
            mat[:2] /= factor
            pair.append((rgb[::factor, ::factor], depth[::factor, ::factor], valid[::factor, ::factor], mat))
        geometry = rectification(pair, pose)
        base = predicted[0][-1]
        loss, used = base.new_zeros(()), 0
        count = sum(int(item[2][::2*factor, ::2*factor].sum()) for item in predicted)
        if geometry is None:
            return loss, used, count
        exposures = [(s[0], self.lut[s[0]]) for s in pair]
        matches = {}
        # Compare both lights at the opposite view on a shared support.
        for a,b in (lights, (lights[0],1-lights[1]), (1-lights[0],lights[1])):
            matches[a,b] = stereo_pair([exposures[0][a], exposures[1][b]], geometry)
        mode = int(lights[0] != lights[1])
        for i, (_, _, valid, k, q) in enumerate(predicted):
            z, ok, bins = matches[tuple(lights)][i]
            opposite = (lights[0],1-lights[1]) if i == 0 else (1-lights[0],lights[1])
            valid = valid[::2*factor, ::2*factor]
            mask = torch.as_tensor(ok & matches[opposite][i][1], device=q.device) & valid
            log_z = (k[0, 0]/(q.shape[1]*q[::2*factor, ::2*factor].clamp_min(1e-5))).log()
            gain = self.pixel_gain[mode][bins] if self.mode == "adaptive" else self.fixed[mode]
            step = torch.as_tensor(gain, device=q.device, dtype=q.dtype)*(
                torch.as_tensor(z, device=q.device, dtype=q.dtype).clamp_min(.001).log()-log_z.detach())
            target = log_z.detach()[mask]+step[mask]
            loss = loss+.5*(log_z[mask]-target).square().sum()
            used += int(mask.sum())
        # A missing stereo measurement leaves the supervised monocular target alone.
        return (loss/max(1, count) if self.mode != "off" else base.new_zeros(())), used, count


@torch.inference_mode()
def native_rgb_validation(model, transform, data, smoke):
    from depth_full_eval import depth_scores,metric_denominator
    model.eval()
    groups = defaultdict(list)
    floored_count=valid_count=zero_count=raw_nonfinite=0
    for i in range(min(8, len(data)) if smoke else len(data)):
        (rgb, depth, mask, _), _, _ = data[i]
        if mask.any():
            output=model.infer(transform(rgb))
            device=output["inverse_depth"].device
            valid=torch.as_tensor(mask,device=device)
            target=torch.as_tensor(depth[mask],device=device)
            if not bool(torch.isfinite(output["focallength_px"]) & (output["focallength_px"]>0)):
                raise ValueError(f"Invalid estimated focal length for {data.rows[i]}")
            inv=output["inverse_depth"][valid]
            zero_count+=int((inv==0).sum())
            if output.get("depth") is not None:
                raw_nonfinite+=int((~torch.isfinite(output["depth"][valid])).sum())
            denominator,floored=metric_denominator(output["inverse_depth"],valid,rgb.shape[1])
            groups[data.rows[i][0]].append(depth_scores(output["focallength_px"]/denominator,target))
            floored_count+=floored
            valid_count+=target.numel()
    if not groups:
        raise ValueError("No valid native RGB validation scenes")
    values=np.mean([np.mean(rows,axis=0) for rows in groups.values()],axis=0).tolist()
    return values,dict(development_floored_inverse_pixels=floored_count,development_valid_pixels=valid_count,
                       development_zero_inverse_pixels=zero_count,development_raw_nonfinite_metric_pixels=raw_nonfinite)


def memory_pressure():
    return free_gib() < 10 or torch.cuda.mem_get_info()[0] < 3*2**30
