"""Compare an independent, training-time stereo observation under physical lighting changes."""
import ctypes

import cv2
import numpy as np

from controlled_observations import RUN, real_inputs, wait_for_ram, write_json
from depth_full_data import Hypersim
from score_controlled_observations import fit, scene_rows, score


CONDITIONS = ("view", "both")


def rectification(pair, pose):
    height, width = pair[0][1].shape
    size = (width, height)
    cameras = [item[3].astype(np.float64) for item in pair]
    geometry, _ = rectified_cameras(cameras, size, pose)
    if geometry is None:
        return None
    r0, r1, p0, p1, axis = geometry
    return rectification_maps(cameras, size, r0, r1, p0, p1, axis)


def rectified_cameras(cameras, size, pose):
    if pose is None:
        return None, "missing_pose"
    if np.linalg.norm(pose[:, 3]) < 1e-5:
        return None, "zero_baseline"
    r0, r1, p0, p1, _, _, _ = cv2.stereoRectify(cameras[0], None, cameras[1], None, size,
        pose[:, :3].astype(np.float64), pose[:, 3:].astype(np.float64), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    axis = int(np.argmax(np.abs(p1[:2, 3])))
    if not np.isfinite(p1).all() or not .25 < p1[axis, axis]/cameras[0][0, 0] < 4:
        return None, "rectified_focal"
    return (r0, r1, p0, p1, axis), "ok"


def rectification_maps(cameras, size, r0, r1, p0, p1, axis):
    width, height = size
    y, x = np.indices((height, width), dtype=np.float32)[:, ::2, ::2]
    grid = np.stack([x, y], -1)
    pixels = np.stack([x, y, np.ones_like(x)], -1)
    maps = []
    for k, r, p in zip(cameras, (r0, r1), (p0, p1)):
        mx, my = cv2.initUndistortRectifyMap(k, None, r, p[:, :3], size, cv2.CV_32FC1)
        xy = cv2.undistortPoints(grid.reshape(-1, 1, 2), k, None, R=r, P=p[:, :3]).reshape(grid.shape)
        z_factor = (pixels @ np.linalg.inv(k).T @ r.T)[..., 2]
        maps.append((mx, my, xy, z_factor))
    return axis, p1[axis, 3], maps


def stereo_pair(images, geometry, trace=None):
    axis, translation, maps = geometry
    gray, masks = [], []
    for rgb, (mx, my, _, _) in zip(images, maps):
        h, w = rgb.shape[:2]
        img = cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mx, my, cv2.INTER_LINEAR)
        valid = (mx >= 1) & (mx < w-2) & (my >= 1) & (my < h-2)
        if axis == 1:
            img, valid = img.T, valid.T
        if translation > 0:
            img, valid = np.fliplr(img), np.fliplr(valid)
        gray.append(np.ascontiguousarray(img))
        masks.append(valid)
    h, w = gray[0].shape
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=min(96, (w//2//16)*16),
        blockSize=3, P1=72, P2=288, uniquenessRatio=8, speckleWindowSize=0)
    disparities = [matcher.compute(gray[0], gray[1]).astype(np.float32)/16,
        np.fliplr(matcher.compute(np.fliplr(gray[1]).copy(), np.fliplr(gray[0]).copy())).astype(np.float32)/16]
    yy, xx = np.indices((h, w))
    observations = []
    for i, sign in ((0, -1), (1, 1)):
        d = disparities[i]
        other_x = np.rint(xx+sign*d).astype(int)
        lookup_x = other_x.clip(0, w-1)
        reverse = disparities[1-i][yy, lookup_x]
        valid = masks[i] & masks[1-i][yy, lookup_x] & (other_x >= 0) & (other_x < w) & (d > .25) & (reverse > .25)
        cycle = np.abs(d-reverse)
        kernel = np.ones((3, 3), np.uint8)
        spread = cv2.dilate(d, kernel)-cv2.erode(d, kernel)
        xy, z_factor = maps[i][2:]
        px, py = xy[..., 0], xy[..., 1]
        if axis == 1:
            px, py = py, px
        if translation > 0:
            px = w-1-px
        ix, iy = np.rint(px).astype(int).clip(0,w-1), np.rint(py).astype(int).clip(0,h-1)
        keep = (px >= 0) & (px <= w-1) & (py >= 0) & (py <= h-1) & (z_factor > 0) & valid[iy, ix]
        z = abs(translation)/(d[iy, ix].clip(.25)*z_factor.clip(.001))
        keep &= np.isfinite(z) & (z >= .25) & (z <= 30.)
        if trace is not None:
            inside = (px >= 0) & (px <= w-1) & (py >= 0) & (py <= h-1) & (z_factor > 0)
            mapped = inside & masks[i][iy,ix]
            forward = mapped & (d[iy,ix] > .25)
            reverse_valid = inside & valid[iy,ix]
            trace.append([inside,mapped,forward,reverse_valid,keep])
        bins = 3*np.digitize(cycle[iy, ix], [.25, 1.])+np.digitize(spread[iy, ix], [1., 4.])
        observations.append((z, keep, bins))
    return observations


def cases_for(pair, lights, pose, predictions):
    geometry = rectification(pair, pose) if pose is not None else None
    bank = {}
    if geometry is not None:
        for left in (0, 1):
            for right in (0, 1):
                bank[left, right] = stereo_pair([lights[0][left], lights[1][right]], geometry)
    cases = []
    for side in (0, 1):
        truth = pair[side][1][::2, ::2]
        for light in (0, 1):
            pred = predictions[side, light, ::2, ::2]
            full = pair[side][2][::2, ::2] & np.isfinite(pred) & (pred >= .25) & (pred <= 30.)
            keep = full.copy()
            observations = []
            if geometry is None:
                keep[:] = False
            else:
                different = (light, 1-light) if side == 0 else (1-light, light)
                observations = [bank[key][side] for key in ((light, light), different)]
                for _, ok, _ in observations:
                    keep &= ok
            a = np.log(pred[keep]/truth[keep]).astype(np.float64)
            moments = {}
            for mode, (z, _, bins) in zip(CONDITIONS, observations):
                delta = np.log(z[keep]/pred[keep])
                weights = [None, a*a, a*delta, delta**2, np.minimum(a*a, (a+delta)**2)]
                moments[mode] = np.stack([np.bincount(bins[keep], weights=weight, minlength=9)
                                          for weight in weights], axis=1).tolist()
            missing = full & ~keep
            cases.append(dict(side=side, anchor_light=light, grid=full.size, full_valid=int(full.sum()),
                              eligible=int(keep.sum()), unsupported_anchor_sse=float(
                                  np.square(np.log(pred[missing]/truth[missing]).astype(np.float64)).sum()),
                              moments=moments))
    return cases


def main():
    cv2.setNumThreads(1)
    if not ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000):
        raise OSError("Cannot set below-normal CPU priority")
    dataset = Hypersim("validation_24", 4, pair=True)
    stored = RUN/"stereo_cases"
    stored.mkdir(exist_ok=True)
    lut = np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)
    data = {}
    for split in ("train", "dev", "real"):
        data[split] = []
        for row in scene_rows(split):
            scene = row["scene"]
            wait_for_ram()
            cached = RUN/"predictions"/f"{split}-{scene}.npy"
            if cached.exists():
                if split == "real":
                    pair, lights, pose = real_inputs(scene)
                else:
                    a, b, pose = dataset.read_pair(row["row"])
                    pair = [a, b]
                    lights = [[cam[0], lut[cam[0]]] for cam in pair]
                cases = cases_for(pair, lights, pose, np.load(cached, mmap_mode="r"))
            elif row["cases"]:
                raise FileNotFoundError(cached)
            else:
                cases = []
            result = dict(scene=scene, split=split, cases=cases)
            write_json(stored/f"{split}-{scene}.json", result)
            data[split].append(result)
    gains = fit(data["train"], CONDITIONS)
    reports = {}
    for split in ("dev", "real"):
        reports[split] = {name: score(data[split], gains, split, CONDITIONS, with_fallback=fallback)
                          for name, fallback in (("supported", False), ("with_fallback", True))}
    passed = all(min(domain["with_fallback"]["summary"]["equal_condition_mean"][key] for key in
                 ("reduction_vs_condition_fixed_pct","reduction_vs_shuffled_pct")) > 0 for domain in reports.values())
    write_json(RUN/"stereo_results.json", dict(pooled_gain=gains[0], fixed_gain=gains[1].tolist(),
               pixel_gain=gains[2].tolist(), reports=reports, student_exploration=passed))
    for split, reports_for_split in reports.items():
        print(split, reports_for_split["with_fallback"]["summary"]["equal_condition_mean"])
    print("stereo student exploration", passed)


if __name__ == "__main__":
    main()
