"""Rectified stereo pairs from the same procedural textured surfaces, with z-buffers."""
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "studio43_first_20261124"
sys.path.insert(0, str(PREVIOUS))
from train import DepthNet, infer, tensor_input

CACHE = HERE.parents[2] / "research-sources" / "gain42_video"
HEIGHT, WIDTH = 120, 160
Y, X = np.indices((HEIGHT, WIDTH), np.float32)
COUNTS = [1, 1, 2, 2, 4, 4, 8, 10]


def pair(seed, split, parent, variant):
    scene = np.random.default_rng([seed, split, parent])
    view = np.random.default_rng([seed, split, parent, variant, 77])
    focal = scene.uniform(.78, 1.35) * WIDTH
    baseline = view.uniform(.035, .095)
    origin = view.uniform(-.065, .065)
    exposure = view.uniform(.68, 1.25)
    surfaces = []
    for i in range(16):
        z = scene.uniform(.38, 2.1) if i else scene.uniform(2.2, 3.6)
        cx, cy = scene.uniform(-.54, .54) * WIDTH, scene.uniform(-.48, .48) * HEIGHT
        a, b = scene.uniform(8, 35), scene.uniform(6, 22)
        shape = scene.integers(0, 2)
        if i % 3 == 1:
            a, b = scene.uniform(11, 39), scene.uniform(.8, 3.7)
        angle = scene.uniform(-np.pi, np.pi)
        color = scene.uniform(38, 228, 3)
        texture = scene.uniform(.03, .36)
        frequency = scene.uniform(.04, .70, 2)
        phase = scene.uniform(-3, 3)
        surfaces.append((z, cx, cy, a, b, shape, angle, color, texture, frequency, phase, i))
    frames = []
    for camera in (origin, origin + baseline):
        rgb = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        depth = np.zeros((HEIGHT, WIDTH), np.float32)
        labels = np.full(depth.shape, -1, np.int16)
        for z, cx, cy, a, b, shape, angle, color, texture, freq, phase, i in sorted(surfaces, reverse=True):
            xx = X - (WIDTH-1)/2 + focal * camera / z - cx
            yy = Y - (HEIGHT-1)/2 - cy
            u = np.cos(angle)*xx + np.sin(angle)*yy
            v = -np.sin(angle)*xx + np.cos(angle)*yy
            if i == 0:
                mask = np.ones(depth.shape, bool)
            elif shape:
                mask = (np.abs(u) < a) & (np.abs(v) < b)
            else:
                mask = (u/a)**2 + (v/b)**2 < 1
            factor = 1 + texture * np.sin(freq[0]*u[mask] + phase) * np.cos(freq[1]*v[mask])
            rgb[mask] = color * factor[:, None] * exposure
            depth[mask] = z
            labels[mask] = i
        rgb = np.uint8(np.clip(rgb + view.normal(0, 2, rgb.shape), 0, 255))
        frames.append((rgb, depth, labels))
    return frames, focal, baseline


def disparities(a, b):
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=48, blockSize=3,
                                    P1=72, P2=288, uniquenessRatio=8, speckleWindowSize=0)
    left = matcher.compute(a, b).astype(np.float32)/16
    right = np.fliplr(matcher.compute(np.fliplr(b).copy(), np.fliplr(a).copy())).astype(np.float32)/16
    return left, right


def make_views(seed, split, parent, variant):
    frames, f, base = pair(seed, split, parent, variant)
    gray = [cv2.cvtColor(v[0], cv2.COLOR_RGB2GRAY) for v in frames]
    disps = disparities(*gray)
    output = []
    for i, direction in enumerate((-1, 1)):
        rgb, depth, labels = frames[i]
        d, other = disps[i], disps[1-i]
        def project(disparity):
            pos = np.rint(X + direction*disparity).astype(int)
            return (pos >= 0) & (pos < WIDTH), np.clip(pos, 0, WIDTH-1)
        inside, exact = project(f*base/depth)
        visible = inside & (labels == frames[1-i][2][Y.astype(int), exact])
        inside, ix = project(d)
        reverse = other[Y.astype(int), ix]
        consistency = np.abs(d-reverse)
        good = visible & inside & (d > .25) & (reverse > .25) & (consistency <= 1)
        dx = cv2.Sobel(gray[i].astype(np.float32)/255, -1, 1, 0)/4
        dy = cv2.Sobel(gray[i].astype(np.float32)/255, -1, 0, 1)/4
        observation = np.maximum(d, 0)/(f*base)
        output.append((tensor_input(rgb, f), depth, observation, good, np.hypot(dx,dy),
                       consistency, int(visible.sum())))
    return output
