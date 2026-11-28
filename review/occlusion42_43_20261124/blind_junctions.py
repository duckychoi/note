"""Exploratory RGB-only junction detection on a fresh synthetic split."""
import json

import cv2
import numpy as np

from common import ROOT
from diagnose import front_choice
from junctions import render


def detect(image):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 24, 65)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 11, minLineLength=9, maxLineGap=5)
    if lines is None:
        return []
    segments = lines.reshape(-1, 4).astype(float)
    starts, ends = segments[:, :2], segments[:, 2:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    directions = (ends - starts) / lengths[:, None]
    candidates = []
    for i, (start, axis) in enumerate(zip(starts, directions)):
        for j in range(i):
            other = directions[j]
            cross = axis[0] * other[1] - axis[1] * other[0]
            if abs(cross) < 0.72:
                continue
            offset = starts[j] - start
            u = (offset[0] * other[1] - offset[1] * other[0]) / cross
            v = (offset[0] * axis[1] - offset[1] * axis[0]) / cross
            if not (-6 <= u <= lengths[i] + 6 and -6 <= v <= lengths[j] + 6):
                continue
            x, y = start + u * axis
            angles = np.arctan2(directions[[i, j], 1], directions[[i, j], 0])
            candidates.append((x, y, *angles))
    if not candidates:
        return []
    points = np.array(candidates)
    distances = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=2)
    rank = np.maximum(0, 1 - distances / 8).sum(axis=1)
    picks = []
    for idx in np.argsort(-rank, kind="stable"):
        if any(distances[idx, chosen] < 17 for chosen in picks):
            continue
        picks.append(idx)
        if len(picks) == 4:
            break
    return points[picks].tolist()


def evaluate(seeds, split):
    records = []
    for seed in seeds:
        for base in range(8):
            states = []
            for state in (0, 1):
                frames, _, _, axes = render(seed, split, base, state, True, return_axes=True)
                image, depth, queries = frames[0]
                predictions = detect(image)
                scored = [[False, False, False] for _ in axes]
                labels = [
                    np.sign(depth.ravel()[q[1]] - depth.ravel()[q[0]])
                    for q in queries[::24]
                ]
                assert len(labels) == len(axes) and all(labels)
                supplied = [front_choice(image, axis) == label for axis, label in zip(axes, labels)]
                pairs = sorted(
                    (float(np.hypot(p[0] - a[0], p[1] - a[1])), i, j)
                    for i, a in enumerate(axes) for j, p in enumerate(predictions)
                )
                used = set()
                for distance, i, j in pairs:
                    if distance > 8 or scored[i][0] or j in used:
                        continue
                    used.add(j)
                    pred = predictions[j]
                    similarity = np.abs(np.cos(np.subtract.outer(pred[2:], axes[i][2:])))
                    order = (0, 1) if np.trace(similarity) >= np.trace(similarity[:, ::-1]) else (1, 0)
                    direction_ok = min(similarity[k, order[k]] for k in (0, 1)) >= np.cos(np.pi / 9)
                    sign = front_choice(image, pred) * (1 if order == (0, 1) else -1)
                    scored[i] = [True, direction_ok, bool(direction_ok and sign == labels[i])]
                states.append((scored, supplied, labels))
            assert np.array_equal(states[0][2], -np.array(states[1][2]))
            a = np.array([state[0] for state in states])
            supplied = np.array([state[1] for state in states])
            records.append(dict(seed=seed, base=base, crossings=a.shape[1],
                                located=int(a[:, :, 0].sum()), directions=int(a[:, :, 1].sum()),
                                correct=int(a[:, :, 2].sum()), both=int(a[:, :, 2].all(axis=0).sum()),
                                supplied_both=int(supplied.all(axis=0).sum())))
    return records


if __name__ == "__main__":
    rows = evaluate((6323, 6337, 6353, 6361, 6379), 23)
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key not in ("seed", "base")}
    (ROOT / "blind_junctions.json").write_text(
        json.dumps(dict(split=23, totals=totals, rows=rows), indent=2) + "\n", encoding="utf-8"
    )
    print(totals)
