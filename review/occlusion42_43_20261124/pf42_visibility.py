"""Measure missed occlusions and test ray-depth evidence on a new DSLR holdout."""
import argparse
from collections import Counter
import json

import cv2
import numpy as np

from pf42_crossview import comparison, decide, entries, load, masks, mix_evidence, risk_decode, soft_curve
from pf42_crossview_data import DEPTH, HELD, OUT, atlas, build, curves, grid, move, peak, scene, select


NEW = ("playground", "terrace")
TEMP, RATE = .20, .05
PARTITION = "visibility-held"


def scene_context(name, partition):
    paths = list(entries(partition, [name]))
    cams = scene(name)
    cameras = {str(c.path): c for c in cams}
    anchors = [cameras[str(load(p)["ref"])] for p in paths]
    return paths, cams, cameras, anchors


def map_values(pix, inside, atlas_data):
    xy, depths = atlas_data
    h, w = depths.shape
    x = np.rint((pix[..., 0] - xy[0, 0, 0]) * (w - 1) / (xy[0, -1, 0] - xy[0, 0, 0])).astype(int)
    y = np.rint((pix[..., 1] - xy[0, 0, 1]) * (h - 1) / (xy[-1, 0, 1] - xy[0, 0, 1])).astype(int)
    in_map = inside & (x>=0) & (x<w) & (y>=0) & (y<h)
    value = depths[y.clip(0, h - 1), x.clip(0, w - 1)]
    return np.where(in_map, value, np.nan), in_map


def nearest_grid(truth):
    target = np.where(np.isfinite(truth)&(truth>0), truth, 1.)
    return abs(np.log(DEPTH[None]) - np.log(target[:, None])).argmin(1)


def diagnose():
    by_scene = {}
    for name in HELD:
        paths, cams, cameras, anchors = scene_context(name, "held")
        depths, stats = {}, Counter()
        for path in paths:
            obs, truth = load(path), load(path.with_suffix(".truth.npz"))
            ref = cameras[str(obs["ref"])]
            common = masks(obs, truth)["all"]
            idx = nearest_grid(truth["truth"])
            rows = np.arange(len(idx))
            ray = ref.rays(obs["tally"])
            for j, source in enumerate(obs["sources"]):
                src = cameras[str(source)]
                known = (truth["occlusion"][:, j] == 1) & common
                known &= np.isfinite(obs["cost"][rows, j, idx]) & (obs["visibility"][rows, j, idx] > 0)
                if not known.any():
                    continue
                if src not in depths:
                    depths[src] = atlas(src, cams, anchors, with_reason=True)
                pix, z, inside = src.project(move(ref, src, ray, DEPTH[idx, None]))
                xy, depth, why_missing = depths[src]
                near, in_map = map_values(pix, inside, (xy, depth))
                reason, _ = map_values(pix, inside, (xy, why_missing))
                recovered = obs["visibility"][rows, j, idx] < .5
                missing = known & ~recovered
                stats.update(recovered=int((known&recovered).sum()), outside=int((missing&~in_map).sum()),
                             unknown=int((missing&in_map&~np.isfinite(near)).sum()),
                             bad_depth=int((missing&np.isfinite(near)).sum()))
                for code, cause in enumerate(("not_enough_camera_pairs", "not_enough_valid_costs", "disagreeing_depths"), 1):
                    stats[cause] += int((missing & (reason==code)).sum())
                sub = missing & np.isfinite(near)
                foreground = src.depth(pix)
                stats["bad_depth_has_lidar"] += int((sub & np.isfinite(foreground)).sum())
                stats["overestimated_front"] += int((sub & (near > 1.10 * foreground)).sum())
        by_scene[name] = dict(stats)
        print(name, by_scene[name], flush=True)
    (OUT / "visibility-diagnosis.json").write_text(json.dumps(by_scene, indent=2), encoding="utf8")


def consensus_map(src, cams, excluded):
    xy = grid(src, 52, 36, 5)
    flat = xy.reshape(-1, 2)
    partners = select(src, [c for c in cams if c not in excluded], 6, True)
    if len(partners) < 2:
        return xy, np.full(xy.shape[:2], np.nan, dtype=np.float32)
    curves_by_cam = np.stack([curves(src, cam, flat) for cam, *_ in partners], axis=1)
    evidence, support = soft_curve(curves_by_cam, np.full(curves_by_cam.shape, .85, dtype=np.float32), TEMP)
    likelihood = mix_evidence(evidence, RATE).sum(1)
    mode = np.argmax(np.where(support>=2, likelihood, -np.inf), axis=1)
    z = DEPTH[mode]
    pair_depth = np.stack([peak(curves_by_cam[:, i])[0] for i in range(len(partners))], axis=1)
    log_pair = np.log(pair_depth)
    matches = abs(log_pair - np.log(z)[:, None]) < .10
    n = matches.sum(1)
    center = np.exp(np.where(matches, log_pair, 0.).sum(1) / np.maximum(1, n))
    return xy, np.where(n>=2, center, np.nan).reshape(xy.shape[:2])


def fuse(evidence, support, groups, median, shared, *, posterior=False):
    if shared:
        score = sum(mix_evidence(evidence[:, groups==g].sum(1), RATE) for g in np.unique(groups))
    else:
        score = mix_evidence(evidence, RATE).sum(1)
    estimate, probability = risk_decode(score, support>=2, median)
    if posterior:
        return probability
    return np.where(np.isfinite(median), estimate, np.nan)


def state_evidence(rgb, z, inside, front):
    delta = np.maximum(.08, .10 * front)
    occ = inside & (z > front + delta)
    impossible = inside & (z < front - delta)
    return np.where(occ, 0., np.where(impossible, np.log(.05), rgb)), occ, impossible


def predictions(obs, ref, sources, maps, use_lidar=False, *, posterior=False):
    ray = ref.rays(obs["tally"])
    median = decide(obs, "median", TEMP, RATE)
    rgb, support = soft_curve(obs["cost"], np.full(obs["cost"].shape,.85,dtype=np.float32), TEMP)
    only_visibility = rgb.copy()
    occ_at_z = []
    for j, src in enumerate(sources):
        pix, z, inside = src.project(move(ref, src, ray[:, None], DEPTH[None, :, None]))
        if use_lidar:
            front = src.depth(pix.reshape(-1, 2)).reshape(z.shape)
        else:
            front, _ = map_values(pix, inside, maps[src])
        rgb[:, j], occ, _ = state_evidence(rgb[:, j], z, inside, front)
        only_visibility[:, j] = np.where(occ, 0., only_visibility[:, j])
        occ_at_z.append(occ)
    if posterior:
        return fuse(rgb, support, obs["groups"], median, True, posterior=True)
    pred = [fuse(rgb, support, obs["groups"], median, mode) for mode in (False, True)]
    pred.append(fuse(only_visibility, support, obs["groups"], median, False))
    return pred, np.stack(occ_at_z, axis=1)


def evaluate():
    rows = []
    for name in NEW:
        paths, cams, cameras, anchors = scene_context(name, PARTITION)
        depths = {}
        for path in paths:
            obs = load(path)
            ref = cameras[str(obs["ref"])]
            sources = [cameras[str(p)] for p in obs["sources"]]
            for c in sources:
                if c not in depths:
                    depths[c] = consensus_map(c, cams, anchors)
            pred = {"median": decide(obs,"median",TEMP,RATE), "old_ind": decide(obs,"visibility",TEMP,RATE),
                    "old_shared": decide(obs,"shared",TEMP,RATE)}
            (pred["geo_ind"], pred["geo_shared"], pred["geo_occlusion_only"]), occlusion = predictions(obs, ref, sources, depths)
            output = OUT / f"visibility-{path.stem}.npz"
            np.savez_compressed(output, **pred)
            (pred["lidar_ind"], pred["lidar_shared"], pred["lidar_occlusion_only"]), _ = predictions(obs, ref, sources, depths, use_lidar=True)
            truth = load(path.with_suffix(".truth.npz"))
            keep = masks(obs, truth)
            z = truth["truth"]
            idx = nearest_grid(z)
            i = np.arange(len(z))
            eligible = keep["all"][:,None] & np.isfinite(obs["cost"][i,:,idx])
            actual = eligible & (truth["occlusion"] == 1)
            old = (obs["visibility"][i, :, idx] < .5) & (obs["visibility"][i, :, idx] > 0)
            new = occlusion[i, :, idx]
            clear = eligible & (truth["occlusion"] == 0)
            counts = dict(label=int(actual.sum()), clear=int(clear.sum()), old=int((actual&old).sum()),
                          geo=int((actual&new).sum()), old_fp=int((clear&old).sum()), geo_fp=int((clear&new).sum()))
            report = {}
            for part in ("all", "hard", "lidar_occluded"):
                ok = keep[part]
                if ok.any():
                    report[part] = dict(n=int(ok.sum()), absrel={a:float(abs(p[ok]/z[ok]-1).mean()) for a,p in pred.items()})
            rows.append(dict(scene=name, frame=path.stem, scores=report, occlusion=counts))
            print(name, path.stem, report.get("hard"), counts, flush=True)
    summaries = {}
    for part in ("all", "hard", "lidar_occluded"):
        selected = [r for r in rows if part in r["scores"]]
        if not selected:
            continue
        report = [r["scores"][part] for r in selected]
        scenes = {}
        for scene_name in NEW:
            same = [r["scores"][part]["absrel"] for r in selected if r["scene"]==scene_name]
            if same:
                scenes[scene_name] = {a:float(np.mean([r[a] for r in same])) for a in same[0]}
        methods = report[0]["absrel"].keys()
        means = {a:float(np.mean([r[a] for r in scenes.values()])) for a in methods}
        summaries[part] = dict(n=sum(r["n"] for r in report), means=means, scenes=scenes,
                               vs={b:comparison({g:r["geo_shared"] for g,r in scenes.items()},
                                                {g:r[b] for g,r in scenes.items()})
                                   for b in ("old_ind", "geo_ind")})
    means = summaries["hard"]["means"]
    gate = means["geo_shared"] < min(means["old_ind"],means["geo_ind"])
    info = dict(summaries=summaries, rows=rows, encoder_gate=gate)
    (OUT/"visibility-results.json").write_text(json.dumps(info,indent=2,allow_nan=False),encoding="utf8")
    print(summaries, "encoder gate", gate, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("diagnose", "build", "evaluate"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "build":
        for name in NEW:
            if not all(c.truth.is_file() for c in scene(name)):
                raise FileNotFoundError(f"Complete the depth archive extraction for {name} first")
            build(name,PARTITION)
    elif args.stage == "diagnose":
        diagnose()
    else:
        evaluate()
