"""Acquire a scene-disjoint expansion subset from the official Hypersim ZIPs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import json
from pathlib import Path
from zipfile import ZipFile

from hypersim_fetch import (ARCHIVE, ROOT, SCENES, RemoteZip, frame_assets, has_asset,
                           scene_camera_files, scene_files, scene_pose_assets)


FRAMES = (0, 13, 26, 39, 52, 65)
SCALE_MANIFEST = Path(__file__).with_name("encoder_training_scale.json")
STAGES = ("train_24", "train_64", "validation_24", "train_full")


def selection_for(stage):
    if stage == "train_full":
        selection = {}
        with (ROOT / "splits.csv").open(newline="", encoding="utf8") as stream:
            for row in csv.DictReader(stream):
                if row["included_in_public_release"] == "True" and row["split_partition_name"] == "train":
                    selection.setdefault(row["scene_name"], []).append((row["camera_name"], int(row["frame_id"])))
        return selection
    manifest = json.loads(SCALE_MANIFEST.read_text(encoding="utf8"))
    return manifest["stages"][stage]


def stage_data(stage, fetch, workers):
    selection = selection_for(stage)
    complete = sum(all(has_asset(path) for _, path in frame_assets(scene, camera, frame))
                   for scene, frames in selection.items() for camera, frame in frames)
    total = sum(len(frames) for frames in selection.values())
    print(f"{stage}: {complete}/{total} RGB-depth pairs across {len(selection)} selected scenes", flush=True)
    pose_groups = sum(all(has_asset(path) for _, path in scene_pose_assets(scene, frames))
                      for scene, frames in selection.items())
    print(f"{stage}: all selected camera poses in {pose_groups}/{len(selection)} scenes", flush=True)
    if fetch and (complete < total or pose_groups < len(selection)):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            task = partial(scene_camera_files, include_poses=True)
            for _ in pool.map(task, selection, selection.values()):
                pass
        stage_data(stage, fetch=False, workers=workers)


def poses(scene):
    suffixes = ("metadata_scene.csv",)+tuple(f"cam_00/camera_keyframe_{part}.hdf5"
                                             for part in ("frame_indices","positions","orientations"))
    if all((ROOT/scene/Path(name).name).exists() for name in suffixes):
        return
    with RemoteZip(f"{ARCHIVE}/{scene}.zip") as remote, ZipFile(remote) as archive:
        for name in suffixes:
            member, = (item for item in archive.namelist() if item.endswith("/_detail/"+name))
            (ROOT/scene/Path(name).name).write_bytes(archive.read(member))
    print("Camera poses",scene,flush=True)


def main(only_poses=False):
    if only_poses:
        manifest = json.loads((ROOT/"encoder_expansion_manifest.json").read_text(encoding="utf8"))
        selected = [scene for split in manifest["split"].values() for scene in split]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(poses,selected))
        print("Completed camera poses",len(selected),"scenes",flush=True)
        return
    with (ROOT/"splits.csv").open(newline="", encoding="utf8") as stream:
        rows = csv.DictReader(stream)
        pools = {"train":{}, "val":{}}
        for row in rows:
            scene, split = row["scene_name"], row["split_partition_name"]
            if scene in SCENES or split not in pools or row["camera_name"] != "cam_00":
                continue
            if row["included_in_public_release"] == "True" and int(row["frame_id"]) in FRAMES:
                pools[split].setdefault(scene, set()).add(int(row["frame_id"]))
    with (ROOT/"camera.csv").open(newline="",encoding="utf8") as stream:
        calibrated = {row["scene_name"] for row in csv.DictReader(stream)}
    selected = {}
    for split, limit in (("train", 20),("val", 6)):
        candidates = sorted(scene for scene, ids in pools[split].items()
                            if len(ids)==len(FRAMES) and scene in calibrated)
        volumes = set()
        names = []
        for scene in candidates:
            volume = scene.split("_")[1]
            if volume not in volumes:
                volumes.add(volume)
                names.append(scene)
            if len(names) == limit:
                break
        assert len(names) == limit, (split, len(names))
        selected[split] = names
    jobs = [(scene, list(FRAMES)) for names in selected.values() for scene in names]
    (ROOT/"encoder_expansion_manifest.json").write_text(json.dumps(
        dict(split=selected, frames=FRAMES, official_partition=True),indent=2),encoding="utf8")
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda item:scene_files(*item), jobs))
    print("Completed",len(jobs),"scenes",len(jobs)*len(FRAMES),"frames",flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--poses", action="store_true", help="Fetch camera poses for the pilot")
    mode.add_argument("--stage", choices=STAGES, help="Count local RGB-depth pairs in the selected stage")
    cli.add_argument("--fetch", action="store_true", help="Fetch missing RGB-depth pairs for --stage")
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    if args.fetch and args.stage is None:
        cli.error("--fetch requires --stage")
    if args.workers < 1:
        cli.error("--workers must be positive")
    if args.stage:
        stage_data(args.stage, args.fetch, args.workers)
    else:
        main(args.poses)
