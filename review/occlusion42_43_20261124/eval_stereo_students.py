"""Blind RGB depth evaluation of completed training arms, grouped by HAMMER physical scene."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

from controlled_observations import RUN, SOURCE, write_json
from run_stereo_students import ORIGINAL, capacity


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hammer_scenes(archive):
    from depth_full_eval import sample_names
    names=sample_names(archive,indoor=False)
    if len(names)!=775 or len(set(names))!=775:
        raise ValueError("Unexpected held-out image inventory")
    scenes={}
    for name in names:
        match=re.match(r"^HAMMER/(scene[0-9]+)_",name)
        if match is None:
            raise ValueError(f"Unknown physical scene for {name}")
        scenes.setdefault(match.group(1),[]).append(name)
    return scenes


def training_records(path):
    entries=[]
    for line in (path/"training.jsonl").read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        if "resumed_at" in row:
            entries=[e for e in entries if e.get("step",0)<=row["resumed_at"]]
        else:
            entries.append(row)
    return entries


def best_development(path):
    entries = training_records(path)
    if not (path/"completed").exists() or entries[0]["config"]["smoke"]:
        raise ValueError(f"Incomplete full run {path}")
    return min(record["development_absrel"] for record in entries if "development_absrel" in record)


def main(geometry=False,seed=4244):
    if not capacity():
        raise SystemExit(75)
    import torch
    from depth_full_eval import compare, evaluate
    from stereo_student import memory_pressure
    sys.path.insert(0, str(SOURCE/"ml-depth-pro-2"/"src"))
    from depth_pro_2 import create_model_and_transforms

    if geometry:
        from select_stereo_pairs import OUT
        home = OUT/f"full-students-{seed}"
        arm_paths = {label:home/label for label in ("d2","silog","fixed")}
        arm_paths.update(kf=home/"adaptive",real=ORIGINAL/"real")
    else:
        home = RUN/"full-students-4244"
        arm_paths = {label: ORIGINAL/label for label in ("d2", "silog", "fixed", "real")}
        arm_paths.update({label: home/mode for label,mode in (("student_off","off"),("student_fixed","fixed"),("kf","adaptive"))})
    dev = {label:best_development(folder) for label,folder in arm_paths.items()}
    reference = min(("d2","silog") if geometry else ("student_off","student_fixed"), key=dev.get)
    previous_reference = min(("d2","fixed","silog"), key=dev.get)
    if geometry:
        evidence = {}
        learning_curves = {}
        for arm in ("d2","silog","fixed","kf"):
            entries = training_records(arm_paths[arm])
            meta = entries[0]
            cfg = meta["config"]
            final_step = max(x["step"] for x in entries if "development_absrel" in x)
            trajectory=[(x["step"],x["development_absrel"]) for x in entries if "development_absrel" in x]
            best_step,best_error=min(trajectory,key=lambda x:x[1])
            mid=[v for s,v in trajectory if s<=final_step//2]
            learning_curves[arm]=dict(development_checkpoints=len(trajectory),best_step=best_step,
                best_rgb_absrel=best_error,final_rgb_absrel=trajectory[-1][1],best_in_final_quarter=best_step>=.75*final_step,
                second_pass_improvement_over_best_first_pass_percent=100*(1-best_error/min(mid)) if mid else None)
            if (cfg["seed"]!=seed or cfg["epochs"]!=2 or meta["training_rows"]!=59543 or final_step!=59543*cfg["epochs"] or cfg["steps"] is not None
                or cfg["development_mode"]!="native_rgb_scene_mean"):
                raise ValueError(f"Incomplete matched training in {arm}")
            evidence[arm] = (meta["selection_digest"],cfg["stereo_calibration_sha256"],cfg["seed"],cfg["stride"],
                             cfg["lr"],cfg["epochs"],cfg["val_every"],meta["development_rows"])
            if cfg["arm"]!=(arm if arm in ("d2","silog") else reference) or cfg["stereo_mode"]!=dict(
                    d2="off",silog="off",fixed="fixed",kf="adaptive")[arm]:
                raise ValueError(f"Unmatched training objective in {arm}")
        if len(set(evidence.values()))!=1:
            raise ValueError("Different training manifests or budgets")
    archive_path = SOURCE/"encoder42_eval"/"HAMMER.zip"
    archive_sha = sha(archive_path)
    if archive_sha != "d5a92384b112db9d0b468de9700d639f071ff464b5501560f4a33bcfd6a27098":
        raise ValueError("Held-out archive SHA-256 mismatch")
    torch.set_num_threads(1)
    torch.hub.set_dir(str(SOURCE/"ml-depth-pro-2-cache"/"hub"))
    model, transform = create_model_and_transforms("depthpro2-s", device="cuda")
    outputs = {}
    shard_dir = home/"hammer-scenes"
    shard_dir.mkdir(exist_ok=True)
    with ZipFile(archive_path) as archive:
        scenes=hammer_scenes(archive)
        names=[name for frames in scenes.values() for name in frames]
        checkpoints={}
        for label, folder in arm_paths.items():
            ckpt = folder/"best.pt"
            digest = sha(ckpt)
            checkpoints[label]=digest
            mismatch = model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False)
            if mismatch.unexpected_keys or any(not key.startswith("fov.") for key in mismatch.missing_keys):
                raise ValueError(f"Invalid weights for {label}")
            results = {}
            for scene, frames in scenes.items():
                path = shard_dir/f"{label}-{scene}.json"
                if not path.exists():
                    if memory_pressure():
                        raise SystemExit(75)
                    rows = evaluate(model, transform, archive, frames, stride=1)
                    write_json(path, dict(archive_sha256=archive_sha, checkpoint_sha256=digest, rows=rows))
                saved = json.loads(path.read_text(encoding="utf8"))
                if saved["checkpoint_sha256"] != digest or saved["archive_sha256"] != archive_sha or set(saved["rows"]) != set(frames):
                    raise ValueError(f"Stale scene scores at {path}")
                results.update(saved["rows"])
            outputs[label] = results
    report = compare(outputs, indoor=True)
    report.update(frames=len(names), comparison_reference_matched=reference, comparison_reference_original=previous_reference,
                  checkpoint_selection=dev, primary="rgb", test="HAMMER", comparison_unit="physical scene",
                  student_seed=seed, confirmatory=False,archive_sha256=archive_sha,checkpoints_sha256=checkpoints)
    if geometry:
        report["matched_training"]=evidence
        report["development_learning_curves"]=learning_curves
        report["nyu_reference_scope"]="Prior NYU reference uses 119086 single-image exposures and known-focal development selection. Each new synthetic student uses 238172 image exposures and native RGB selection. NYU is a descriptive reference."
        report["rgb_3pct"]={}
        for label in ("d2","silog","fixed","real"):
            comparison=report["compared_with_kf"][label]
            value=comparison["relative_reduction_percent"][0]
            interval=comparison["ci95"][0]
            report["rgb_3pct"][label]=dict(reduction=value,ci95=interval,point_at_least_3=value>=3,
                                           lower_bound_at_least_3=interval[0]>=3)
    write_json(home/"hammer_rgb_results.json", report)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",action="store_true")
    parser.add_argument("--seed",type=int,default=4244)
    from run_stereo_students import gpu_slot
    with gpu_slot():
        args=parser.parse_args()
        import torch
        try:
            main(args.geometry,args.seed)
        except torch.cuda.OutOfMemoryError:
            print("GPU memory pressure; resume at the last completed physical scene.",flush=True)
            raise SystemExit(75)
