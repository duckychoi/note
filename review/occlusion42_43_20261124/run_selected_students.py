"""Run the four matched full-data arms, then the fixed RGB-only HAMMER comparison."""
import json
import argparse
import msvcrt
from pathlib import Path
import subprocess
import sys
import time

from controlled_observations import SOURCE, write_json
from eval_stereo_students import best_development
from run_stereo_students import capacity
from select_stereo_pairs import OUT


def main(seed=4244):
    home=OUT/f"full-students-{seed}"
    home.mkdir(exist_ok=True,parents=True)
    scripts=Path(__file__).resolve().parent
    work=home/"queue.jsonl"
    def record(event):
        with work.open("a",encoding="utf8") as f:
            print(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),event=event)),file=f,flush=True)
    lock=(home/"queue.lock").open("a+b")
    if not lock.tell():
        lock.write(b"1")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    manifest=OUT/"pairs-full.json"
    calibration=OUT/"stereo_results.json"
    record("waiting for official 59543-anchor geometry manifest and calibration")
    while not manifest.exists() or not calibration.exists():
        time.sleep(30)
    if len(json.loads(manifest.read_text()))!=59543:
        raise ValueError("Incomplete official training set")
    for label in ("d2","silog","fixed","adaptive","hammer"):
        folder=home/label
        if label=="hammer":
            if (home/"hammer_rgb_results.json").exists():
                break
            record("waiting for verified HAMMER archive")
            while not (SOURCE/"encoder42_eval"/"HAMMER.zip").exists():
                time.sleep(30)
            args=[str(scripts/"eval_stereo_students.py"),"--geometry","--seed",str(seed)]
        else:
            if (folder/"completed").exists():
                continue
            if label in ("d2","silog"):
                arm,mode=label,"off"
            else:
                scores={k:best_development(home/k) for k in ("d2","silog")}
                arm,mode=min(scores,key=scores.get),label
                write_json(home/"development_choice.json",dict(reference=arm,absrel=scores))
            args=[str(scripts/"depth_full_train.py"),"--arm",arm,"--stereo-mode",mode,"--epochs","2",
                  "--seed",str(seed),"--pair-manifest",str(manifest),"--stereo-calibration",str(calibration),
                  "--output",str(folder)]
        while True:
            while not capacity():
                time.sleep(30)
            time.sleep(4)
            if not capacity():
                continue
            resume=label!="hammer" and (folder/"latest.train.pt").exists()
            record(("resume " if resume else "launch ")+label)
            with (home/f"{label}-console.log").open("a",encoding="utf8") as log:
                code=subprocess.call([sys.executable,"-u",*args,*(["--resume"] if resume else [])],
                                     stdout=log,stderr=subprocess.STDOUT)
            if code == 76:
                record(f"{label} deferred for theory review; checkpoints retained")
                raise SystemExit(0)
            if code==75:
                record(label+" waiting for RAM or GPU capacity")
                time.sleep(30)
                continue
            if code:
                record(f"{label} failed with exit {code}")
                raise SystemExit(code)
            if label!="hammer":
                (folder/"completed").touch()
            record(label+" completed")
            break


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=4244)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--remaining-seeds",action="store_true")
    mode.add_argument("--all-seeds",action="store_true")
    args=parser.parse_args()
    if args.remaining_seeds or args.all_seeds:
        # The three seeds and all choices are fixed before encountering test scores.
        # Wait only for the file to exist. Do not use its measured value as a gate.
        if args.remaining_seeds:
            while not (OUT/"full-students-4244"/"hammer_rgb_results.json").exists():
                time.sleep(45)
        for seed in ((4245,4246) if args.remaining_seeds else (4244,4245,4246)):
            main(seed)
        from summarize_selected_students import summarize
        summarize()
    else:
        main(args.seed)
