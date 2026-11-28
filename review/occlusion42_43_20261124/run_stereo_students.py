"""Run matched stereo students serially after the original comparison, with memory headroom."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time

from controlled_observations import RUN, SOURCE, free_gib, write_json


ORIGINAL = SOURCE / "encoder42_eval/full-comparison-4243"
HERE = Path(__file__).resolve().parent


def yield_previous_training(output):
    """Give the updated first-seed experiment the next GPU turn while its worker is live."""
    import msvcrt
    if Path(output).resolve().parent!=ORIGINAL.resolve():
        return
    selected=RUN.parent/"control43-geometry-pairs-v1"
    home=selected/"full-students-4244"
    lock=home/"queue.lock"
    started=time.monotonic()
    while lock.exists() and not (home/"adaptive"/"completed").exists():
        if not (selected/"pairs-full.json").exists() and time.monotonic()-started>600:
            return
        with lock.open("r+b") as handle:
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except PermissionError:
                pass
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
                return
        time.sleep(30)


@contextmanager
def gpu_slot():
    """Serialize the local #42 training and evaluation workers on the shared GPU."""
    import msvcrt
    path = SOURCE/"encoder42_eval"/"depth-training-gpu.lock"
    with path.open("a+b") as handle:
        if not handle.tell():
            handle.write(b"1")
            handle.flush()
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                break
            except PermissionError:
                time.sleep(15)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def capacity():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    vram, usage = map(int, raw.strip().splitlines()[0].split(","))
    return free_gib() >= 16 and vram >= 12000 and usage < 45


def main():
    home = RUN/"full-students-4244"
    home.mkdir(exist_ok=True, parents=True)
    def record(event):
        with (home/"queue.jsonl").open("a", encoding="utf8") as f:
            print(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), event=event)), file=f)
    import msvcrt
    with (home/"worker.lock").open("a+b") as lock:
        if not lock.tell():
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        record("waiting for original D2, fixed, KF, NYU and SiLog evaluation")
        while not (ORIGINAL/"diode_indoor_with_silog.json").is_file():
            time.sleep(45)
        # The trigger file only signals completion. Selection uses training logs.
        scores = {}
        for candidate in ("d2", "silog"):
            log = (ORIGINAL/candidate/"training.jsonl").read_text(encoding="utf8").splitlines()
            rows = [json.loads(line) for line in log if line.strip()]
            scores[candidate] = min(r["development_absrel"] for r in rows if "development_absrel" in r)
        recipe = min(scores, key=scores.get)
        write_json(home/"recipe.json", dict(recipe=recipe, initial_dev_scores=scores, seed=4244, epochs=1,
                   student_checkpoint_selection="native RGB scene macro on synthetic development set"))
        os.environ.update(PYTHONPATH=str(SOURCE/"ml-depth-pro-2-deps"), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for mode in ("off", "fixed", "adaptive"):
            out = home/mode
            if (out/"completed").exists():
                continue
            while True:
                if not capacity():
                    time.sleep(45)
                    continue
                time.sleep(4)
                if not capacity():
                    continue
                cmd = [sys.executable, "-u", str(HERE/"depth_full_train.py"), "--arm", recipe, "--stereo-mode", mode,
                       "--seed", "4244", "--epochs", "1", "--output", str(out)]
                if (out/"latest.train.pt").exists():
                    cmd.append("--resume")
                record("starting "+mode)
                with (home/(mode+"-console.log")).open("a", encoding="utf8") as stream:
                    code = subprocess.call(cmd, stdout=stream, stderr=subprocess.STDOUT)
                if code == 75:
                    record("released memory "+mode)
                    time.sleep(45)
                    continue
                if code:
                    record(f"{mode} failed with exit {code}")
                    raise SystemExit(code)
                (out/"completed").touch()
                break
        record("all three matched students finished")


if __name__ == "__main__":
    main()
