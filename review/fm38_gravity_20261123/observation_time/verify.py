"""Audit measurements from raw peaks and replay sample trajectories in MuJoCo."""
import hashlib
import json

import mujoco
import numpy as np

from simulate import CONFIRM_SEEDS, HERE, SNAP, TEST_SEEDS, TRAIN_SEEDS
from swing import BAND, DT, LEVELS, PROBE_SPEED, SPEED, build, observe
from world import INERTIA


def stored(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def recount(name):
    with np.load(HERE / f"{name}.npz") as raw, np.load(HERE / f"{name}_choices.npz") as choices:
        wins = abs(raw["peak"] - raw["goal"][:, None]) <= BAND
        np.testing.assert_array_equal(wins, choices["truth"])
        np.testing.assert_array_equal(raw["action"], np.tile(np.arange(4), (len(wins), 1)))
        by_model = {}
        for method in ("prior", "direct", "static"):
            selected = np.moveaxis(choices[method], -1, 0)
            # Evaluate every scene by indexing the four independently computed engine outcomes.
            hits = np.stack([outcomes[picks] for outcomes, picks in zip(wins, selected)])
            by_model[method] = hits.sum(axis=0)
    return by_model, len(wins), int(wins.any(axis=1).sum())


def replay(name):
    with np.load(HERE / f"{name}.npz") as raw:
        for i in (0, 511, 1023):
            m = build(LEVELS["source"][raw["level"][i]])
            m.dof_damping[0] = INERTIA * raw["damping"][i]
            start = raw["start"][i]
            for j, time in enumerate(SNAP):
                data = mujoco.MjData(m)
                data.qpos[0], data.qvel[0] = start, PROBE_SPEED
                mujoco.mj_step(m, data, nstep=round(time/DT))
                np.testing.assert_allclose(data.qpos[0], raw["probe"][i,j], rtol=0, atol=1e-12)
            assert observe(m, start, PROBE_SPEED, probe=True) == raw["probe"][i, SNAP == .55][0]
            for a, speed in enumerate(SPEED):
                future, peak = observe(m, start, speed)
                np.testing.assert_allclose(future, raw["future"][i,a], rtol=0, atol=1e-12)
                np.testing.assert_allclose(peak, raw["peak"][i,a], rtol=0, atol=1e-12)


def audit():
    lock, main, confirmation = (stored(n) for n in ("selection.json", "results.json", "confirmation.json"))
    assert hashlib.sha256((HERE/"policies.joblib").read_bytes()).hexdigest() == lock["model_sha256"]
    starts = set()
    names = [f"train_{seed}" for seed in TRAIN_SEEDS] + ["validation"]
    names += [f"strong_{seed}" for seed in (*TEST_SEEDS, *CONFIRM_SEEDS)]
    for name in names:
        with np.load(HERE/f"{name}.npz") as raw:
            current = set(raw["start"].tolist())
            assert len(current) == len(raw["start"]) and starts.isdisjoint(current)
            starts.update(current)
    for seed in TEST_SEEDS:
        with np.load(HERE/f"seen_{seed}.npz") as a, np.load(HERE/f"strong_{seed}.npz") as b:
            for key in ("start", "goal", "level", "action"):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_allclose(b["damping"] - a["damping"], 1.4)
    val, _, _ = recount("validation")
    for method in ("prior", "direct"):
        np.testing.assert_array_equal(val[method].sum(0), lock[method]["validation_counts"])
        assert val[method].sum(0).argmax() == lock[method]["all"]
    audited_prior = {}
    for domain in ("seen", "strong"):
        data = main["domains"][domain]
        blocks = [recount(f"{domain}_{seed}") for seed in TEST_SEEDS]
        n = sum(b[1] for b in blocks)
        assert n == data["total"] and sum(b[2] for b in blocks) == data["reachable"]
        for method in ("prior","direct"):
            counts = sum(b[0][method] for b in blocks)
            np.testing.assert_allclose(counts/n, data["methods"][method]["per_training_set"])
            np.testing.assert_allclose(counts.mean(0)/n, data["methods"][method]["rates"])
            if method == "prior":
                audited_prior[domain] = counts.sum(0)
        np.testing.assert_allclose(sum(b[0]["static"] for b in blocks).mean()/n, data["no_probe"])
    target = stored("strong_selection.json")
    assert target["idx"] == np.argmax(target["counts"])
    np.testing.assert_array_equal(target["counts"], audited_prior["strong"])
    blocks = [recount(f"strong_{seed}") for seed in CONFIRM_SEEDS]
    n, possible = (sum(b[k] for b in blocks) for k in (1,2))
    counts = sum(b[0]["prior"] for b in blocks).mean(0)
    for j, rate in confirmation["rates"].items():
        np.testing.assert_allclose(counts[int(j)]/n, rate)
    assert n == confirmation["n"] and possible == confirmation["reachable"]
    np.testing.assert_allclose(counts[target["idx"]]/possible, confirmation["conditional"])
    sample = [f"{domain}_{seed}" for domain in ("seen","strong") for seed in TEST_SEEDS]
    sample.extend(f"strong_{seed}" for seed in CONFIRM_SEEDS)
    for name in sample:
        replay(name)
    return {"passed": True, "mujoco": mujoco.__version__, "unique_starts": len(starts),
            "replayed_scenes":len(sample)*3, "snapshot_checks":len(sample)*3*len(SNAP),
            "main_action_replays":len(sample)*3*len(SPEED), "model_sha256":lock["model_sha256"]}


if __name__ == "__main__":
    result = audit()
    (HERE/"checks.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(result)
