"""Fetch the three absent camera positions from the original Middlebury release."""
from concurrent.futures import ThreadPoolExecutor
import json

from repeat_capture_acquire import OUT, SETS, fetch_one


def main():
    jobs = [(year, scene, f"Illum{light}/Exp{exposure}/view{view}.png")
            for year, scenes in SETS.items() for scene in scenes
            for light in (1, 2) for exposure in (1, 2) for view in (2, 4, 6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch_one, jobs))
    (OUT / "multibase-source.json").write_text(json.dumps(rows, indent=1), encoding="utf8")
    print(len(rows), "original photos", sum(r["bytes"] for r in rows), "bytes", flush=True)


if __name__ == "__main__":
    main()
