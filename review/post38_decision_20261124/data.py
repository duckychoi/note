"""Download the two primary-source datasets and record file hashes."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2] / "research-sources" / "post38_decision_20261124"
SCENES = {
    2005: "Art Books Dolls Laundry Moebius Reindeer".split(),
    2006: "Aloe Baby1 Baby2 Baby3 Bowling1 Bowling2 Cloth1 Cloth2 Cloth3 Cloth4 Flowerpots Lampshade1 Lampshade2 Midd1 Midd2 Monopoly Plastic Rocks1 Rocks2 Wood1 Wood2".split(),
}
HAPT = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"


def fetch(pair):
    name, url = pair
    target = CACHE / name
    if not target.exists():
        for attempt in range(3):
            try:
                response = requests.get(url,timeout=(12,90))
                response.raise_for_status()
                temp = target.with_suffix(".part")
                temp.write_bytes(response.content)
                with ZipFile(temp) as z:
                    assert z.testzip() is None
                temp.replace(target)
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
    return name,dict(url=url,sha256=sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    CACHE.mkdir(parents=True,exist_ok=True)
    urls = [("hapt.zip", HAPT)]
    for year,names in SCENES.items():
        urls.extend((name+".zip",f"https://vision.middlebury.edu/stereo/data/scenes{year}/ThirdSize/zip-7views/{name}-7views.zip") for name in names)
    with ThreadPoolExecutor(max_workers=5) as pool:
        manifest = dict(pool.map(fetch,urls))
    (HERE/"inputs.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print(f"{len(manifest)} archives")
