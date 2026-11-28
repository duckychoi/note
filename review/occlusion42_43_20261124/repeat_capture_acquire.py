"""Download original rectified photographs, measured disparities and exposure metadata."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OUT = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval/repeat-capture"
SETS = {
    2005: "Art Books Dolls Laundry Moebius Reindeer".split(),
    2006: ("Aloe Baby1 Baby2 Baby3 Bowling1 Bowling2 Cloth1 Cloth2 Cloth3 Cloth4 Flowerpots "
           "Lampshade1 Lampshade2 Midd1 Midd2 Monopoly Plastic Rocks1 Rocks2 Wood1 Wood2").split(),
}
THREAD = threading.local()


def fetch_one(args):
    year, scene, member = args
    url = f"https://vision.middlebury.edu/stereo/data/scenes{year}/ThirdSize/{scene}/{member}"
    target = OUT / "physical" / scene / member
    if not target.exists():
        if not hasattr(THREAD, "http"):
            THREAD.http = requests.Session()
            THREAD.http.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.3,
                               status_forcelist=(429,500,502,503,504))))
        response = THREAD.http.get(url, timeout=35)
        response.raise_for_status()
        data = response.content
        if member.endswith(".png") and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(url)
        if member.endswith(".txt"):
            float(data.strip())
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".part")
        temp.write_bytes(data)
        temp.replace(target)
    data = target.read_bytes()
    return dict(year=year,scene=scene,member=member,url=url,bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def main():
    jobs = []
    for year, names in SETS.items():
        for scene in names:
            members = ["disp1.png","disp5.png","dmin.txt"]
            for light in (1,2):
                for exposure in (0,1,2):
                    prefix = f"Illum{light}/Exp{exposure}/"
                    members.append(prefix+"exposure_ms.txt")
                    if exposure in (1,2):
                        members.extend(prefix+f"view{v}.png" for v in (1,3,5))
            jobs.extend((year,scene,member) for member in members)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = [task.result() for task in as_completed([pool.submit(fetch_one,j) for j in jobs])]
    (OUT/"physical-source.json").write_text(json.dumps(sorted(rows,key=lambda r:r["url"]),indent=1),
                                            encoding="utf8")
    print(len(rows),"verified files",sum(r["bytes"] for r in rows),"bytes",flush=True)


if __name__ == "__main__":
    main()
