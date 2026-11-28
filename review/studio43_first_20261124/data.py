"""Download only the preselected ETH3D frames and official RealBokeh test pairs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2]/"research-sources"/"studio43_first"
DEV = ("cables_1","desk_changing_1")
TEST = ("einstein_1","mannequin_1","plant_1","reflective_1","sofa_1","table_1","ceiling_1","sfm_lab_room_1")
ETH = "https://www.eth3d.net/data/slam/datasets/"
HF = "https://huggingface.co/datasets/timseizinger/RealBokeh_3MP/resolve/"


def get(url,headers=None):
    for attempt in range(5):
        try:
            r = requests.get(url,headers=headers,timeout=(15,75))
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt==4:
                raise
            time.sleep(attempt+1)


class Remote(io.RawIOBase):
    def __init__(self,url):
        self.url = url
        self.pos = 0
        r = get(url,{"Range":"bytes=0-0","Accept-Encoding":"identity"})
        assert r.status_code==206
        self.size = int(r.headers["content-range"].split("/")[-1])
        self.directory = zipfile.ZipFile(self)

    def seek(self,offset,whence=0):
        self.pos = {0:0,1:self.pos,2:self.size}[whence]+offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self,n=-1):
        end = self.size if n<0 else min(self.size,self.pos+n)
        if end<=self.pos:
            return b""
        data = self.chunk(self.pos,end)
        self.pos = end
        return data

    def chunk(self,start,end):
        r = get(self.url,{"Range":f"bytes={start}-{end-1}","Accept-Encoding":"identity"})
        assert r.status_code==206 and r.headers["Content-Range"].startswith(f"bytes {start}-")
        return r.content

    def member(self,name):
        path = CACHE/"eth"/name
        info = self.directory.getinfo(name)
        if path.exists() and zlib.crc32(path.read_bytes())==info.CRC:
            return path
        buf = self.chunk(info.header_offset,min(self.size,info.header_offset+info.compress_size+2048))
        hdr = struct.unpack_from("<4s5H3I2H",buf)
        assert hdr[0]==b"PK\x03\x04"
        start = 30+hdr[-2]+hdr[-1]
        stream = buf[start:start+info.compress_size]
        data = zlib.decompress(stream,-15) if info.compress_type==8 else stream
        assert zlib.crc32(data)==info.CRC
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        return path


def sequence(scene):
    mono = Remote(ETH+scene+"_mono.zip")
    rgbd = Remote(ETH+scene+"_rgbd.zip")
    k = np.loadtxt(mono.member(scene+"/calibration.txt")).tolist()
    pairs = [s.split() for s in rgbd.member(scene+"/associated.txt").read_text().splitlines() if s and not s.startswith("#")]
    starts = [int((len(pairs)-4)*q) for q in (.15,.5,.85)]
    indices = [len(pairs)//2] if scene in DEV else [i+j for i in starts for j in range(4)]
    assert len(set(indices))==len(indices)
    rows = []
    for i in indices:
        t,r,td,d = pairs[i]
        assert abs(float(t)-float(td))<.002
        row = {"kind":"eth","scene":scene,"key":f"{scene}_{i}","index":i,"time":float(t),
               "fx":k[0],"image":"eth/"+scene+"/"+r,"depth":"eth/"+scene+"/"+d}
        mono.member(scene+"/"+r)
        rgbd.member(scene+"/"+d)
        rows.append(row)
    return rows


def optical(args):
    revision,scene = args
    def asset(name):
        path = CACHE/"realbokeh"/name
        if not path.exists():
            b = get(HF+revision+"/test/"+name).content
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b)
        return path
    m = json.loads(asset(f"metadata/{scene}.json").read_text())
    source = m["source_image"]
    target = m["target_images"][m["target_avs"].index(2)]
    for name in (source,target):
        asset(name)
    assert m["source_av"]==22 and m["focus_plane_distance"]>0
    return dict(kind="optical",scene=str(scene),key="optical_"+str(scene),
                image="realbokeh/"+source,target="realbokeh/"+target,
                focus=m["focus_plane_distance"],lens_mm=m["focal_length"])


def load_image(path,flag=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(CACHE/path,np.uint8),flag)
    assert img is not None
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if flag==cv2.IMREAD_COLOR else img


if __name__=="__main__":
    CACHE.mkdir(parents=True,exist_ok=True)
    revision = get("https://huggingface.co/api/datasets/timseizinger/RealBokeh_3MP").json()["sha"]
    names = get("https://huggingface.co/api/datasets/timseizinger/RealBokeh_3MP/tree/"+revision+"/test/metadata?limit=1000").json()
    ids = sorted(int(Path(n["path"]).stem) for n in names)
    chosen = [ids[i] for i in np.linspace(0,len(ids)-1,32,dtype=int)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        eth_jobs = [pool.submit(sequence,s) for s in (*DEV,*TEST)]
        photos = list(pool.map(optical,[(revision,s) for s in chosen]))
        rows = [row for job in eth_jobs for row in job.result()]+photos
    paths = {r[k] for r in rows for k in ("image","target","depth") if k in r}
    log = dict(revision=revision,rows=rows,sha256={n:hashlib.sha256((CACHE/n).read_bytes()).hexdigest() for n in sorted(paths)})
    (HERE/"inputs.json").write_text(json.dumps(log,indent=1),encoding="utf8")
    print("Selected frames",len(rows),"optical scenes",len(photos),flush=True)
