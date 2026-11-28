"""Fetch named members of the authors' public Chrome Screw ZIP, verifying ZIP CRCs."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import zipfile
import zlib
import requests

HERE=Path(__file__).resolve().parent
CACHE=HERE.parents[2]/"research-sources"/"physical43_20261124"
OBJECT=os.environ.get("ROBI_OBJECT","Chrome_screw")
ARCHIVES={"Chrome_screw":"1qqG0KVW6_Upxc95Wuk9HecR0JrxGZSd4","Gear":"1qWcz7AD_9JwwFGskUfDFUirw0KOiGJ75"}
if OBJECT!="Chrome_screw":
    CACHE=CACHE/OBJECT
URL=f"https://drive.usercontent.google.com/download?id={ARCHIVES[OBJECT]}&export=download&confirm=t"
CAL="https://raw.githubusercontent.com/junyang224/ROBI_Toolkit/main/cam/"


def output_file(filename):
    return HERE/(filename if OBJECT=="Chrome_screw" else OBJECT.lower()+"_"+filename)


def byte_range(begin,end):
    for attempt in range(4):
        try:
            with requests.get(URL,headers={"Range":f"bytes={begin}-{end}","Accept-Encoding":"identity"},
                              timeout=(12,45),stream=True) as r:
                r.raise_for_status()
                if r.status_code!=206 or not r.headers["Content-Range"].startswith(f"bytes {begin}-"):
                    raise IOError("source did not serve the requested byte range")
                return r.content,int(r.headers["Content-Range"].split("/")[-1])
        except (requests.RequestException,IOError):
            if attempt==3:
                raise


class MetadataReader(io.RawIOBase):
    def __init__(self):
        self.position=0
        self.size=byte_range(0,0)[1]

    def seek(self,offset,whence=0):
        self.position=({0:0,1:self.position,2:self.size}[whence]+offset)
        if self.position<0:
            raise OSError("negative offset")
        return self.position

    def tell(self):
        return self.position

    def read(self,n=-1):
        end=self.size if n<0 else min(self.size,self.position+n)
        if end<=self.position:
            return b""
        data=byte_range(self.position,end-1)[0]
        self.position+=len(data)
        return data


@lru_cache(maxsize=1)
def index():
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/"directory.json"
    if not path.exists():
        with zipfile.ZipFile(MetadataReader()) as z:
            items={i.filename:[i.header_offset,i.compress_size,i.CRC] for i in z.infolist() if not i.is_dir()}
        path.write_text(json.dumps(items),encoding="utf8")
    return json.loads(path.read_text(encoding="utf8"))


def cached(name):
    path=CACHE/name
    offset,length,crc=index()[name]
    if path.exists() and zlib.crc32(path.read_bytes())==crc:
        return path
    buf=byte_range(offset,offset+length+1023)[0]
    h=struct.unpack_from("<4s5H3I2H",buf)
    assert h[0]==b"PK\x03\x04"
    start=30+h[-2]+h[-1]
    stream=buf[start:start+length]
    if len(stream)!=length:
        stream=byte_range(offset+start,offset+start+length-1)[0]
    data=zlib.decompress(stream,-15) if h[3]==8 else stream
    assert zlib.crc32(data)==crc
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return path


def views(scene,sensor,kind):
    pattern=rf".*/Scene_{scene}/{re.escape(kind)}/DEPTH_view_(\d+)\.png$"
    return {int(m[1]):name for name in index() if (m:=re.fullmatch(pattern,name,re.I))}


SENSORS=("Ensenso","RealSense")
SCENES=(4,5,8,9)


def selection():
    out=[]
    for scene in SCENES:
        for sensor in SENSORS:
            raw=views(scene,sensor,sensor+"/Depth")
            gt={int(m[1]):n for n in index() if (m:=re.fullmatch(
                rf".*/Scene_{scene}/GT_Depth/(?:DEPTH_)?view_(\d+)\.png",n,re.I))}
            ids=sorted(raw)
            for k in (len(ids)//4,len(ids)//2):
                first,second,hold=ids[k],ids[k+3],sorted(gt)[3*len(gt)//4]
                gt_ids=[min(gt,key=lambda v:(abs(v-i),v)) for i in (first,second,hold)]
                out.append(dict(scene=scene,sensor=sensor,ids=[first,second,hold],
                                raw=[raw[first],raw[second]],gt=[gt[i] for i in gt_ids]))
    return out


def prepare():
    rows=selection()
    paths=set()
    for row in rows:
        for name in row["raw"]+row["gt"]:
            paths.update((name,name[:-3]+"json"))
        root=row["raw"][0].split("/Depth/")[0]
        for side in ("LEFT","RIGHT"):
            paths.add(f"{root}/StereoPattern/{side}_view_{row['ids'][0]}.bmp")
        other=next(s for s in SENSORS if s!=row["sensor"])
        alternate=row["raw"][0].replace("/"+row["sensor"]+"/","/"+other+"/")
        paths.update((alternate,alternate[:-3]+"json"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(cached,sorted(paths)))
    calibrations=[s+"_"+v+".yml" for s in SENSORS for v in ("DEPTH","LEFT","RIGHT")]
    for filename in calibrations:
        f=CACHE/filename
        if not f.exists():
            response=requests.get(CAL+f.name,timeout=20)
            response.raise_for_status()
            f.write_bytes(response.content)
    manifest={"archive_url":URL,"file_count_in_archive":len(index()),"selection":rows,
              "sha256":{n:hashlib.sha256((CACHE/n).read_bytes()).hexdigest() for n in sorted(paths)},
              "calibration_sha256":{n:hashlib.sha256((CACHE/n).read_bytes()).hexdigest() for n in calibrations}}
    output_file("robi_inputs.json").write_text(json.dumps(manifest,indent=1),encoding="utf8")
    print("downloaded",len(paths),"archive members")


if __name__=="__main__":
    ix=index()
    print(len(ix),"members")
    for scene in (4,5,8,9):
        print(scene,[k for k in ix if f"/Scene_{scene}/" in k][:3])
    print("folders:",sorted({re.sub(r"/[^/]+$","",k).split("/Scene_4/")[-1] for k in ix if "/Scene_4/" in k})[:65])
    print("timing names:",[n for n in ix if re.search(r"time|clock|command|trigger|latency|\.bag$|\.log$",n,re.I)][:20])
    prepare()
