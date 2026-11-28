"""Read the official split ZIP by validated HTTP ranges without downloading 75 GB."""
import hashlib
import json
import re
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[2]/"research-sources/gain42_video"
BASE = "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/5c2c4aa229800355648cd268040aa814f8dc94f0/"
FILES = ("ycbv_train_real.z01","ycbv_train_real.zip")
SIZES = (48318382080,27404732247)


def block(disk, offset, size):
    if offset+size > SIZES[disk]:
        first=SIZES[disk]-offset
        return block(disk,offset,first)+block(disk+1,0,size-first)
    for trial in range(4):
        try:
            r=requests.get(BASE+FILES[disk]+f"?g42block={offset}-{size}",
                           headers={"Range":f"bytes={offset}-{offset+size-1}","Accept-Encoding":"identity"},
                           timeout=(15,80))
            r.raise_for_status()
            assert r.status_code==206 and len(r.content)==size
            assert r.headers["Content-Range"].startswith(f"bytes {offset}-")
            return r.content
        except (requests.RequestException,AssertionError):
            if trial==3:
                raise
            time.sleep(1+trial)


def directory():
    tail=block(1,SIZES[1]-128,128)
    record=struct.unpack_from("<4sQ2H2L4Q",tail,tail.index(b"PK\x06\x06"))
    size,start=record[-2:]
    folder=CACHE/"central"
    folder.mkdir(exist_ok=True)
    def part(offset):
        n=min(4<<20,size-offset)
        path=folder/f"{offset}.bin"
        if not path.exists() or path.stat().st_size!=n:
            path.write_bytes(block(1,start+offset,n))
        return path.read_bytes()
    with ThreadPoolExecutor(8) as pool:
        data=b"".join(pool.map(part,range(0,size,4<<20)))
    pos=0
    members={}
    while pos<len(data):
        h=struct.unpack_from(zipfile.structCentralDir,data,pos)
        assert h[0]==b"PK\x01\x02"
        pos+=zipfile.sizeCentralDir
        name=data[pos:pos+h[12]].decode("utf8")
        extra=data[pos+h[12]:pos+h[12]+h[13]]
        pos+=sum(h[12:15])
        if not re.search(r"(?:^|/)(?:00000[0-9]|00001[0-9]|00002[0-3])/",name):
            continue
        values=[h[11],h[10],h[18],h[15]]
        while len(extra)>=4:
            tag,length=struct.unpack_from("<HH",extra)
            field,extra=extra[4:4+length],extra[4+length:]
            if tag==1:
                for j,limit in enumerate((0xffffffff,0xffffffff,0xffffffff,0xffff)):
                    if values[j]==limit:
                        width=8 if j<3 else 4
                        values[j]=int.from_bytes(field[:width],"little")
                        field=field[width:]
        members[name]=(values[3],values[2],values[1],h[9])
    return members


def payload(index,name):
    disk,pos,n,crc=index[name]
    # Local file metadata and payload are small for each requested member.
    buf=block(disk,pos,min(2048+n,SIZES[disk]-pos))
    hdr=struct.unpack_from("<4s5H3I2H",buf)
    assert hdr[0]==b"PK\x03\x04"
    start=30+hdr[-2]+hdr[-1]
    data=buf[start:start+n]
    if len(data)!=n:
        data=block(disk,pos+start,n)
    raw=zlib.decompress(data,-15) if hdr[3]==8 else data
    assert zlib.crc32(raw)==crc
    return raw


def main():
    index=directory()
    gtfiles=sorted(n for n in index if n.endswith("/scene_gt.json"))
    assert len(gtfiles)==24, gtfiles[:3]
    chosen=[]
    entries={}
    names=[]
    for path in gtfiles:
        prefix=path[:-len("scene_gt.json")]
        scene=int(prefix.rstrip("/").split("/")[-1])
        raw=payload(index,path)
        gt=json.loads(raw)
        ids=sorted(map(int,gt))
        if not any(17<=obj["obj_id"]<=20 for obj in gt[str(ids[len(ids)//2])]):
            continue
        chosen.append(scene)
        indices=[ids[i] for i in np.linspace(0,len(ids)-1,8,dtype=int)]
        cam=json.loads(payload(index,prefix+"scene_camera.json"))
        dest=f"test/{scene:06d}/"
        for name,meta in (("scene_gt.json",gt),("scene_camera.json",cam)):
            entries[dest+name]=json.dumps({str(i):meta[str(i)] for i in indices}).encode()
        for i in indices:
            for sub in ("rgb","depth"):
                names.append((dest+f"{sub}/{i:06d}.png",prefix+f"{sub}/{i:06d}.png"))
            for j,obj in enumerate(gt[str(i)]):
                if 17<=obj["obj_id"]<=20:
                    s=f"mask_visib/{i:06d}_{j:06d}.png"
                    names.append((dest+s,prefix+s))
        if len(chosen)==8:
            break
    assert chosen
    def retrieve(item):
        dest,src=item
        return dest,payload(index,src)
    with ThreadPoolExecutor(12) as pool:
        entries.update(pool.map(retrieve,names))
    with zipfile.ZipFile(CACHE/"supplement.zip","w") as out:
        for n,b in sorted(entries.items()):
            out.writestr(n,b)
    meta={"scenes":chosen,"files":{n:hashlib.sha256(b).hexdigest() for n,b in sorted(entries.items())}}
    (HERE/"extra_manifest.json").write_text(json.dumps(meta,indent=1),encoding="utf8")
    print(chosen,len(entries),flush=True)


if __name__=="__main__":
    main()
