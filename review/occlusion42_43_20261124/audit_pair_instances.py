"""Check visible-object identity and stereo depth accuracy after pair selection is frozen."""
import ctypes
import hashlib
import json
from zipfile import ZipFile

import cv2
import h5py
import numpy as np

from controlled_observations import RUN, wait_for_ram, warp_geometry, write_json
from depth_full_data import Hypersim
from hypersim_fetch import ARCHIVE,RemoteZip
from select_stereo_pairs import OUT
from stereo_controlled_observations import rectification,stereo_pair


def labels(scene,camera,frames):
    folder=OUT/"instance-ids"/scene/camera
    folder.mkdir(parents=True,exist_ok=True)
    paths={frame:folder/f"{frame:04d}.semantic_instance.hdf5" for frame in set(frames)}
    missing={frame:p for frame,p in paths.items() if not p.exists()}
    if missing:
        with RemoteZip(f"{ARCHIVE}/{scene}.zip") as reader, ZipFile(reader) as archive:
            for frame,path in missing.items():
                member=f"{scene}/images/scene_{camera}_geometry_hdf5/frame.{frame:04d}.semantic_instance.hdf5"
                data=archive.read(member)
                temp=path.with_suffix(".partial")
                temp.write_bytes(data)
                temp.replace(path)
    arrays={}
    for frame,path in paths.items():
        with h5py.File(path) as source:
            arrays[frame]=source["dataset"][:]
    return arrays,{str(f):hashlib.sha256(p.read_bytes()).hexdigest() for f,p in paths.items()}


def measurements(data,row,ids,pred):
    a,b,pose=data.read_pair(row)
    xy,_,visible=warp_geometry(a,b,pose)
    other_id=ids[row[3]]
    anchor_id=ids[row[2]][::8,::8]
    valid=(anchor_id>0)&a[2][::2,::2]
    px=np.rint(4*xy[...,0]).clip(0,other_id.shape[1]-1).astype(int)
    py=np.rint(4*xy[...,1]).clip(0,other_id.shape[0]-1).astype(int)
    same=valid&visible&(anchor_id==other_id[py,px])
    z=a[1][::2,::2]
    geometry=rectification((a,b),pose)
    labels_checked=accurate=supported=0
    if geometry is not None:
        lut=np.rint(255*(np.arange(256)/255)**1.4).astype(np.uint8)
        lights=[[i[0],lut[i[0]]] for i in (a,b)]
        for light in (0,1):
            both=[stereo_pair([lights[0][light],neighbor],geometry)[0] for neighbor in lights[1]]
            common=valid&(pred[light]>=.25)&(pred[light]<=30)&both[0][1]&both[1][1]
            supported+=int(common.sum())
            labels_checked+=int((common&same).sum())
            accurate+=int((common&((np.abs(both[0][0]/z-1)<=.1)&(np.abs(both[1][0]/z-1)<=.1))).sum())
    return dict(labeled_valid=int(valid.sum()),same_instance_visible=int(same.sum()),
                supported_labeled=supported,supported_same_instance_visible=labels_checked,
                both_stereo_depths_within_10pct=accurate)


def main():
    cv2.setNumThreads(1)
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x4000)
    data=Hypersim("validation_24",4,True)
    rows=[]
    for file in sorted(RUN.glob("dev-*.json")):
        old=json.loads(file.read_text())["row"]
        scene,c,anchor,before=old
        new=json.loads((OUT/f"pairs-dense-{scene}.json").read_text())["row"]
        receipt=OUT/f"instances-{scene}.json"
        if not receipt.exists():
            wait_for_ram()
            ids,hashes=labels(scene,c,(anchor,before,new[3]))
            prior=np.load(RUN/"predictions"/f"dev-{scene}.npy",mmap_mode="r")[0,:,::2,::2]
            counts=[measurements(data,row,ids,prior) for row in (old,new)]
            if counts[0]["labeled_valid"]!=counts[1]["labeled_valid"]:
                raise ValueError("Changed anchor mask")
            write_json(receipt,dict(scene=scene,rows=[old,new],sha256=hashes,counts=counts))
        rows.append(json.loads(receipt.read_text()))
    if len(rows)!=46:
        raise ValueError("Missing scene")
    summary={}
    for i,method in enumerate(("original","all_trajectory")):
        counts=[row["counts"][i] for row in rows]
        totals={key:sum(c[key] for c in counts) for key in counts[0]}
        labeled=[v for v in counts if v["labeled_valid"]]
        summary[method]=dict(totals=totals,labeled_scenes=len(labeled),
            no_labeled_scene_ids=[row["scene"] for row in rows if not row["counts"][i]["labeled_valid"]],
            scene_mean_visible_same_instance_pct=100*np.mean(
            [v["same_instance_visible"]/v["labeled_valid"] for v in labeled]),
            both_depths_accurate_within_supported_pct=100*totals["both_stereo_depths_within_10pct"]/totals["supported_labeled"])
    write_json(OUT/"instance_audit.json",dict(summary=summary,rows=rows))
    print(summary)


if __name__=="__main__":
    main()
