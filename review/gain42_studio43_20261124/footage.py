"""Deterministic video split, registered sensor depths and thin-object silhouette masks."""
import json
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from scenes import CACHE, tensor_input

ARCHIVE = CACHE / "ycbv_test_bop19.zip"
DEV = (48, 49)
TEST = tuple(range(50, 60))
THIN = {17, 18, 19, 20}
SIZE = (320, 240)


def decode(z, name, flag):
    image = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)
    if image is None:
        raise ValueError(name)
    return image


def narrow_boundary(mask):
    # Pad with background for the thickness calculation; crop margins from scoring.
    mask = np.pad(mask > 0, 2)
    core = distance_transform_edt(mask) >= 5
    far = distance_transform_edt(~core) > 7 if core.any() else np.ones(mask.shape, bool)
    rim = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)
    contour = rim & far
    if not contour.any():
        return np.zeros(mask.shape, bool)[2:-2,2:-2]
    band = (distance_transform_edt(~contour) <= 2)[2:-2,2:-2].copy()
    band[:2] = band[-2:] = False
    band[:,:2] = band[:,-2:] = False
    return band


def read(selection, archive=ARCHIVE, test_frames=12):
    rows = []
    with ZipFile(archive) as z:
        names = set(z.namelist())
        for scene in selection:
            prefix = f"test/{scene:06d}/"
            cameras = json.loads(z.read(prefix+"scene_camera.json"))
            gt = json.loads(z.read(prefix+"scene_gt.json"))
            ids = sorted(int(k) for k in cameras if prefix+f"rgb/{int(k):06d}.png" in names)
            indices = [len(ids)//2] if scene in DEV else np.linspace(0,len(ids)-1,test_frames,dtype=int)
            for index in indices:
                frame = ids[index]
                camera = cameras[str(frame)]
                im = decode(z,prefix+f"rgb/{frame:06d}.png",cv2.IMREAD_COLOR)
                fx = camera["cam_K"][0] * SIZE[0]/im.shape[1]
                rgb = cv2.cvtColor(cv2.resize(im,SIZE,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                raw = decode(z,prefix+f"depth/{frame:06d}.png",cv2.IMREAD_UNCHANGED)
                depth = cv2.resize(raw,SIZE,interpolation=cv2.INTER_NEAREST).astype(np.float32)
                depth *= camera["depth_scale"] / 1000
                x = tensor_input(rgb, fx)
                if scene in DEV:
                    iy,ix = np.indices((4,8))
                    y = ((iy+.5)*depth.shape[0]/4).astype(int).ravel()
                    col = ((ix+.5)*depth.shape[1]/8).astype(int).ravel()
                    val = depth[y,col]
                    ok = (val>=.2)&(val<=5)
                    rows.append((x,y[ok],col[ok],np.log(val[ok]),(scene,frame)))
                    continue
                band = np.zeros(depth.shape,bool)
                inside = band.copy()
                ids_with_mask = []
                for obj_idx, item in enumerate(gt[str(frame)]):
                    if item["obj_id"] not in THIN:
                        continue
                    path = prefix+f"mask_visib/{frame:06d}_{obj_idx:06d}.png"
                    mask = cv2.resize(decode(z,path,cv2.IMREAD_GRAYSCALE),SIZE,interpolation=cv2.INTER_NEAREST)>0
                    region = narrow_boundary(mask)
                    if region.any():
                        ids_with_mask.append(item["obj_id"])
                    band |= region
                    inside |= region & mask
                rows.append((x,depth,band,inside,scene,frame,ids_with_mask))
    return rows
