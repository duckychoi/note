"""Virtual repeated exposures of the measured Middlebury surfaces, with a BRDF intervention."""
import argparse
import json
import zlib

import cv2
import numpy as np
from PIL import Image

from repeat_capture_acquire import OUT

FOCAL = 3740/3
BASELINE = .16
LOG_MIN = .02
VIEWS = (.5,1.)  # Rectified indices 3 and 5, with index 1 as reference.


def load_gray(path):
    rgb = np.array(Image.open(path).convert("RGB"),np.float32)/255
    lin = np.where(rgb <= .04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return lin @ np.array([.2126,.7152,.0722],np.float32)


def sample(image, x):
    y = np.broadcast_to(np.arange(x.shape[0],dtype=np.float32)[:,None],x.shape)
    return cv2.remap(image.astype(np.float32),x.astype(np.float32),y,
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def unit(vec):
    return vec / np.maximum(np.linalg.norm(vec,axis=-1,keepdims=True),1e-10)


def geometry(src):
    disp = np.asarray(Image.open(src/"disp1.png"),np.float32)/3
    right = np.asarray(Image.open(src/"disp5.png"),np.float32)/3
    dmin = float((src/"dmin.txt").read_text())/3
    h,w = disp.shape
    y,x = np.indices(disp.shape,dtype=np.float32)
    inv = (disp+dmin)/(FOCAL*BASELINE)
    z = 1/inv
    xyz = np.stack([(x-(w-1)/2)*z/FOCAL,(y-(h-1)/2)*z/FOCAL,z],axis=-1)
    normal = unit(np.cross(np.gradient(xyz,axis=1),np.gradient(xyz,axis=0)))
    normal *= np.where((normal*xyz).sum(-1)>0,-1.,1.)[:,:,None]
    good = (disp>0) & (x-disp>=2) & (abs(sample(right,x-disp)-disp)<.8)
    local_range = cv2.dilate(disp,np.ones((3,3),np.uint8))-cv2.erode(disp,np.ones((3,3),np.uint8))
    good &= local_range < 2
    return x,disp,dmin,inv,xyz,normal,good


def raster_correspondence(x,disp,fraction):
    # Scanline rasterization of adjacent vertices, dropping triangles across discontinuities.
    h,w=x.shape
    left=(x-fraction*disp)[:,:-1]
    span=1-fraction*np.diff(disp,axis=1)
    valid=(disp[:,:-1]>0)&(disp[:,1:]>0)&(abs(np.diff(disp,axis=1))<2)&(span>1e-4)
    y,src_x=np.indices(left.shape)
    zbuf=np.full(h*w,-np.inf,np.float32)
    samples=[]
    for offset in range(4):  # valid span is strictly below 3 pixels
        dest=np.ceil(left).astype(int)+offset
        a=(dest-left)/np.maximum(span,1e-4)
        m=valid&(dest>=0)&(dest<w)&(a>=0)&(a<=1)
        index=(y*w+dest)[m]
        depth=(disp[:,:-1]+a*np.diff(disp,axis=1))[m].astype(np.float32)
        np.maximum.at(zbuf,index,depth)
        samples.append((index,depth,(src_x+a)[m]))
    mapping=np.full(h*w,-100,np.float32)
    for index,depth,origin in samples:
        use=depth>=zbuf[index]
        mapping[index[use]]=origin[use]
    return mapping.reshape(h,w)


def plane_normals(disp,inv,xyz):
    # Estimate the slope over several quantized disparities while excluding jumps.
    slopes=[]
    for axis in (1,0):
        numerator=np.zeros_like(inv)
        denominator=np.zeros_like(inv)
        for t in (-4,-3,-2,-1,1,2,3,4):
            other=np.roll(disp,-t,axis=axis)
            valid=(disp>0)&(other>0)&(np.abs(other-disp)<=2)
            edge=[slice(None),slice(None)]
            edge[axis]=slice(-t,None) if t>0 else slice(None,-t)
            valid[tuple(edge)]=False
            w=np.exp(-.5*(t/2.5)**2)*valid
            numerator+=w*t*(other-disp)/(FOCAL*BASELINE)
            denominator+=w*t*t
        slopes.append(-numerator/(np.maximum(denominator,1e-8)*inv**2))
    rays=xyz*inv[:,:,None]
    dx=rays*slopes[0][:,:,None]
    dy=rays*slopes[1][:,:,None]
    dx[:,:,0]+=1/(inv*FOCAL)
    dy[:,:,1]+=1/(inv*FOCAL)
    normal=unit(np.cross(dx,dy))
    return normal*np.where((normal*xyz).sum(-1)>0,-1.,1.)[:,:,None]


def main(smooth=False):
    cv2.setNumThreads(1)
    folder=OUT/("renders-plane" if smooth else "renders")
    folder.mkdir(exist_ok=True)
    for src in sorted((OUT/"physical").iterdir()):
        x,disp,dmin,inv,xyz,normal,good=geometry(src)
        if smooth:
            normal=plane_normals(disp,inv,xyz)
        albedo=.12+.65*np.sqrt(load_gray(src/"Illum1/Exp1/view1.png"))
        dist=float(np.median(xyz[...,2][good]))
        lamps=dist*np.array([[-.36,-.30,.30],[.38,.12,.32]])
        maps=[raster_correspondence(x,disp,v) for v in VIEWS]
        support=[mapping>=0 for mapping in maps]
        means=np.empty((2,2,2,*x.shape),np.float32)
        surface_log=np.empty_like(means)
        for view,(fraction,mapping) in enumerate(zip(VIEWS,maps)):
            camera=np.array([fraction*BASELINE,0.,0.])
            outgoing=unit(camera-xyz)
            for light,lamp in enumerate(lamps):
                toward=lamp-xyz
                beam=unit(toward)
                irradiance=np.minimum(1.1,.48*dist**2/np.maximum((toward**2).sum(-1),.01))
                diffuse=.035+albedo*(.10+irradiance*np.maximum(0,(normal*beam).sum(-1)))
                halfway=unit(beam+outgoing)
                spec=.48*irradiance*np.maximum(0,(normal*halfway).sum(-1))**36
                for material,signal in enumerate((diffuse,diffuse+spec)):
                    means[material,view,light]=np.clip(sample(signal,mapping),0,1)
                    surface_log[material,view,light]=np.log(np.clip(signal,LOG_MIN,1))
        frames=np.empty((2,2,2,2,*x.shape),np.uint16)
        for mat,rep,v,l in np.ndindex(2,2,2,2):
            seed=[20261126,zlib.crc32(src.name.encode()),mat,rep,v,l]
            rng=np.random.default_rng(np.random.SeedSequence(seed))
            mean=means[mat,v,l]
            signal=(rng.poisson(mean*4096)+rng.normal(0,2,mean.shape))/4096
            frames[mat,rep,v,l]=np.rint(65535*np.clip(signal,0,1)).astype(np.uint16)
        ideal_cycle=(surface_log[:,1,0]-surface_log[:,1,1])-(surface_log[:,0,0]-surface_log[:,0,1])
        np.savez_compressed(folder/(src.name+".npz"),frames=frames,expected=means,
                            valid=np.stack(support),source_x=np.stack(maps),ideal_cycle=ideal_cycle,lights=lamps)
    (folder/"capture.json").write_text(json.dumps(dict(source="Measured Middlebury disparity surface, textured with a photograph",
        materials=["diffuse","diffuse plus view-dependent Blinn-Phong"],views=[3,5],lights=2,
        anchor_view=1,repeats=2,shape="material,repeat,view,light,height,width",
        units="linear intensity",levels=65535,shot_electrons=4096,read_noise_electrons=2,
        normals="local slope radius 4" if smooth else "central differences",
        limitations="Single visible surface reconstruction, direct illumination, no cast shadows or indirect transport. Texture includes baked illumination.",
        rng="PCG64 SeedSequence(20261126,crc32(scene),material,repeat,view,light)"),indent=1),encoding="utf8")
    print("Rendered",len(list(folder.glob("*.npz"))),"surfaces",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-normals",action="store_true")
    main(parser.parse_args().smooth_normals)
