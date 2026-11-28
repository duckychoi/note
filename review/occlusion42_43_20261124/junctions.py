"""Z-buffered crossing surfaces. Appearance and left geometry are paired across order reversals."""
import cv2
import numpy as np

H,W=120,160
Y,X=np.indices((H,W),np.float32)


def color(rng):
    return .58*rng.uniform(35,220)+.42*rng.uniform(30,225,3)


def render(seed,split,base,swapped,lines,return_axes=False):
    rng=np.random.default_rng([seed,split,base,93011])
    fx=float(rng.uniform(.78,1.35)*W)
    baseline=rng.uniform(.035,.095)
    far=rng.uniform(2.3,3.6)
    bg=color(rng)
    light=rng.uniform(-.16,.16,2)
    stripe=rng.uniform(11,26)
    theta=rng.uniform(0,np.pi)
    pairs=[]
    for i,(cx,cy) in enumerate(((40,33),(117,33),(41,86),(116,86))):
        cx,cy=cx+rng.uniform(-5,5),cy+rng.uniform(-4,4)
        angle=rng.uniform(0,np.pi)
        angles=(angle,angle+rng.uniform(.90,2.15))
        depths=sorted(rng.uniform(.48,min(2.2,far-.12),2))
        if depths[1]-depths[0]<.14:
            depths[0]=max(.31,depths[1]-.16)
        exchange=int(rng.integers(2))^int(swapped)
        pair=[]
        for j in (0,1):
            pair.append((cx,cy,angles[j],rng.uniform(16,24),rng.uniform(2.1,7.8),color(rng),
                         rng.uniform(-.22,.22),rng.uniform(.12,.48),rng.uniform(-3,3),
                         rng.choice([-1,1])*rng.uniform(25,85),rng.uniform(.65,1.7),depths[j^exchange],2*i+j))
        pairs.append(pair)
    frames=[]
    bg_amp=rng.choice([-1,1])*rng.uniform(28,65)
    for side,cam in enumerate((0,baseline)):
        xx=X+fx*cam/far
        illum=1+light[0]*(xx/W-.5)+light[1]*(Y/H-.5)
        coord=xx*np.cos(theta)+Y*np.sin(theta)
        bgline=np.clip(1.3-abs((coord+stripe/2)%stripe-stripe/2),0,1) if lines else 0
        rgb=(illum[...,None]*bg+bg_amp*np.asarray(bgline)[...,None]).astype(np.float32)
        depth=np.full((H,W),far,np.float32)
        label=np.full((H,W),-1,np.int16)
        masks={}
        near_center={}
        for pair in pairs:
            for cx,cy,ang,length,width,c,texture,freq,phase,ink,ink_width,z,idx in pair:
                dx,dy=X+fx*cam/z-cx,Y-cy
                u=np.cos(ang)*dx+np.sin(ang)*dy
                v=-np.sin(ang)*dx+np.cos(ang)*dy
                inside=(abs(u)<length)&(abs(v)<width)
                masks[idx]=inside
                near_center[idx]=(abs(u)<.78*length)&(abs(v)<.70*width)
                take=inside&(depth>z)
                tint=1+texture*np.sin(freq*u[take]+phase)*.36
                marking=np.clip(ink_width+.5-abs(v[take]),0,1) if lines else np.zeros(take.sum())
                rgb[take]=tint[:,None]*c+ink*marking[:,None]
                depth[take]=z
                label[take]=idx
        noise=np.random.default_rng([seed,split,base,side,12039]).normal(0,2,rgb.shape)
        rgb=np.uint8(np.clip(rgb+noise,0,255))
        ids=[]
        for i in range(len(pairs)):
            a,b=2*i,2*i+1
            points=[np.flatnonzero(near_center[k]&(label==k)&~masks[other]) for k,other in ((a,b),(b,a))]
            middle=np.flatnonzero(masks[a]&masks[b]&((label==a)|(label==b)))
            if not all(len(p) for p in (*points,middle)):
                continue
            qrng=np.random.default_rng([seed,split,base,side,i,437])
            p,q=[qrng.choice(k,12,replace=True) for k in points]
            ids.extend(zip(p,q,[True]*12))
            mid=qrng.choice(middle,12,replace=True)
            behind=np.where(label.ravel()[mid]==a,q,p)
            ids.extend(zip(mid,behind,[False]*12))
        frames.append((rgb,depth,np.asarray(ids,dtype=np.int32).reshape(-1,3)))
    if return_axes:
        # Locations and directions only, blind to the order and both metric depths.
        axes=[(p[0][0],p[0][1],p[0][2],p[1][2]) for p in pairs]
        return frames,fx,float(baseline),axes
    return frames,fx,float(baseline)
