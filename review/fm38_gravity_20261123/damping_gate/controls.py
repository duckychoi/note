"""Source-only low-dimensional alternatives to the latent residual flow."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from world import SPEED, PROBE_SPEED
from swing import TIMES, PROBE_TIME


def trajectory(q0,speed,k,c,times):
    q,v,stiff,drag=np.broadcast_arrays(q0,speed,k,c)
    q,v=q.copy(),v.copy()
    points=[]
    last=0.
    def accel(angle,rate):
        return -stiff*np.sin(angle)-drag*rate
    for end in times:
        steps=int(np.ceil((end-last)/.01))
        h=(end-last)/steps
        for _ in range(steps):
            a=accel(q,v)
            vb=v+h*a/2
            b=accel(q+h*v/2,vb)
            vc=v+h*b/2
            c3=accel(q+h*vb/2,vc)
            vd=v+h*c3
            d=accel(q+h*vc,vd)
            q=q+h*(v+2*vb+2*vc+vd)/6
            v=v+h*(a+2*b+2*c3+d)/6
        points.append(q)
        last=end
    return np.stack(points,-1)


def fitted_drag_path(data,c):
    start,after=data["observed"][:,:2].T
    starts,rev=np.unique(start,return_inverse=True)
    grid=np.linspace(2,32,301)
    curves=trajectory(starts[:,None],PROBE_SPEED,grid[None],c,[PROBE_TIME])[...,0]
    k=grid[abs(curves[rev]-after[:,None]).argmin(1)]
    return trajectory(start,SPEED[data["a"]],k,c,TIMES)


def fit_drag(train):
    grid=np.linspace(0,3,31)
    error=[float(abs(fitted_drag_path(train,c)-train["observed"][:,2:]).mean()) for c in grid]
    return float(grid[np.argmin(error)]),error


def features(data):
    return np.column_stack([data["observed"][:,:2],data["k"],SPEED[data["a"]],data["prior"]])


def fit_ridge(train):
    model=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),
                        StandardScaler(),Ridge(alpha=1.))
    model.fit(features(train),train["observed"][:,2:]-train["prior"])
    return model
