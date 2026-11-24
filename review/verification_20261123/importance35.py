"""Give the particle proposal correct mixture weights and validation-tuned noise."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import i0e, i1e, logsumexp, ndtri, softmax
from scipy.stats import qmc

from particle35 import solver

HERE = Path(__file__).resolve().parent
K = 256


def weighted_energy(x,w,ref):
    return float(2*(w@cdist(x,ref)).mean()-w@cdist(x,x)@w-cdist(ref,ref).mean())


def energy_to_population(x,w):
    """Exact Euclidean energy distance to the 2D isotropic Gaussian mixture (Rice means)."""
    def rice(d,sd):
        u = np.square(d/(2*sd))
        return np.sqrt(np.pi/2)*sd*((1+2*u)*i0e(u)+2*u*i1e(u))
    g = solver.G
    xy = w@rice(cdist(x,g.mu),g.s)@g.w
    reference = g.w@rice(cdist(g.mu,g.mu),np.hypot(g.s[:,None],g.s))@g.w
    return float(2*xy-w@cdist(x,x)@w-reference)


def all_steps(rng,n,tau,drift):
    x = rng.standard_normal((K,2))
    lw = np.full(K,-np.log(K))
    for i in range(n):
        mu = x+solver.G.velocity(x,i/n)/n if drift else x
        # Systematic allocation to components of the current weighted Gaussian mixture.
        cdf = np.cumsum(np.exp(lw))
        ids = np.searchsorted(cdf,(rng.random()+np.arange(K))/K).clip(max=K-1)
        variance = tau**2/n
        x = mu[ids]+np.sqrt(variance)*rng.standard_normal((K,2))
        logq = logsumexp(lw[None]-cdist(x,mu,"sqeuclidean")/(2*variance),axis=1)-np.log(2*np.pi*variance)
        ratios = solver.G.logpdf_t(x,(i+1)/n)-logq
        lw = ratios-logsumexp(ratios)
    return x,np.exp(lw)


def terminal(rng,sd):
    x = sd*rng.standard_normal((K,2))
    logq = -np.log(2*np.pi*sd**2)-np.square(x).sum(1)/(2*sd**2)
    return x,softmax(solver.G.logpdf_t(x,1.)-logq)


def draw(seed,method,n,parameter):
    rng = np.random.default_rng(seed)
    if method=="terminal":
        return terminal(rng,parameter)
    if method in ("mixture","density_only"):
        return all_steps(rng,n,parameter,drift=method=="mixture")
    if method=="heun_sobol":
        x = ndtri(qmc.Sobol(2,scramble=True,seed=seed).random_base2(8))
        steps = n//2
        for i in range(steps):
            start = solver.G.velocity(x,i/steps)
            end = solver.G.velocity(x+start/steps,(i+1)/steps)
            x = x+(start+end)/(2*steps)
        return x,np.full(K,1/K)
    x,_ = solver.heun(n,rng)
    return x,np.full(K,1/K)


def run(seed,method,n,parameter):
    ref = solver.G.sample(K,np.random.default_rng(seed+10000))
    x,w = draw(seed,method,n,parameter)
    return weighted_energy(x,w,ref)


if __name__ == "__main__":
    solver.K = K
    result = {}
    for n in (4,8,32,64):
        variants = [("heun",(0.,)),("heun_sobol",(0.,))]
        if n in (4,8):
            variants += [("terminal",(1.5,2.5,3.5)),("mixture",(.15,.4,1.)),("density_only",(.15,.4,1.))]
        for method,options in variants:
            best = min(options,key=lambda a:np.mean([run(s,method,n,a) for s in range(98200,98206)]))
            elapsed = []
            for repeat in range(17):
                start = perf_counter()
                draw(98400+repeat,method,n,best)
                elapsed.append(perf_counter()-start)
            result[f"{n}/{method}"] = dict(parameter=best,median_draw_ms=1000*np.median(elapsed),
                        validation_seeds=list(range(98200,98206)),
                        evaluations={s:run(s,method,n,best) for s in range(98300,98312)},
                        population_ed={s:energy_to_population(*draw(s,method,n,best)) for s in range(98300,98312)})
    (HERE/"importance35.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
