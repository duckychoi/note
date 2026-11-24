"""Check a ballistic product distribution against its exact law and against impulsive motion."""
import contextlib
import io
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]/"research-sources/physics-smc-generation/src/toy_physics_potential_smc.py"
with contextlib.redirect_stdout(io.StringIO()):
    toy = runpy.run_path(str(SRC))
S, T, SIG, G = (toy[k] for k in ("S","T","SIG","G"))
TIME = np.arange(T)
BASE = -0.5*G*TIME**2
LAMBDA = 2.0
EIG, V = np.linalg.eigh(S.T@S)
EIG = np.maximum(EIG,0)
FACTOR = 1 / np.sqrt(1 + 2*LAMBDA*SIG**2*EIG)
TARGET = SIG**2 * np.sum(EIG * FACTOR**2)


def noise_shrink(y, factors):
    return ((y-BASE)@V*factors)@V.T + BASE


def tempered_sample(y, rng):
    """pCN about a known unjittered flight preserves its Gaussian innovation prior."""
    h,v,eps = y
    beta = 0.
    for beta_next in np.linspace(0,LAMBDA,7)[1:]:
        energy = np.square(eps@S.T).sum(1)
        weights = np.exp(-(beta_next-beta)*(energy-energy.min()))
        idx = rng.choice(len(h), len(h), p=weights/weights.sum())
        eps,energy = eps[idx].copy(),energy[idx]
        h,v = h[idx],v[idx]
        for _ in range(8):
            fresh = np.sqrt(0.75)*eps + 0.5*SIG*rng.standard_normal(eps.shape)
            proposed = np.square(fresh@S.T).sum(1)
            accept = np.log(rng.random(len(eps))) < -beta_next*(proposed-energy)
            eps[accept],energy[accept] = fresh[accept],proposed[accept]
        beta = beta_next
    return BASE + h[:,None] + v[:,None]*TIME + eps


def moments(samples):
    curv = (samples-BASE)@S.T
    residual_target = SIG**2*(S@V*FACTOR**2)@(S@V).T
    error = np.linalg.norm(curv.T@curv/len(samples)-residual_target) / np.linalg.norm(residual_target)
    return dict(energy=float(np.square(curv).sum(1).mean()), curvature_second_moment_error=float(error),
                first_difference_sd=float(np.diff(samples,axis=1)[:,0].std()))


def fit_contact(y, base=BASE, sigma=SIG):
    """Least squares on all frames with BIC penalty and a search over impact time."""
    design = np.column_stack([np.ones(T),TIME])
    def fit(x):
        coef = (y-base)@np.linalg.pinv(x).T
        fitted = base+coef@x.T
        return fitted, np.square(fitted-y).sum(1), coef
    straight, best, _ = fit(design)
    selected = straight.copy()
    times = np.full(len(y),-1)
    for onset in range(5,19):
        candidate, error, coef = fit(np.column_stack((design,np.maximum(TIME-onset,0))))
        penalized = error + sigma**2*(np.log(T) + 2*np.log(14))
        take = (coef[:,-1]>0) & (penalized<best)
        selected[take],best[take],times[take] = candidate[take],penalized[take],onset
    return straight,selected,times


def event_test(rng, n):
    struck = rng.random(n)<0.15
    onset = rng.integers(5,19,n)
    true = BASE + rng.uniform(-1,1,(n,1)) + rng.uniform(.3,1.2,(n,1))*TIME
    true += struck[:,None]*rng.uniform(.45,.85,(n,1))*np.maximum(TIME-onset[:,None],0)
    observed = true + SIG*rng.standard_normal(true.shape)
    plain, segmented, detect = fit_contact(observed)
    estimates = {"observation":observed,"hard_ballistic":plain,"event_fit":segmented}
    return dict(n_events=int(struck.sum()), **{k:float(np.sqrt(np.square(x-true).mean())) for k,x in estimates.items()},
                event_recall=float((detect[struck]>=0).mean()),
                event_time_within_one=float((np.abs(detect[struck]-onset[struck])<=1).mean()),
                false_positive_rate=float((detect[~struck]>=0).mean()),
                hard_ballistic_event_rmse=float(np.sqrt(np.square(plain[struck]-true[struck]).mean())),
                event_fit_event_rmse=float(np.sqrt(np.square(segmented[struck]-true[struck]).mean())))


def main():
    rows = []
    for seed in range(9120,9132):
        rng = np.random.default_rng(seed)
        n = 2000
        h,v = rng.uniform(-1,1,n), rng.uniform(.3,1.2,n)
        eps = SIG*rng.standard_normal((n,T))
        raw = BASE + h[:,None]+v[:,None]*TIME + eps
        weights = np.exp(-LAMBDA*toy["E_phys"](raw))
        toy["phys_smc"].__globals__["rng"] = np.random.default_rng(seed+900)
        samples = dict(raw=raw, original_gradient=toy["phys_guide"](raw),
                       original_smc=toy["phys_smc"](raw), exact=noise_shrink(raw,FACTOR),
                       terminal_importance=raw[rng.choice(n,n,p=weights/weights.sum())],
                       tempered_pcn=tempered_sample((h,v,eps),rng))
        rows.append(dict(seed=seed, methods={k:moments(x) for k,x in samples.items()}, event=event_test(rng,n)))
    return dict(lambda_target=LAMBDA, analytical_mean_energy=TARGET, analytic_raw_energy=SIG**2*np.sum(EIG),
                seeds=rows)


if __name__ == "__main__":
    (HERE/"physics36.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
