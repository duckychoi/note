"""Contrast marginal, transition, and event predictions in controlled planning problems."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
PATHS = np.array(list(itertools.product((-1,1),repeat=4)))
SAME = PATHS[:,1:] == PATHS[:,:-1]
EVENT = SAME.all(1)
UNSEEN = PATHS[:,0] == PATHS[:,-1]
FEATURES = np.column_stack([SAME,EVENT]).astype(float)


def utility_regret(pred, truth, safe):
    return float(max(safe,truth) - (truth if pred>safe else safe))


def simple_world(rng, stay, n=1024):
    training = rng.random((n,3)) < stay
    est = training.mean()
    out = {}
    for h in (4,8,16):
        truth = stay**(h-1)
        guesses = dict(point_mean=0.,independent=.5**(h-1),fixed_branch=1.,learned_transition=est**(h-1))
        # A gap-valued point prediction has no valid same-sign path through the corridor.
        out[h] = {k:dict(event_p=float(v),true_p=truth,mean_regret=float(np.mean(
            [utility_regret(v,truth,c) for c in (.1,.4,.7)]))) for k,v in guesses.items()}
    return out


def all_starts(string):
    cycle = np.array([1 if ch=="+" else -1 for ch in string])
    return cycle[(np.arange(len(cycle))[:,None]+np.arange(4))%len(cycle)]


def project(base, rho, event_rate):
    """KL projection onto per-edge staying rates and one observed path event rate."""
    # Separate finite-sample estimates can violate the Frechet bounds. Project
    # the event frequency to the feasible interval, then smooth towards the base.
    bounded = np.clip(event_rate,max(0.,3*rho-2),rho)
    feasible_rate = .999*bounded + .001*(base@EVENT)
    targets = np.array([rho,rho,rho,feasible_rate])
    logbase = np.log(base)
    def probs(theta):
        return softmax(logbase + FEATURES@theta)
    fit = least_squares(lambda theta:probs(theta)@FEATURES-targets,np.zeros(4),
                        ftol=1e-14,xtol=1e-14,gtol=1e-14)
    result = probs(fit.x)
    assert np.max(np.abs(result@FEATURES-targets)) < 1e-7
    return result


def matched_pair(rng, n=1024):
    worlds = {"regular":all_starts("+++---"), "uneven":all_starts("+-----+++++-")}
    # Each observation can be multiplied by an independent half-normal magnitude.
    # This gives exact standard-normal marginals and identical adjacent pair laws.
    training = {k:clip[rng.integers(len(clip),size=n)] for k,clip in worlds.items()}
    rho = float(np.mean([np.mean(x[:,1:]==x[:,:-1]) for x in training.values()]))
    base = 0.5*np.prod(np.where(SAME,rho,1-rho),axis=1)
    rows = {}
    for name, clips in worlds.items():
        x = training[name]
        rate = (np.sum((x[:,1:]==x[:,:-1]).all(1))+.5)/(n+1)
        fitted = project(base,rho,rate)
        direct = np.mean(np.all(x[:,None] == PATHS,axis=-1),axis=0)
        truth = float(np.mean((clips[:,1:]==clips[:,:-1]).all(1)))
        unseen_truth = float(np.mean(clips[:,0]==clips[:,-1]))
        rows[name] = {}
        for method,q in [("markov_one",base),("event_projection",fitted),("four_frame_counts",direct)]:
            p = float(q@EVENT)
            rows[name][method] = dict(event_p=p,true_event_p=truth,
                regret=utility_regret(p,truth,.2), second_task_p=float(q@UNSEEN),
                second_task_true=unseen_truth, max_edge_change=float(np.max(np.abs(q@SAME-rho))))
    return rows


def rotation(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])


def position_control():
    q,k = np.array([.4,.9]),np.array([-.3,.8])
    def score(i,j):
        return float((rotation(i*.4)@q)@(rotation(j*.4)@k))
    error = {}
    for horizon in (1,8,32,64):
        true = rotation(horizon*.4)@np.array([1.,0.])
        pred = rotation(horizon*.42)@np.array([1.,0.])
        error[horizon] = float(np.linalg.norm(true-pred))
    return dict(shift_dot_error=abs(score(3,7)-score(103,107)),rollout_error=error)


if __name__ == "__main__":
    rows = []
    for seed in range(11120,11132):
        rng = np.random.default_rng(seed)
        rows.append(dict(seed=seed, worlds={stay:simple_world(rng,stay) for stay in (.15,.5,.97)},
                         matched=matched_pair(rng)))
    result = dict(seeds=rows,rotation_control=position_control())
    (HERE/"planning38.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
