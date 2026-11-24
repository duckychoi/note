"""Exploratory follow-up: infer height, launch velocity and restitution with unknown contacts."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.special import softmax

from contact36 import BASE, DT, SIGMA, TIME, fit_contact, model, rmse, trajectories

HERE = Path(__file__).resolve().parent


def event_bank():
    # Uniform grid prior. Closed-form flights and instantaneous bounces on a known floor.
    heights,velocity,restitution = [x.ravel() for x in np.meshgrid(
        np.linspace(.32,1.72,71),np.linspace(-1.15,.5,34),np.linspace(0,.9,10),indexing="ij")]
    g, floor = 9.81, .06
    initial_impact_speed = np.sqrt(velocity**2+2*g*(heights-floor))
    first = (velocity+initial_impact_speed)/g
    result = heights[:,None]+velocity[:,None]*(DT*TIME)+BASE
    for j,t in enumerate(DT*TIME):
        touched = t>=first
        elapsed = (t-first).clip(min=0)
        result[touched,j] = floor
        # After infinitely many rebounds the total remaining flight time is finite.
        stop_time = (2/g)*initial_impact_speed*restitution/(1-restitution)
        flying = touched & (elapsed < stop_time)
        r = restitution[flying]
        ratio = elapsed[flying]/stop_time[flying]
        bounce = np.floor(np.log1p(-ratio)/np.log(r)).astype(int)
        since = elapsed[flying] - stop_time[flying]*(1-r**bounce)
        speed = initial_impact_speed[flying]*r**(bounce+1)
        result[flying,j] = floor + speed*since - .5*g*since**2
    return result


def prediction(bank,observed,temperature):
    # Distances are computed with inner products; constants depending only on y cancel.
    logits = (observed@bank.T - .5*(bank**2).sum(1))/(SIGMA**2*temperature)
    weights = softmax(logits,axis=1)
    mean = weights@bank
    return mean,bank[logits.argmax(1)]


def main():
    bank = event_bank()
    material_sets = {"within_materials":(.18,.4,.75),"new_materials":(.12,.55,.95)}
    models = {k:[model(d) for d in v] for k,v in material_sets.items()}
    validation = [trajectories(models["within_materials"],s) for s in (31000,31001)]
    temp_grid = (.75,1.,2.,4.)
    temperature = min(temp_grid,key=lambda temp:np.mean(
        [rmse(y,prediction(bank,obs,temp)[0]) for y,obs,_ in validation]))
    width = min((5,7,9,11),key=lambda w:np.mean([rmse(y,savgol_filter(obs,w,2)) for y,obs,_ in validation]))
    rows = []
    for name,material in models.items():
        for seed in range(31200,31212):
            y,obs,first = trajectories(material,seed)
            hit = first>=0
            mean,mode = prediction(bank,obs,temperature)
            _,kink,_ = fit_contact(obs,BASE,SIGMA)
            estimates = dict(model_average=mean,model_map=mode,impulse=kink,savgol=savgol_filter(obs,width,2))
            rows.append(dict(materials=name,seed=seed,contacts=int(hit.sum()),n=len(hit),
                errors={k:dict(all=rmse(v,y),contact=rmse(v[hit],y[hit]),free_flight=rmse(v[~hit],y[~hit]))
                        for k,v in estimates.items()}))
    return dict(prior_grid_size=len(bank),temperatures=temp_grid,temperature=temperature,savgol_width=width,
                validation_seeds=[31000,31001],material_sets=material_sets,rows=rows)


if __name__ == "__main__":
    (HERE/"bounce36.json").write_text(json.dumps(main(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
