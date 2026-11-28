"""Paired seed and physical-scene uncertainty for independent single-image RGB AbsRel."""
import json

import numpy as np

from controlled_observations import write_json
from select_stereo_pairs import OUT


def summarize():
    seeds=(4244,4245,4246)
    labels=("d2","silog","fixed","kf","real","development_selected")
    scores=[]
    scenes=None
    matched_design=None
    references={}
    for seed in seeds:
        root=OUT/f"full-students-{seed}"
        report=json.loads((root/"hammer_rgb_results.json").read_text())
        if seed!=report["student_seed"] or report["primary"]!="rgb" or report["frames"]!=775:
            raise ValueError("Unmatched evaluation protocol")
        settings=report["matched_training"]
        if set(settings)!={"d2","silog","fixed","kf"} or any(v[2]!=seed for v in settings.values()):
            raise ValueError("Recorded training seeds do not match the evaluation label")
        design={tuple(v[:2]+v[3:]) for v in settings.values()}
        if len(design)!=1 or (matched_design is not None and matched_design!=design):
            raise ValueError("Training manifests and budgets must be identical across seeds")
        matched_design=design
        if scenes is None:
            scenes=sorted(report["scene_ids"])
        if sorted(report["scene_ids"])!=scenes:
            raise ValueError("Physical scenes changed")
        reference=report["comparison_reference_matched"]
        if reference not in ("d2","silog"):
            raise ValueError("Unknown development-selected comparison")
        references[seed]=reference
        values=[]
        for label in labels:
            name=reference if label=="development_selected" else label
            means=[]
            for scene in scenes:
                rows=json.loads((root/"hammer-scenes"/f"{name}-{scene}.json").read_text())["rows"]
                means.append(np.mean([x["rgb"][0] for x in rows.values()]))
            if not np.isclose(np.mean(means),report["scores"][name][0][0],atol=1e-10):
                raise ValueError("Scene and evaluation means disagree")
            values.append(means)
        scores.append(values)
    # Sample scenes once across all seeds and methods, and seeds once across all scenes and methods.
    grid=np.asarray(scores).transpose(0,2,1)
    average=grid.mean(axis=(0,1))
    rng=np.random.default_rng(4243)
    ids=rng.integers(len(seeds),size=(20000,len(seeds),1))
    groups=rng.integers(len(scenes),size=(20000,1,len(scenes)))
    boot=grid[ids,groups].mean(axis=(1,2))
    decisions={}
    k=labels.index("kf")
    for j,name in enumerate(labels):
        if j==k:
            continue
        point=100*(1-average[k]/average[j])
        low,high=np.percentile(100*(1-boot[:,k]/boot[:,j]),[2.5,97.5])
        totals=grid.sum(axis=1,keepdims=True)-grid
        leave_one=100*(1-totals[:,:,k].mean(0)/totals[:,:,j].mean(0))
        decisions[name]=dict(relative_reduction_percent=point,percentile_ci95=[low,high],
            point_at_least_three_percent=bool(point>=3),lower_bound_at_least_three_percent=bool(low>=3),
            per_seed_reduction_percent=(100*(1-grid[:,:,k].mean(1)/grid[:,:,j].mean(1))).tolist(),
            leave_one_scene_out_relative_reduction_percent=dict(zip(scenes,leave_one.tolist())))
    write_json(OUT/"hammer_three_seed_rgb.json",dict(seeds=seeds,physical_scenes=scenes,frames=775,
        rgb_absrel=dict(zip(labels,average)),development_selected=references,comparisons=decisions,
        primary="development_selected",scope="public Depth Pro 2-S adaptation on Hypersim, local learning recipes",
        nyu_reference="same legacy real-trained model across seeds, unmatched image exposure count and checkpoint selection"))


if __name__=="__main__":
    summarize()
