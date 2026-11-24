"""Recompute original validation/test aggregates for the three archived primary models."""
import json

from fresh import HERE, SOURCE, METRICS, flow, g, np, read, torch


def main():
    torch.set_num_threads(4)
    runs = [r for r in read("gate5.json")["runs"] if r["role"]=="primary"]
    weights = torch.load(SOURCE/"gate5_primary_models.pt", map_location="cpu", weights_only=True)
    threshold = read("gate3.json")["config"]["xthr"]
    data = {}
    for r in runs:
        model = flow.Proposal()
        model.load_state_dict(weights[f"{r['M']}_{r['train_seed']}"])
        model.eval()
        n = r["N"]
        for i in range(4):
            case = g.make_cases(51200+i, 48, test=True)
            result = g.run_pf_full(case,n,r["pf_temp"],threshold,500+n+i,
                init=flow.proposal_init(model,case,n,0),resample_after=g.CUT-1)
            for k in METRICS:
                data[f"{r['train_seed']}_{i}_{k}"] = result[k]
    np.savez_compressed(HERE/"primary_replay.npz", **data)
    delta = {str(r["train_seed"]): float(max(abs(data[f"{r['train_seed']}_{i}_energy"].mean()-e)
        for i,e in enumerate(r["energy_per_seed"]))) for r in runs}
    (HERE/"replay.json").write_text(json.dumps(delta,indent=2),encoding="utf8")
    print(delta)


if __name__=="__main__":
    main()
