"""Train the #38 planner on source demonstrations and evaluate its branches on paired new starts."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from swing import BAND, CACHE, PROBE_SPEED, PROBE_TIME, SPEED, TIMES, raster
from models import AngleReader, Data, SEEDS, action_tensor, ensure_reacher_checkpoint, load, predict, vision

SOURCE = Path(os.environ.get("LEWAM_SOURCE", HERE.parents[3]/"research-sources/le-wam"))
sys.path.insert(0, str(SOURCE/"src"))
from probe_planner import Prior, ProbePlanner, VisualLift, identify

DOMAINS = ("source","shift")
OUT = CACHE/"probe_planner"


def actions(a):
    return action_tensor("reacher", a)


def combine(first, second):
    return Prior(*(torch.cat([getattr(first,k),getattr(second,k)]) for k in ("history","strength","angles","anchor")))


def make_planner(train):
    read = AngleReader(train).cuda().eval()
    read.load_state_dict(torch.load(CACHE/"reader.pt", weights_only=True))
    with torch.no_grad():
        before, after = read(train.z[:,:2]).cpu().numpy().T
    nominal = np.median(identify(before, after, PROBE_SPEED, PROBE_TIME))
    lift = VisualLift.from_demonstrations(train.angles,train.z)
    return ProbePlanner(load("reacher"),read,lift,nominal,SPEED,TIMES).cuda().eval()


def prior_for(net, data, mode, donor=None):
    z = data.z
    after = z[:,1] if donor is None else donor.z[:,1]
    observation = net.observe(z[:,0], None if mode=="photo" else after, PROBE_SPEED,PROBE_TIME)
    return net.build_prior(observation,data.a)


def training_sets(net,data):
    return (combine(prior_for(net,data,"photo"),prior_for(net,data,"probe")),
            actions(data.a).repeat(2,1,1),data.z[:,2:].repeat(2,1,1))


def train_residual(net, dataset, validation, seed, *, checkpoint_dir=OUT):
    torch.manual_seed(seed)
    p, block, target = dataset
    n = len(target)//2
    net.residual_scale.copy_((target-p.anchor).flatten(0,1).std(0).clamp_min(.025))
    pars = [v for v in net.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(pars,lr=7e-5,weight_decay=.0005,fused=True)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    best = np.inf
    history = []

    def objective(batch, ids):
        prior, blocks, y = batch
        with torch.autocast("cuda",dtype=torch.bfloat16):
            return net.flow_loss(prior.at(ids),blocks[ids],y[ids])

    for step in range(1,1001):
        net.train()
        ids = torch.randint(n,(32,),device="cuda",generator=rng)
        loss = objective(dataset,torch.cat([ids,ids+n]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pars,1.)
        optimizer.step()
        if step%200==0:
            net.eval()
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(49212)
                ids = torch.arange(len(validation[2]),device="cuda")
                valid = torch.stack([objective(validation,ix) for ix in ids.split(64)]).mean().item()
            history.append([step,valid])
            if valid<best:
                best=valid
                torch.save(net.state_dict(),checkpoint_dir/f"planner_{seed}.pt")
    if not np.isfinite(best):
        raise ValueError("Residual flow had no finite validation loss.")
    net.load_state_dict(torch.load(checkpoint_dir/f"planner_{seed}.pt",weights_only=True))
    return net.eval(),history


@torch.inference_mode()
def forecast(net,prior,data,residual):
    rng = torch.Generator(device="cuda").manual_seed(4830)
    rows = []
    for i in range(0,len(data.a),16):
        sl = slice(i,i+16)
        path = net.forecast(prior.at(sl),actions(data.a[sl]),residual=residual,generator=rng)
        rows.append(net.score(path,data.goal[sl],BAND))
    return torch.cat(rows).cpu().numpy().reshape(-1,4,2)


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True)
    ensure_reacher_checkpoint()
    visual = load("reacher")
    bank={}
    tr,val = (Data(split,visual,bank) for split in ("train","val"))
    test = {d:Data("integration_"+d,visual,bank) for d in DOMAINS}
    assert torch.equal(test["source"].z[:,0],test["shift"].z[:,0])
    assert torch.equal(test["source"].goal,test["shift"].goal)
    model = make_planner(tr)
    train_data, val_data = training_sets(model,tr),training_sets(model,val)
    priors = {}
    raw = {}
    log = {"fit":{}, "nominal_strength":float(model.nominal_strength)}
    for d,other in (("source","shift"),("shift","source")):
        x=test[d]
        for mode in ("photo","probe","swapped"):
            priors[d,mode]=prior_for(model,x,mode,test[other] if mode=="swapped" else None)
            raw[f"prior_{mode}_{d}"]=forecast(model,priors[d,mode],x,False)
        with np.load(CACHE/f"integration_{d}.npz") as f:
            raw["goal_"+d]=f["goal"].reshape(-1,4)
            raw["peak_"+d]=f["peak"].reshape(-1,4)
            raw["level_"+d]=f["level"].reshape(-1,4)[:,0]
            raw["start_probe_angles_"+d]=f["angles"][::4,:2]
        raw["k_"+d]=priors[d,"probe"].strength[::4].cpu().numpy()
    for seed in SEEDS:
        print("baseline",seed,flush=True)
        legacy=load("reacher")
        state=torch.load(CACHE/f"flow_{seed}.pt",weights_only=True)
        for k in ("predictor","action_encoder"):
            getattr(legacy,k).load_state_dict({n[len(k)+1:]:v for n,v in state.items() if n.startswith(k+".")})
        for d,x in test.items():
            raw[f"legacy_{seed}_{d}"]=predict("flow",legacy,model.read_angle,x,"probe")
        del legacy
        print("training",seed,flush=True)
        model,log["fit"][str(seed)] = train_residual(make_planner(tr),train_data,val_data,seed)
        for d,x in test.items():
            for mode in ("probe","swapped"):
                raw[f"flow_{mode}_{seed}_{d}"]=forecast(model,priors[d,mode],x,True)
        np.savez_compressed(HERE/"forecasts.npz",**raw)
        (HERE/"fit.json").write_text(json.dumps(log,indent=2),encoding="utf8")
        print("finished",seed,flush=True)


def load_artifact(seed):
    # Instantiate dimensions of the saved visual readout and lift directly from the deployed artifact.
    state=torch.load(OUT/f"planner_{seed}.pt",map_location="cuda",weights_only=True)
    read=AngleReader(stats=(state["read_angle.center"],state["read_angle.scale"])).cuda()
    lift=VisualLift(state["lift.angles"],state["lift.embeddings"])
    net=ProbePlanner(load("reacher"),read,lift,state["nominal_strength"].item(),SPEED,TIMES).cuda()
    net.load_state_dict(state)
    return net.eval()


def deploy_example(index, d, seed, image_paths=None, goal_degrees=None, residual=None):
    net=load_artifact(seed)
    if residual is None:
        settings=json.loads((HERE/"deployment.json").read_text(encoding="utf8"))
        net.use_residual_by_default=settings[str(seed)]["use_residual"]
    if image_paths is None:
        with np.load(CACHE/f"integration_{d}.npz") as f:
            before,after=[raster(q) for q in f["angles"][index*4,:2]]
            goal_degrees=np.rad2deg(f["goal"][index*4])
    else:
        from PIL import Image
        before,after=[np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    return net.select_action(before,after,encode_rgb=vision,action_blocks=actions,
                             goal_degrees=goal_degrees,probe_speed=PROBE_SPEED,elapsed=PROBE_TIME,
                             residual=residual,generator=torch.Generator(device="cuda").manual_seed(4830))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--demo",type=int)
    ap.add_argument("--images",nargs=2,metavar=("BEFORE","AFTER"))
    ap.add_argument("--goal-degrees",type=float)
    overrides=ap.add_mutually_exclusive_group()
    overrides.add_argument("--prior-only",action="store_true")
    overrides.add_argument("--full-flow",action="store_true")
    ap.add_argument("--domain",choices=DOMAINS,default="shift")
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.images is not None and args.goal_degrees is None:
        ap.error("--images requires --goal-degrees")
    if args.demo is not None or args.images is not None:
        override=False if args.prior_only else True if args.full_flow else None
        print(json.dumps(deploy_example(args.demo,args.domain,SEEDS[0],args.images,args.goal_degrees,
                                        override),indent=2))
    else:
        evaluate()
