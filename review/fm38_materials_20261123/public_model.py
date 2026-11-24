"""Strict loading and shared image preprocessing for the authors' released weights."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTConfig,ViTModel

from worlds import ROOT,TASKS,render

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get("FLOW_JEPA_SOURCE",HERE.parents[2]/"research-sources/Flow-JEPA"))
sys.path.insert(0,str(SOURCE))
from jepa import JEPA
from module import Embedder,FlowTrajectoryPredictor,MLP

CKPT={"push":("pusht","14f3566de4a071b7e54708d59304eabab670e46d59c26ca1b5ab0f725f1d898e"),
      "press":("cube","f83e2a83f95f8ad3ee43d9f6cfb2ec342a8393ef91b58fc78970610bde444337"),
      "reacher":("reacher","83395e2883fb301fdfa91d5794691b6720054aa3e06dec3fafa5d4cba3ebb3c9")}


def load(task):
    name,digest=CKPT[task]
    folder=ROOT/"checkpoints"/name
    weights=folder/"weights_epoch_20.pt"
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=digest:
            raise ValueError("Checkpoint content differs from the public release")
    cfg=json.loads((folder/"config.json").read_text())
    settings=lambda k:{a:v for a,v in cfg[k].items() if not a.startswith("_")}
    # Exact tiny configuration used by stable_pretraining.backbone.utils.vit_hf.
    config=ViTConfig(hidden_size=192,num_hidden_layers=12,num_attention_heads=3,
                     intermediate_size=768,patch_size=14,image_size=224)
    encoder=ViTModel(config,add_pooling_layer=False,use_mask_token=False)
    projection=settings("projector")
    projection["norm_fn"]=torch.nn.BatchNorm1d
    model=JEPA(encoder,FlowTrajectoryPredictor(**settings("predictor")),Embedder(**settings("action_encoder")),
               MLP(**projection),num_flow_steps=cfg["num_flow_steps"],flow_source=cfg["flow_source"],
               flow_source_noise_scale=cfg["flow_source_noise_scale"],freeze_encoder=True,freeze_projector=True)
    state=torch.load(weights,map_location="cpu",weights_only=True)
    if "encoder.encoder.layer.0.attention.attention.query.weight" in model.state_dict():
        # Transformers renamed the same ViT linear layers between releases.
        names={"attention.q_proj":"attention.attention.query","attention.k_proj":"attention.attention.key",
               "attention.v_proj":"attention.attention.value","attention.o_proj":"attention.output.dense",
               "mlp.fc1":"intermediate.dense","mlp.fc2":"output.dense"}
        for key in list(state):
            if key.startswith("encoder.layers."):
                dest=key.replace("encoder.layers.","encoder.encoder.layer.")
                for old,new in names.items():
                    dest=dest.replace(old,new)
                state[dest]=state.pop(key)
    model.load_state_dict(state,strict=True)
    return model.to("cuda").eval()


def preprocess(images):
    if not images.flags.writeable:
        images=images.copy()
    t=torch.from_numpy(images).to("cuda").permute(0,3,1,2).float()/255
    mean=t.new_tensor([.485,.456,.406])[None,:,None,None]
    sd=t.new_tensor([.229,.224,.225])[None,:,None,None]
    return F.interpolate((t-mean)/sd,(224,224),mode="bilinear",align_corners=False,antialias=True)


@torch.no_grad()
def vision(model,images):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        z=model.encode({"pixels":preprocess(images)[:,None]})["emb"]
    return z[:,0,0].float()


def action_tensor(task,a):
    # Each action is a physical primitive, repeated across the five public action blocks.
    # The original scalar units are incompatible with our new tasks; use a common standardized index.
    dim=10 if task in ("push","reacher") else 25
    v=(a.float()-1.5)/np.sqrt(1.25)
    out=v.new_zeros((len(a),5,5,dim//5))
    out[:,:,:,0]=v[:,None,None]
    return out.flatten(2)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=TASKS)
    ap.add_argument("--download",action="store_true",help="Fetch the two pinned public checkpoints before encoding.")
    args=ap.parse_args()
    for task in ([args.task] if args.task else TASKS):
        if args.download:
            import shutil
            from huggingface_hub import hf_hub_download
            name=CKPT[task][0]
            folder=ROOT/"checkpoints"/name
            folder.mkdir(parents=True,exist_ok=True)
            for filename in ("config.json","weights_epoch_20.pt"):
                cached=hf_hub_download("HuoYanchen/Flow-JEPA",f"flow-jepa-{name}/{filename}",
                                       revision="f26a1a5f3d1e0aeeb68377a7e7463088f244c79c")
                shutil.copyfile(cached,folder/filename)
        net=load(task)
        for split in ("train","val","source","shift"):
            with np.load(ROOT/f"{task}_{split}.npz") as data:
                values=data["path"]
            # Identical rasters have exactly identical frozen encodings.
            unique,idx=[],[]
            mapping={}
            for state in values.reshape(-1,values.shape[-1]):
                img=render(task,state)
                key=hashlib.blake2b(img,digest_size=20).digest()
                if key not in mapping:
                    mapping[key]=len(unique)
                    unique.append(img)
                idx.append(mapping[key])
            encoded=np.concatenate([vision(net,np.stack(unique[i:i+96])).cpu().numpy()
                                    for i in range(0,len(unique),96)])
            np.save(ROOT/f"{task}_{split}_z.npy",encoded[idx].reshape(*values.shape[:2],192))
            print(task,split,len(unique),"distinct frames",flush=True)
