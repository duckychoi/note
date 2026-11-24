import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from swing import CACHE, generate

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--gate",action="store_true")
    args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    for domain in ("source","shift"):
        name,n,seed=("gate",256,49317) if args.gate else ("integration",512,49211)
        split=f"{name}_{domain}"
        np.savez_compressed(CACHE/f"{split}.npz",**generate(split,domain,n,seed))
