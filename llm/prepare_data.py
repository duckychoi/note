"""토큰 샤드 만들기 (GPU 머신·인터넷 환경에서 실행; 이 서버에서는 미실행·미검증).

    pip install datasets tiktoken
    python llm/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --out /data/tok/fineweb_edu --max_tokens 7e9
    python llm/prepare_data.py --dataset HuggingFaceTB/finemath --name finemath-4plus --out /data/tok/finemath --max_tokens 5e8

GPT-2 BPE(tiktoken, 어휘 50257 → 모델 어휘 50304로 패딩) uint16 샤드, 문서 끝마다 EOT. 마지막 샤드 1개는 검증용으로 남겨라.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="train"); ap.add_argument("--text_key", default="text")
    ap.add_argument("--out", required=True); ap.add_argument("--shard_tokens", type=float, default=1e8)
    ap.add_argument("--max_tokens", type=float, default=1e10)
    a = ap.parse_args()
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2"); eot = enc._special_tokens["<|endoftext|>"]
    ds = load_dataset(a.dataset, name=a.name, split=a.split, streaming=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    buf, total, shard = [], 0, 0
    cap = int(a.shard_tokens)
    for ex in ds:
        ids = enc.encode_ordinary(ex[a.text_key]) + [eot]
        buf.extend(ids); total += len(ids)
        while len(buf) >= cap:
            np.array(buf[:cap], np.uint16).tofile(out / f"shard_{shard:05d}.bin"); buf = buf[cap:]; shard += 1
        if total >= a.max_tokens:
            break
    if buf:
        np.array(buf, np.uint16).tofile(out / f"shard_{shard:05d}.bin")
    print(f"{total/1e9:.2f}B 토큰, 샤드 {shard + 1}개 → {out}")


if __name__ == "__main__":
    main()
