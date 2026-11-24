"""LLM 트랙 스모크(CPU): 문자 토크나이저 + 작은 GPT로
 (1) 사전학습 3단계(일반 const → 어닐링 linear 감쇠·혼합 변경 → 문맥 확장 RoPE 기저↑), dense 와 MoE(aux-free) 둘 다
 (2) 콜드스타트 SFT → GRPO → 온폴리시 증류  를 끝까지 돌린다. 성능 판정 아님."""
import json, os, random, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpo as G, pretrain as PT
from model import GPT, GPTConfig

CHARS = "0123456789+=\n abcdefghijklmnopqrstuvwxyz."


class CharTok:
    eos_id, pad_id = len(CHARS), len(CHARS) + 1
    def encode(self, s): return [CHARS.index(c) for c in s]
    def decode(self, ids): return "".join(CHARS[i] for i in ids if i < len(CHARS))


def make_shards(d, text, n_files=2):
    d.mkdir(parents=True, exist_ok=True)
    ids = np.array(CharTok().encode(text), np.uint16)
    for i, part in enumerate(np.array_split(ids, n_files)):
        part.tofile(d / f"shard_{i:03d}.bin")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="llm_smoke_"))
    random.seed(0); torch.manual_seed(0)
    try:
        rng = random.Random(0)
        add = "".join(f"{a}+{b}={a+b}\n" for a, b in ((rng.randint(0, 9), rng.randint(0, 9)) for _ in range(6000)))
        prose = "".join(rng.choice(["the cat sat. ", "a dog ran. ", "we add numbers. "]) for _ in range(4000))
        make_shards(tmp / "web", prose + add[:20000]); make_shards(tmp / "math", add); make_shards(tmp / "val", add[-8000:], 1)
        V = len(CHARS) + 2
        for moe in (False, True):
            cfg = {"model": {"vocab": V, "d": 64, "n_layer": 3, "n_head": 4, "n_kv_head": 2, "max_seq": 128,
                             "moe": moe, "n_experts": 4, "top_k": 2, "n_shared": 1, "first_dense": 1, "balance": "aux_free"},
                   "val_data": [[str(tmp / "val/*.bin"), 1.0]], "eval_every": 40, "eval_iters": 4, "log_every": 10, "bf16": False,
                   "stages": [
                       {"name": "general", "data": [[str(tmp / "web/*.bin"), 1.0]], "seq_len": 32, "micro_batch": 16, "accum": 1,
                        "steps": 120, "warmup": 10, "lr_start": 3e-3, "shape": "const"},
                       {"name": "anneal", "data": [[str(tmp / "web/*.bin"), 0.5], [str(tmp / "math/*.bin"), 0.5]], "seq_len": 32,
                        "micro_batch": 16, "accum": 1, "steps": 40, "lr_start": 3e-3, "lr_end": 3e-4, "shape": "linear"},
                       {"name": "long", "data": [[str(tmp / "math/*.bin"), 1.0]], "seq_len": 96, "rope_base": 100000.0,
                        "micro_batch": 4, "accum": 2, "steps": 10, "lr_start": 3e-4, "shape": "const"}]}
            (tmp / "c.json").write_text(json.dumps(cfg))
            out = PT.main(["--config", str(tmp / "c.json"), "--out", str(tmp / f"pt_{moe}"), "--cpu"])
            recs = [json.loads(l) for l in open(out / "log.jsonl")]
            vl = [r["val_loss"] for r in recs if "val_loss" in r]
            tag = "MoE(aux-free)" if moe else "dense"
            extra = f", 전문가 부하 max/mean {recs[-1].get('moe_load_max_over_mean', float('nan')):.2f}" if moe else ""
            print(f"[pretrain {tag}] val loss {vl[0]:.3f} → {vl[-1]:.3f}{extra}")
            assert all(np.isfinite(r["loss"]) for r in recs) and vl[-1] < vl[0]
        # ---- 후학습: dense 체크포인트에서
        ck = torch.load(tmp / "pt_False/stage2.pt")
        model = GPT(GPTConfig(**ck["config"])); model.load_state_dict(ck["model"])
        tok = CharTok()
        data = [{"prompt": f"{a}+{b}=", "target": str(a + b), "answer": str(a + b)} for a in range(10) for b in range(10)]
        reward = lambda text, ex: 1.0 if text.strip() == ex["answer"] else 0.0
        acc0 = G.evaluate(model, tok, data, reward, 3, "cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(60):
            G.sft_step(model, opt, tok, random.sample(data, 16), "cpu")
        acc1 = G.evaluate(model, tok, data, reward, 3, "cpu")
        print(f"[SFT] 정확도 {acc0:.2f} → {acc1:.2f}")
        cfg = G.GRPOConfig(group=8, prompts_per_step=8, max_new=3, lr=5e-5, minibatch=64, inner_epochs=1, max_pass=0.99)
        ref = GPT(GPTConfig(**ck["config"])); ref.load_state_dict(model.state_dict()); ref.eval()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        stats = [G.grpo_step(model, ref, opt, tok, data, reward, cfg, "cpu") for _ in range(30)]
        acc2 = G.evaluate(model, tok, data, reward, 3, "cpu")
        kept = sum(s["groups_kept"] for s in stats)
        print(f"[GRPO] 30스텝, 유지 그룹 {kept}, 평균 보상 {stats[0]['reward_mean']:.2f} → {stats[-1]['reward_mean']:.2f}, 정확도 {acc1:.2f} → {acc2:.2f}")
        assert kept > 0 and all(np.isfinite(s["reward_mean"]) for s in stats)
        student = GPT(GPTConfig(**{**ck["config"], "n_layer": 2}))
        opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
        kls = [G.distill_step(student, model.eval(), opt, tok, data, cfg, "cpu") for _ in range(30)]
        print(f"[distill] KL {np.mean(kls[:5]):.3f} → {np.mean(kls[-5:]):.3f}")
        assert np.isfinite(kls).all()
        print("SMOKE OK — 사전학습 3단계(dense/MoE)·SFT·GRPO·증류 실행 (성능 판정 아님)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
