"""생성 트랙 스모크(CPU): 작은 무작위 잠재로 이미지→공동(이미지+영상) 2단계 커리큘럼, 샘플러, 과적합 손실 감소 확인."""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow as FL, train as T
from dit import DiT


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gen_smoke_"))
    try:
        rng = np.random.default_rng(0)
        # 학습 가능한 신호: 클래스별 고정 잠재 패턴 + 작은 잡음
        protos = rng.normal(0, 1, (4, 4, 1, 8, 8)).astype(np.float32)
        for name, T_, n in (("img", 1, 64), ("vid", 3, 32)):
            d = tmp / name; d.mkdir()
            y = rng.integers(0, 4, n)
            x = np.repeat(protos[y], T_, axis=2) + 0.05 * rng.normal(size=(n, 4, T_, 8, 8))
            np.save(d / "latents.npy", x.astype(np.float16)); np.save(d / "labels.npy", y)
        cfg = {"model": {"in_ch": 4, "d": 64, "depth": 2, "heads": 4, "d_ctx": 64, "n_classes": 4},
               "res_shift": True, "base_tokens": 16, "p_drop": 0.1, "log_every": 1, "bf16": False,
               "stages": [{"image_dir": str(tmp / "img"), "steps": 150, "batch_image": 16, "lr": 3e-3, "warmup": 10},
                          {"image_dir": str(tmp / "img"), "video_dir": str(tmp / "vid"), "image_ratio": 0.3,
                           "steps": 40, "batch_image": 16, "batch_video": 8, "lr": 1e-3, "warmup": 5}]}
        (tmp / "c.json").write_text(json.dumps(cfg))
        out = T.main(["--config", str(tmp / "c.json"), "--out", str(tmp / "run"), "--cpu"])
        recs = [json.loads(l) for l in open(out / "log.jsonl")]
        s0 = [r["loss"] for r in recs if r["stage"] == 0]
        print(f"[stage0 image] loss 처음10 {np.mean(s0[:10]):.3f} → 마지막10 {np.mean(s0[-10:]):.3f}")
        assert all(np.isfinite(r["loss"]) for r in recs) and np.mean(s0[-10:]) < np.mean(s0[:10])
        kinds = {r["kind"] for r in recs if r["stage"] == 1}
        assert kinds == {"image", "video"}, kinds
        ck = torch.load(out / "stage1.pt")
        m = DiT(in_ch=4, d=64, depth=2, heads=4, d_ctx=64, n_classes=4); m.load_state_dict(ck["ema"])
        x = FL.sample(m, (2, 4, 3, 8, 8), torch.tensor([0, 1]), steps=10, cfg_scale=2.0, null_cond=4)
        assert x.shape == (2, 4, 3, 8, 8) and torch.isfinite(x).all()
        # 공유 adaLN vs 블록별 adaLN 파라미터 비교(HG4 설계 확인)
        a = sum(p.numel() for p in DiT(in_ch=4, d=64, depth=6, heads=4, n_classes=4, shared_mod=True).parameters())
        b = sum(p.numel() for p in DiT(in_ch=4, d=64, depth=6, heads=4, n_classes=4, shared_mod=False).parameters())
        print(f"[params] 공유 adaLN {a} vs 블록별 {b} ({(b-a)/b:.0%} 절감)")
        print("SMOKE OK — 2단계 커리큘럼(이미지/영상 혼합)·EMA·CFG 샘플러 동작, 손실 감소 (성능 판정 아님)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
