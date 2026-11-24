"""VLM 스모크(CPU): 작은 무작위 ViT(동결) + 작은 GPT. 합성 이미지(색 사각형)의 색 이름을 답하게
stage0(프로젝터만) → stage1(+LLM) 학습, 손실 감소와 정확도를 확인. 성능 판정 아님."""
import os, random, sys
import numpy as np, torch, timm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm"))
import llava_mini as V
from model import GPT, GPTConfig

COLORS = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 60, 220), "yellow": (230, 220, 30)}
CH = "abcdefghijklmnopqrstuvwxyz ?:\n"


class Tok:
    eos_id, pad_id = len(CH), len(CH) + 1
    def encode(self, s): return [CH.index(c) for c in s]
    def decode(self, ids): return "".join(CH[i] for i in ids if i < len(CH))


def sample(rng):
    name = rng.choice(list(COLORS))
    img = np.full((56, 56, 3), 128, np.float32) + np.random.default_rng(rng.randint(0, 10**9)).normal(0, 10, (56, 56, 3))
    y, x = rng.randint(0, 28), rng.randint(0, 28)
    img[y:y + 28, x:x + 28] = COLORS[name]
    t = torch.from_numpy((img / 255.0 - 0.5) / 0.25).permute(2, 0, 1).float()
    return t, name


def main():
    random.seed(0); torch.manual_seed(0)
    rng = random.Random(0); tok = Tok()
    vit = timm.create_model("vit_small_patch14_dinov2", pretrained=False, dynamic_img_size=True, num_classes=0,
                            depth=2, embed_dim=48, num_heads=3).eval().requires_grad_(False)
    vision_fn = lambda x: vit.forward_intermediates(x, indices=[1], intermediates_only=True, output_fmt="NCHW")[0]
    lm = GPT(GPTConfig(vocab=len(CH) + 2, d=64, n_layer=2, n_head=4, n_kv_head=2))
    vlm = V.VLM(vision_fn, 48, lambda e: lm(inputs_embeds=e), lm.emb, 64)
    prompt = tok.encode("color?\n")
    data = [sample(rng) for _ in range(400)]

    def acc():
        ok = 0
        for img, name in data[:40]:
            ok += tok.decode(vlm.answer(img, prompt, 8, tok.eos_id, tok.pad_id, "cpu")) == name
        return ok / 40

    for stage, steps, lr in ((0, 80, 3e-3), (1, 80, 1e-3)):
        params = V.set_stage(vlm, list(lm.parameters()), stage)
        n_lm = sum(p.requires_grad for p in lm.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        ls = []
        for _ in range(steps):
            b = rng.sample(data, 16)
            loss = vlm.loss(torch.stack([x for x, _ in b]), [prompt] * 16,
                            [tok.encode(n) + [tok.eos_id] for _, n in b], tok.pad_id, "cpu")
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
        print(f"[stage{stage}] 학습 LLM 파라미터 텐서 {n_lm}개, loss {np.mean(ls[:10]):.3f} → {np.mean(ls[-10:]):.3f}, 색 정답률 {acc():.2f} (우연 0.25)")
        assert np.isfinite(ls).all() and np.mean(ls[-10:]) < np.mean(ls[:10])
        if stage == 0:
            assert n_lm == 0, "stage0 은 LLM 동결이어야 함"
    print("SMOKE OK — 프로젝터 정렬 → 공동 학습 2단계, 답 토큰만 손실 (성능 판정 아님)")


if __name__ == "__main__":
    main()
