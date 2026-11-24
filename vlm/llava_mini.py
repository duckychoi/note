"""LLaVA식 축소 VLM — Qwen2.5-VL(2502.13923)·Qwen3-VL(2511.21631) 단계 구조.

구성: 비전 인코더(동결) → 2×2 토큰 병합(pixel-unshuffle) → 2층 MLP 프로젝터 → LLM(inputs_embeds).
시퀀스 = [이미지 토큰][프롬프트][답]  — 손실은 답 토큰만.
단계: stage0 = 프로젝터만 학습(정렬), stage1 = 프로젝터 + LLM (비전은 동결 유지).

    # 실제 실행(GPU, 미검증): SigLIP + Qwen2.5, LLaVA json 형식
    python vlm/llava_mini.py --vision google/siglip-base-patch16-224 --llm Qwen/Qwen2.5-0.5B \
        --stage0_json /data/llava/blip_laion_cc_sbu_558k.json --stage0_images /data/llava/images \
        --stage1_json /data/llava/llava_v1_5_mix665k.json --stage1_images /data/llava --out runs/vlm/b
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """[B,Dv,h,w] → 2×2 병합 → [B, (h/2)(w/2), D]."""

    def __init__(self, dv, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * dv, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, fmap):
        B, C, h, w = fmap.shape
        if h % 2 or w % 2:
            fmap = F.pad(fmap, (0, w % 2, 0, h % 2))
        x = F.pixel_unshuffle(fmap, 2)                      # [B,4C,h/2,w/2]
        return self.mlp(x.flatten(2).transpose(1, 2))


class VLM(nn.Module):
    def __init__(self, vision_fn, dv, lm, embed_fn, d, freeze_vision=True):
        """vision_fn(img)->[B,Dv,h,w]; lm(inputs_embeds)->logits; embed_fn(ids)->embeds."""
        super().__init__()
        self.vision_fn, self.lm, self.embed_fn = vision_fn, lm, embed_fn
        self.proj = Projector(dv, d)
        self.freeze_vision = freeze_vision

    def build(self, images, prompts, answers, pad_id, device):
        with torch.set_grad_enabled(not self.freeze_vision):
            fmap = self.vision_fn(images.to(device))
        img_tok = self.proj(fmap.float()).to(fmap.dtype if fmap.is_floating_point() else torch.float32)
        n_img = img_tok.shape[1]
        seqs = [p + a for p, a in zip(prompts, answers)]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
        lmask = torch.zeros((len(seqs), n_img + T), device=device)
        for i, (p, a) in enumerate(zip(prompts, answers)):
            ids[i, :len(p) + len(a)] = torch.tensor(p + a, device=device)
            lmask[i, n_img + len(p):n_img + len(p) + len(a)] = 1.0
        emb = torch.cat([img_tok.to(self.embed_fn(ids[:, :1]).dtype), self.embed_fn(ids)], 1)
        targets = torch.cat([torch.zeros((len(seqs), n_img), dtype=torch.long, device=device), ids], 1)
        return emb, targets, lmask

    def loss(self, images, prompts, answers, pad_id, device):
        emb, targets, lmask = self.build(images, prompts, answers, pad_id, device)
        logits = self.lm(emb[:, :-1]).float()
        lf = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1), reduction="none")
        m = lmask[:, 1:].reshape(-1)
        return (lf * m).sum() / m.sum().clamp_min(1)

    @torch.no_grad()
    def answer(self, image, prompt_ids, max_new, eos_id, pad_id, device):
        emb, _, _ = self.build(image[None], [prompt_ids], [[]], pad_id, device)
        out = []
        for _ in range(max_new):
            nxt = int(self.lm(emb)[:, -1].argmax(-1))
            if nxt == eos_id:
                break
            out.append(nxt)
            emb = torch.cat([emb, self.embed_fn(torch.tensor([[nxt]], device=device))], 1)
        return out


def set_stage(vlm: VLM, lm_params, stage: int):
    """stage0: 프로젝터만 / stage1: 프로젝터 + LLM."""
    for p in vlm.proj.parameters():
        p.requires_grad_(True)
    for p in lm_params:
        p.requires_grad_(stage >= 1)
    seen, out = set(), []
    for p in list(vlm.parameters()) + list(lm_params):        # 임베딩이 VLM·LLM 양쪽에 등록돼 있어 중복 제거
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); out.append(p)
    return out


# ---------------------------------------------------------------- HF 연결·LLaVA 데이터 (미검증)
def build_hf(vision_name, llm_name, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
    vis = SiglipVisionModel.from_pretrained(vision_name, torch_dtype=dtype).to(device).eval().requires_grad_(False)
    proc = SiglipImageProcessor.from_pretrained(vision_name)
    lm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=dtype).to(device)
    tok = AutoTokenizer.from_pretrained(llm_name)

    def vision_fn(px):
        h = vis(pixel_values=px).last_hidden_state                   # [B,N,Dv]
        g = int(h.shape[1] ** 0.5)
        return h.transpose(1, 2).reshape(h.shape[0], h.shape[2], g, g)

    vlm = VLM(vision_fn, vis.config.hidden_size, lambda e: lm(inputs_embeds=e).logits,
              lm.get_input_embeddings(), lm.config.hidden_size).to(device)
    return vlm, lm, tok, proc


def llava_items(json_path, image_root, limit=None):
    """LLaVA 형식에서 이미지가 있는 첫 human/gpt 한 쌍만 사용(단순화 — 다중 턴은 미구현)."""
    rows = json.loads(Path(json_path).read_text())
    out = []
    for r in rows:
        if "image" not in r:
            continue
        conv = r["conversations"]
        q = conv[0]["value"].replace("<image>", "").strip()
        a = conv[1]["value"].strip()
        out.append({"image": str(Path(image_root) / r["image"]), "prompt": f"USER: {q}\nASSISTANT: ", "answer": a})
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True); ap.add_argument("--llm", required=True)
    ap.add_argument("--stage0_json"); ap.add_argument("--stage0_images"); ap.add_argument("--stage0_limit", type=int)
    ap.add_argument("--stage1_json"); ap.add_argument("--stage1_images"); ap.add_argument("--stage1_limit", type=int)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from PIL import Image
    dev = torch.device("cuda")
    vlm, lm, tok, proc = build_hf(a.vision, a.llm, dev)
    eos = tok.eos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log = open(out / "log.jsonl", "a")
    for stage, js, root, lim, lr in ((0, a.stage0_json, a.stage0_images, a.stage0_limit, 1e-3),
                                     (1, a.stage1_json, a.stage1_images, a.stage1_limit, 2e-5)):
        if not js:
            continue
        items = llava_items(js, root, lim); random.shuffle(items)
        params = set_stage(vlm, list(lm.parameters()), stage)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        steps = len(items) // a.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.03)
        for s in range(steps):
            b = items[s * a.batch:(s + 1) * a.batch]
            px = proc([Image.open(x["image"]).convert("RGB") for x in b], return_tensors="pt")["pixel_values"].to(dev, torch.bfloat16)
            P = [tok.encode(x["prompt"], add_special_tokens=False) for x in b]
            A = [tok.encode(x["answer"], add_special_tokens=False) + [eos] for x in b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = vlm.loss(px, P, A, pad, dev)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            if s % 50 == 0:
                log.write(json.dumps({"stage": stage, "step": s, "loss": loss.item()}) + "\n"); log.flush()
        torch.save({"proj": vlm.proj.state_dict()}, out / f"proj_stage{stage}.pt")
        if stage == 1:
            lm.save_pretrained(out / "llm_stage1")


if __name__ == "__main__":
    main()
