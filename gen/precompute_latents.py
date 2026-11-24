"""잠재 사전계산 (GPU 머신에서 실행; 이 저장소 서버에서는 VAE 가중치가 없어 미검증).

이미지(ImageNet 부분집합, 클래스 폴더):  python gen/precompute_latents.py image --src /data/imagenet100/train --res 128 --out /data/lat/img128
영상(UCF-101, 클래스 폴더의 .avi):       python gen/precompute_latents.py video --src /data/UCF-101 --res 128 --frames 17 --out /data/lat/ucf128

- 이미지 VAE: diffusers AutoencoderKL "stabilityai/sd-vae-ft-ema" (f8, C4), 스케일 0.18215.
- 영상 VAE: diffusers AutoencoderKLWan "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" subfolder="vae" (4×8×8, C16),
  config 의 latents_mean/std 로 정규화. 17프레임 → 잠재 T=5.
  ※ 영상 잠재 채널(16)과 이미지 잠재 채널(4)이 다르면 같은 DiT로 공동 학습할 수 없다 → 공동 학습하려면 이미지도
    Wan VAE로 1프레임 인코딩(T=1)할 것(--image_vae wan). Wan 자체가 이미지=1프레임 영상으로 다룸.
출력: out/latents.npy [N,C,T,H,W] float16, out/labels.npy [N] int64, out/classes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_image_vae(kind, device):
    if kind == "sd":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
        return lambda x: vae.encode(x).latent_dist.sample() * 0.18215          # [B,4,h,w]
    return load_wan_vae(device)


def load_wan_vae(device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
                                           torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    return lambda x: (vae.encode(x if x.dim() == 5 else x[:, :, None]).latent_dist.sample() - mean) / std


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["image", "video"])
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=128); ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--image_vae", choices=["sd", "wan"], default="wan")
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--max_per_class", type=int, default=10**9)
    ap.add_argument("--label_offset", type=int, default=0, help="이미지·영상 클래스를 한 표로 합칠 때(예: UCF는 100)")
    a = ap.parse_args()
    dev = "cuda"
    classes = sorted(p.name for p in Path(a.src).iterdir() if p.is_dir())
    lat, lab = [], []
    if a.kind == "image":
        from PIL import Image
        enc = load_image_vae(a.image_vae, dev)
        files = [(f, ci) for ci, c in enumerate(classes) for f in sorted((Path(a.src) / c).glob("*"))[:a.max_per_class]]
        for s in range(0, len(files), a.batch):
            xs = []
            for f, _ in files[s:s + a.batch]:
                im = Image.open(f).convert("RGB"); w, h = im.size; m = min(w, h)
                im = im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((a.res, a.res), Image.BICUBIC)
                xs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1)
            z = enc(torch.stack(xs).to(dev))
            z = z if z.dim() == 5 else z[:, :, None]
            lat.append(z.half().cpu().numpy()); lab += [ci for _, ci in files[s:s + a.batch]]
    else:
        import torchvision
        enc = load_wan_vae(dev)
        vids = [(f, ci) for ci, c in enumerate(classes) for f in sorted((Path(a.src) / c).glob("*.avi"))[:a.max_per_class]]
        for f, ci in vids:
            v, _, _ = torchvision.io.read_video(str(f), pts_unit="sec", output_format="TCHW")
            if v.shape[0] < a.frames:
                continue
            st = (v.shape[0] - a.frames) // 2
            v = v[st:st + a.frames].float() / 127.5 - 1
            v = torch.nn.functional.interpolate(v, size=(a.res, a.res), mode="bilinear", antialias=True)
            z = enc(v.permute(1, 0, 2, 3)[None].to(dev))
            lat.append(z.half().cpu().numpy()); lab.append(ci)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "latents.npy", np.concatenate(lat)); np.save(out / "labels.npy", np.array(lab, np.int64) + a.label_offset)
    (out / "classes.json").write_text(json.dumps(classes))


if __name__ == "__main__":
    main()
