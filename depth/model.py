"""DINOv2 ViT + DPT 디코더 (Depth Anything V2 구조의 축소 구현).

- 인코더: timm DINOv2 (vit_small/base/large_patch14_dinov2.lvd142m), dynamic_img_size=True.
- 4개 중간층 특징 → reassemble(1x1 투영 + 4/2/1/0.5배 크기) → RefineNet식 융합 → 출력 헤드.
- 출력: 양수 시차(역깊이) [B,H,W] (상대 단계). 미터 단계에선 같은 헤드가 정준 깊이를 낸다(train.py 참조).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

ENCODERS = {
    "vits": ("vit_small_patch14_dinov2.lvd142m", [2, 5, 8, 11], 64),
    "vitb": ("vit_base_patch14_dinov2.lvd142m", [2, 5, 8, 11], 128),
    "vitl": ("vit_large_patch14_dinov2.lvd142m", [4, 11, 17, 23], 256),
}


class ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class FusionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.res1 = ResidualConvUnit(c)
        self.res2 = ResidualConvUnit(c)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = x + self.res1(skip)
        x = self.res2(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True) if size is not None \
            else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out(x)


class DPTHead(nn.Module):
    def __init__(self, in_dim, feat=64, out_ch=(48, 96, 192, 384)):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, c, 1) for c in out_ch])
        self.resize = nn.ModuleList([
            nn.ConvTranspose2d(out_ch[0], out_ch[0], 4, 4),
            nn.ConvTranspose2d(out_ch[1], out_ch[1], 2, 2),
            nn.Identity(),
            nn.Conv2d(out_ch[3], out_ch[3], 3, 2, 1),
        ])
        self.layer_rn = nn.ModuleList([nn.Conv2d(c, feat, 3, padding=1, bias=False) for c in out_ch])
        self.fuse = nn.ModuleList([FusionBlock(feat) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Conv2d(feat, feat // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feat // 2, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 1, 1), nn.Softplus(),      # 양수 출력
        )

    def forward(self, feats, out_hw):
        xs = [self.layer_rn[i](self.resize[i](self.proj[i](f))) for i, f in enumerate(feats)]
        p = self.fuse[3](xs[3], size=xs[2].shape[-2:])
        p = self.fuse[2](p, xs[2], size=xs[1].shape[-2:])
        p = self.fuse[1](p, xs[1], size=xs[0].shape[-2:])
        p = self.fuse[0](p, xs[0])
        y = self.out(p)
        return F.interpolate(y, size=out_hw, mode="bilinear", align_corners=True)[:, 0]


class DepthModel(nn.Module):
    def __init__(self, encoder: str = "vits", pretrained: bool = True, timm_name: str | None = None,
                 layers: list[int] | None = None, feat: int | None = None):
        super().__init__()
        name, idx, f = ENCODERS.get(encoder, (timm_name, layers, feat or 64))
        name = timm_name or name
        self.layers = layers or idx
        self.encoder = timm.create_model(name, pretrained=pretrained, dynamic_img_size=True, num_classes=0)
        self.patch = self.encoder.patch_embed.patch_size[0]
        self.head = DPTHead(self.encoder.embed_dim, feat or f)

    def features(self, x):
        return self.encoder.forward_intermediates(x, indices=self.layers, intermediates_only=True, output_fmt="NCHW")

    def forward(self, x, return_feats: bool = False):
        H, W = x.shape[-2:]
        assert H % self.patch == 0 and W % self.patch == 0, f"입력 크기는 {self.patch}의 배수여야 함"
        feats = self.features(x)
        y = self.head(feats, (H, W))
        return (y, feats) if return_feats else y

    def param_groups(self, lr_decoder: float, enc_ratio: float = 0.1, wd: float = 0.0):
        """인코더 LR = 디코더의 0.1배 (DA2·Depth Pro·MoGe 공통)."""
        return [
            {"params": self.encoder.parameters(), "lr": lr_decoder * enc_ratio, "weight_decay": wd},
            {"params": self.head.parameters(), "lr": lr_decoder, "weight_decay": wd},
        ]
