"""Geometrik augmentation'lar — flip, rotation, affine."""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .registry import BaseAugmentation, PerImageAugmentation, register


@register("horizontal_flip")
class RandomHorizontalFlip(BaseAugmentation):
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return torch.flip(img, dims=[3])


@register("vertical_flip")
class RandomVerticalFlip(BaseAugmentation):
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return torch.flip(img, dims=[2])


@register("rotation90")
class RandomRotation90(PerImageAugmentation):
    """Rastgele 90°/180°/270° döndürme (per-image)."""

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        k = torch.randint(1, 4, (B,), device=img.device)  # 1=90°, 2=180°, 3=270°
        out = img.clone()
        for i in range(B):
            out[i] = torch.rot90(img[i], k[i].item(), dims=[1, 2])
        return out


@register("rotation")
class RandomRotation(PerImageAugmentation):
    """Rastgele açıyla döndürme (bilinear interpolation)."""

    def __init__(self, degrees: float = 30.0, p: float = 0.5):
        super().__init__(p=p)
        self.degrees = degrees

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        angles = (torch.rand(B, device=img.device) * 2 - 1) * self.degrees  # [-deg, +deg]

        # Affine matrix oluştur
        theta = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
        cos_a = torch.cos(angles * math.pi / 180)
        sin_a = torch.sin(angles * math.pi / 180)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = F.affine_grid(theta, img.shape, align_corners=False)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _params(self):
        return {"degrees": self.degrees, "p": self.p}


@register("affine")
class RandomAffine(PerImageAugmentation):
    """Rotation + translate + scale."""

    def __init__(self, degrees: float = 10, translate: float = 0.1,
                 scale_lo: float = 0.9, scale_hi: float = 1.1, p: float = 0.5):
        super().__init__(p=p)
        self.degrees = degrees
        self.translate = translate
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        dev, dt = img.device, img.dtype

        angles = (torch.rand(B, device=dev) * 2 - 1) * self.degrees
        cos_a = torch.cos(angles * math.pi / 180)
        sin_a = torch.sin(angles * math.pi / 180)
        scale = torch.empty(B, device=dev).uniform_(self.scale_lo, self.scale_hi)
        tx = (torch.rand(B, device=dev) * 2 - 1) * self.translate
        ty = (torch.rand(B, device=dev) * 2 - 1) * self.translate

        theta = torch.zeros(B, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = -sin_a * scale
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a * scale
        theta[:, 1, 1] = cos_a * scale
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, img.shape, align_corners=False)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
