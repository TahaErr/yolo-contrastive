"""Filtre augmentation'ları — blur, noise, sharpen."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from .registry import PerImageAugmentation, register
import os


@register("gaussian_blur")
class RandomGaussianBlur(PerImageAugmentation):
    def __init__(self, kernel_size: int = 5, sigma_lo: float = 0.1, sigma_hi: float = 2.0, p: float = 0.5):
        super().__init__(p=p)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        k = self.kernel_size
        pad = k // 2
        results = []
        for i in range(B):
            sigma = torch.empty(1).uniform_(self.sigma_lo, self.sigma_hi).item()
            coords = torch.arange(k, device=img.device, dtype=img.dtype) - (k - 1) / 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            kernel = torch.outer(g, g)
            kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(C, 1, 1, 1)
            single = img[i:i+1]
            blurred = F.conv2d(F.pad(single, (pad, pad, pad, pad), mode="reflect"), kernel, groups=C)
            results.append(blurred)
        return torch.cat(results, dim=0)


@register("gaussian_noise")
class GaussianNoise(PerImageAugmentation):
    def __init__(self, std: float = 0.05, p: float = 0.3):
        super().__init__(p=p)
        self.std = std

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return img + self.std * torch.randn_like(img)


@register("sharpen")
class RandomSharpen(PerImageAugmentation):
    """Unsharp masking ile keskinleştirme.

    Doğru unsharp mask formülü:
        sharpened = img + strength * (img - blurred)
    """
    def __init__(self, strength: float = 1.0, p: float = 0.3):
        super().__init__(p=p)
        self.strength = strength

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        blur_kernel = torch.tensor(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]],
            device=img.device, dtype=img.dtype,
        ) / 16.0
        blur_kernel = blur_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        blurred = F.conv2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), blur_kernel, groups=C)
        return img + self.strength * (img - blurred)
