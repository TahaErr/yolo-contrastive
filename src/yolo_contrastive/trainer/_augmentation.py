"""Augmentation mixin — legacy view2 + new pipeline support."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class AugmentationMixin:
    """Mixin providing view-2 augmentation for the trainer."""

    def _gaussian_blur(self, x: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
        if k < 3 or k % 2 == 0:
            return x
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype
        cache = getattr(self, "_cl_blur_cache", None)
        if cache is None:
            cache = {}
            self._cl_blur_cache = cache
        cache_key = (c, k, float(sigma), str(device), str(dtype))
        kernel = cache.get(cache_key)
        if kernel is None:
            coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            kernel = torch.outer(g, g)
            kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(c, 1, 1, 1)
            cache[cache_key] = kernel
        pad = k // 2
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=c)

    def _make_view2(self, img: torch.Tensor) -> torch.Tensor:
        """Legacy torch-only view2 augmentation."""
        if not torch.is_tensor(img):
            return img
        x2 = img.clone()
        if not torch.is_floating_point(x2):
            x2 = x2.float()
        with torch.no_grad():
            mx = float(x2.max())
        if mx > 1.5:
            x2 = x2 / 255.0

        cfg = getattr(self, "cl_cfg", None)
        flip_p = cfg.flip_p if cfg else 0.5
        gray_p = cfg.gray_p if cfg else 0.2
        blur_p = cfg.blur_p if cfg else 0.5
        blur_k = cfg.blur_k if cfg else 5
        blur_sigma = cfg.blur_sigma if cfg else 1.0
        bright_lo = cfg.brightness_lo if cfg else 0.6
        bright_hi = cfg.brightness_hi if cfg else 1.4
        cont_lo = cfg.contrast_lo if cfg else 0.6
        cont_hi = cfg.contrast_hi if cfg else 1.4

        if flip_p > 0 and torch.rand((), device=x2.device) < flip_p:
            x2 = torch.flip(x2, dims=[3])
        b_factor = torch.empty((x2.shape[0], 1, 1, 1), device=x2.device, dtype=x2.dtype).uniform_(bright_lo, bright_hi)
        x2 = x2 * b_factor
        mean = x2.mean(dim=(2, 3), keepdim=True)
        c_factor = torch.empty((x2.shape[0], 1, 1, 1), device=x2.device, dtype=x2.dtype).uniform_(cont_lo, cont_hi)
        x2 = (x2 - mean) * c_factor + mean
        if gray_p > 0:
            mask = torch.rand((x2.shape[0], 1, 1, 1), device=x2.device) < gray_p
            if mask.any():
                gray = x2.mean(dim=1, keepdim=True).expand_as(x2)
                x2 = torch.where(mask, gray, x2)
        if blur_p > 0:
            m = torch.rand((x2.shape[0],), device=x2.device) < blur_p
            if m.any():
                blurred = self._gaussian_blur(x2, k=blur_k, sigma=blur_sigma)
                x2 = torch.where(m.view(-1, 1, 1, 1), blurred, x2)
        return x2.clamp(0.0, 1.0)

    def make_view2(self, img: torch.Tensor) -> torch.Tensor:
        """Public: new pipeline or legacy fallback."""
        if not torch.is_tensor(img):
            return img
        pipeline = getattr(self, "_cl_aug_pipeline", None)
        if pipeline is not None:
            x = img.clone()
            with torch.no_grad():
                if float(x.max()) > 2.0:
                    x = x / 255.0
            return pipeline(x)
        return self._make_view2(img)
