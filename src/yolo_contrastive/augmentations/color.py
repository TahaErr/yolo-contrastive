"""Renk augmentation'ları — brightness, contrast, saturation, hue, solarize, posterize."""

from __future__ import annotations
import torch
from .registry import PerImageAugmentation, register


@register("brightness")
class RandomBrightness(PerImageAugmentation):
    def __init__(self, lo: float = 0.6, hi: float = 1.4, p: float = 0.8):
        super().__init__(p=p)
        self.lo, self.hi = lo, hi

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        factor = torch.empty(img.shape[0], 1, 1, 1, device=img.device, dtype=img.dtype).uniform_(self.lo, self.hi)
        return img * factor


@register("contrast")
class RandomContrast(PerImageAugmentation):
    def __init__(self, lo: float = 0.6, hi: float = 1.4, p: float = 0.8):
        super().__init__(p=p)
        self.lo, self.hi = lo, hi

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        mean = img.mean(dim=(2, 3), keepdim=True)
        factor = torch.empty(img.shape[0], 1, 1, 1, device=img.device, dtype=img.dtype).uniform_(self.lo, self.hi)
        return (img - mean) * factor + mean


@register("saturation")
class RandomSaturation(PerImageAugmentation):
    """HSV saturation jitter."""
    def __init__(self, lo: float = 0.6, hi: float = 1.4, p: float = 0.8):
        super().__init__(p=p)
        self.lo, self.hi = lo, hi

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        gray = img.mean(dim=1, keepdim=True)  # approximate luminance
        factor = torch.empty(img.shape[0], 1, 1, 1, device=img.device, dtype=img.dtype).uniform_(self.lo, self.hi)
        return gray + (img - gray) * factor


@register("hue")
class RandomHue(PerImageAugmentation):
    """Simplified hue shift via channel rolling."""
    def __init__(self, delta: float = 0.1, p: float = 0.5):
        super().__init__(p=p)
        self.delta = delta

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        shift = torch.empty(B, 1, 1, 1, device=img.device, dtype=img.dtype).uniform_(-self.delta, self.delta)
        # RGB → approximate hue shift by weighted channel mixing
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        cos_h = torch.cos(shift * 3.14159 * 2)
        sin_h = torch.sin(shift * 3.14159 * 2)
        new_r = r * (0.299 + 0.701 * cos_h + 0.168 * sin_h) + \
                g * (0.587 - 0.587 * cos_h + 0.330 * sin_h) + \
                b * (0.114 - 0.114 * cos_h - 0.497 * sin_h)
        new_g = r * (0.299 - 0.299 * cos_h - 0.328 * sin_h) + \
                g * (0.587 + 0.413 * cos_h + 0.035 * sin_h) + \
                b * (0.114 - 0.114 * cos_h + 0.292 * sin_h)
        new_b = r * (0.299 - 0.300 * cos_h + 1.250 * sin_h) + \
                g * (0.587 - 0.588 * cos_h - 1.050 * sin_h) + \
                b * (0.114 + 0.886 * cos_h - 0.203 * sin_h)
        return torch.cat([new_r, new_g, new_b], dim=1)


@register("color_jitter")
class RandomColorJitter(PerImageAugmentation):
    """SimCLR-style: brightness + contrast + saturation + hue birleşik."""
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4,
                 saturation: float = 0.2, hue: float = 0.1, p: float = 0.8):
        super().__init__(p=p)
        self._augs = [
            RandomBrightness(lo=max(0, 1 - brightness), hi=1 + brightness, p=1.0),
            RandomContrast(lo=max(0, 1 - contrast), hi=1 + contrast, p=1.0),
            RandomSaturation(lo=max(0, 1 - saturation), hi=1 + saturation, p=1.0),
            RandomHue(delta=hue, p=1.0),
        ]

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        # Rastgele sıralama (SimCLR paper'daki gibi)
        order = torch.randperm(len(self._augs))
        for i in order:
            img = self._augs[i].apply(img)
        return img


@register("grayscale")
class RandomGrayscale(PerImageAugmentation):
    def __init__(self, p: float = 0.2):
        super().__init__(p=p)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        return gray.expand(-1, img.shape[1], -1, -1)


@register("solarize")
class RandomSolarize(PerImageAugmentation):
    """Threshold üstü pikselleri ters çevir (BYOL'de kullanılır)."""
    def __init__(self, threshold: float = 0.5, p: float = 0.2):
        super().__init__(p=p)
        self.threshold = threshold

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        return torch.where(img > self.threshold, 1.0 - img, img)


@register("posterize")
class RandomPosterize(PerImageAugmentation):
    """Bit derinliğini azalt."""
    def __init__(self, bits: int = 4, p: float = 0.2):
        super().__init__(p=p)
        self.bits = bits

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        levels = 2 ** self.bits
        return (img * levels).floor() / levels


@register("equalize")
class RandomEqualize(PerImageAugmentation):
    """Histogram equalization (approximate, differentiable)."""
    def __init__(self, p: float = 0.2):
        super().__init__(p=p)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        flat = img.view(B, C, -1)
        n_pixels = H * W
        n_bins = 256

        batch_out = []
        for b in range(B):
            channels = []
            for c in range(C):
                pixel_vals = (flat[b, c] * (n_bins - 1)).clamp(0, n_bins - 1).long()
                hist = torch.bincount(pixel_vals, minlength=n_bins).float()
                cdf = hist.cumsum(dim=0)
                cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else cdf[0]
                denom = float(n_pixels) - float(cdf_min)
                if denom <= 0:
                    channels.append(flat[b, c])
                    continue
                cdf_normalized = ((cdf - cdf_min) / denom).clamp(0, 1)
                equalized = cdf_normalized[pixel_vals]
                channels.append(equalized)
            batch_out.append(torch.stack(channels))
        return torch.stack(batch_out).view(B, C, H, W)
