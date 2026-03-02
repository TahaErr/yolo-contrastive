"""Silme/maskeleme augmentation'ları — cutout, random erasing, grid mask."""

from __future__ import annotations
import torch
from .registry import PerImageAugmentation, register


@register("cutout")
class RandomCutout(PerImageAugmentation):
    """Rastgele dikdörtgen bölge sil (sıfırla)."""
    def __init__(self, num_holes: int = 1, max_h: int = 32, max_w: int = 32, p: float = 0.5):
        super().__init__(p=p)
        self.num_holes = num_holes
        self.max_h = max_h
        self.max_w = max_w

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        out = img.clone()
        for _ in range(self.num_holes):
            h = torch.randint(1, self.max_h + 1, (B,))
            w = torch.randint(1, self.max_w + 1, (B,))
            y = torch.randint(0, H, (B,))
            x = torch.randint(0, W, (B,))
            for i in range(B):
                y1, y2 = int(y[i]), min(int(y[i] + h[i]), H)
                x1, x2 = int(x[i]), min(int(x[i] + w[i]), W)
                out[i, :, y1:y2, x1:x2] = 0.0
        return out


@register("random_erasing")
class RandomErasing(PerImageAugmentation):
    """Random Erasing (Zhong et al.) — bölgeyi rastgele değerle doldur."""
    def __init__(self, scale_lo: float = 0.02, scale_hi: float = 0.33,
                 ratio_lo: float = 0.3, ratio_hi: float = 3.3, p: float = 0.5):
        super().__init__(p=p)
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        out = img.clone()
        area = H * W
        for i in range(B):
            target_area = torch.empty(1).uniform_(self.scale_lo, self.scale_hi).item() * area
            aspect = torch.empty(1).uniform_(self.ratio_lo, self.ratio_hi).item()
            eh = int(round((target_area * aspect) ** 0.5))
            ew = int(round((target_area / aspect) ** 0.5))
            if eh < H and ew < W:
                y = torch.randint(0, H - eh, (1,)).item()
                x = torch.randint(0, W - ew, (1,)).item()
                out[i, :, y:y+eh, x:x+ew] = torch.rand(C, eh, ew, device=img.device, dtype=img.dtype)
        return out


@register("grid_mask")
class GridMask(PerImageAugmentation):
    """Grid pattern ile maskeleme."""
    def __init__(self, ratio: float = 0.5, grid_size: int = 16, p: float = 0.5):
        super().__init__(p=p)
        self.ratio = ratio
        self.grid_size = grid_size

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        mask_h = int(self.grid_size * self.ratio)
        mask_w = int(self.grid_size * self.ratio)
        mask = torch.ones(H, W, device=img.device, dtype=img.dtype)
        for y in range(0, H, self.grid_size):
            for x in range(0, W, self.grid_size):
                mask[y:y+mask_h, x:x+mask_w] = 0.0
        off_y = torch.randint(0, self.grid_size, (1,)).item()
        off_x = torch.randint(0, self.grid_size, (1,)).item()
        mask = torch.roll(mask, shifts=(off_y, off_x), dims=(0, 1))
        return img * mask.unsqueeze(0).unsqueeze(0)
