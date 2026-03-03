"""FrequencyBandPrediction v2 — 7 sinifli frekans bandi tahmin.

v1'den farklar:
    - 4 sinif → 7 sinif (single + dual-band masking)
    - Label smoothing destegi (BasePretextTask'tan)
    - Dual-band maskeleme pretrained backbone icin cok daha zor

7 sinif:
    0: none      — orijinal (maskeleme yok)
    1: low       — dusuk frekanslar silinir → sekil kaybi
    2: mid       — orta frekanslar silinir → doku kaybi
    3: high      — yuksek frekanslar silinir → kenar kaybi
    4: low+mid   — sadece yuksek frekans kalir → sadece kenarlar
    5: low+high  — sadece orta frekans kalir → sadece doku
    6: mid+high  — sadece dusuk frekans kalir → sadece sekil

Dual-band maskeleme tek bilgi ekseni birakir → backbone
hangi eksenin kaldigini tanimlamak zorunda → cok zor.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BasePretextTask, register_task


@register_task("freq_band")
class FrequencyBandPrediction(BasePretextTask):
    """7 sinifli frekans bandi maskeleme tahmini."""

    BAND_NAMES = ["none", "low", "mid", "high", "low+mid", "low+high", "mid+high"]

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 256,
        label_smoothing: float = 0.15,
        low_ratio: float = 0.1,
        mid_ratio: float = 0.4,
        smooth_width: float = 0.02,
    ):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim,
                         label_smoothing=label_smoothing)
        self.low_ratio = low_ratio
        self.mid_ratio = mid_ratio
        self.smooth_width = smooth_width
        self._mask_cache = {}
        self._build_head()

    @property
    def task_name(self) -> str:
        return "freq_band"

    @property
    def num_classes(self) -> int:
        return 7

    @property
    def difficulty(self) -> str:
        return "hard"

    def _get_distance_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        fy = torch.arange(H, device=device, dtype=torch.float32) - H / 2.0
        fx = torch.arange(W, device=device, dtype=torch.float32) - W / 2.0
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")

        max_r = (H**2 / 4.0 + W**2 / 4.0) ** 0.5
        dist = (gy**2 + gx**2).sqrt() / max_r

        if len(self._mask_cache) < 8:
            self._mask_cache[key] = dist
        return dist

    def _single_band_mask(self, dist: torch.Tensor, band: str) -> torch.Tensor:
        """Tek band maskesi: low, mid veya high."""
        w = max(self.smooth_width, 1e-4)

        if band == "low":
            return torch.sigmoid((dist - self.low_ratio) / w)
        elif band == "mid":
            keep_low = torch.sigmoid((self.low_ratio - dist) / w)
            keep_high = torch.sigmoid((dist - self.mid_ratio) / w)
            return (keep_low + keep_high).clamp(0.0, 1.0)
        elif band == "high":
            return torch.sigmoid((self.mid_ratio - dist) / w)
        return torch.ones_like(dist)

    def _make_band_mask(self, dist: torch.Tensor, band_id: int) -> torch.Tensor:
        """7 sinif icin maske olustur.

        0: none      → full mask (hepsi korunur)
        1: low       → low silinir
        2: mid       → mid silinir
        3: high      → high silinir
        4: low+mid   → low VE mid silinir (sadece high kalir)
        5: low+high  → low VE high silinir (sadece mid kalir)
        6: mid+high  → mid VE high silinir (sadece low kalir)
        """
        if band_id == 0:
            return torch.ones_like(dist)

        if band_id <= 3:
            # Single-band removal
            names = ["low", "mid", "high"]
            return self._single_band_mask(dist, names[band_id - 1])

        # Dual-band removal: iki maskeyi carpariz
        # (her iki bandin da silinmesi = iki maskenin minimum'u)
        if band_id == 4:  # low+mid removed → only high remains
            m_low = self._single_band_mask(dist, "low")
            m_mid = self._single_band_mask(dist, "mid")
            return (m_low * m_mid).clamp(0.0, 1.0)
        elif band_id == 5:  # low+high removed → only mid remains
            m_low = self._single_band_mask(dist, "low")
            m_high = self._single_band_mask(dist, "high")
            return (m_low * m_high).clamp(0.0, 1.0)
        elif band_id == 6:  # mid+high removed → only low remains
            m_mid = self._single_band_mask(dist, "mid")
            m_high = self._single_band_mask(dist, "high")
            return (m_mid * m_high).clamp(0.0, 1.0)

        return torch.ones_like(dist)

    def _apply_freq_mask(self, img: torch.Tensor, band_id: int) -> torch.Tensor:
        if band_id == 0:
            return img

        C, H, W = img.shape
        freq = torch.fft.fft2(img, dim=(-2, -1))
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

        dist = self._get_distance_grid(H, W, img.device)
        mask = self._make_band_mask(dist, band_id)

        freq_masked = freq_shifted * mask.unsqueeze(0)
        freq_unshifted = torch.fft.ifftshift(freq_masked, dim=(-2, -1))
        img_back = torch.fft.ifft2(freq_unshifted, dim=(-2, -1))

        return img_back.real.clamp(0.0, 1.0)

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = img.shape[0]
        labels = torch.randint(0, 7, (B,), device=img.device)
        out = img.clone()

        for band_id in range(1, 7):
            mask_idx = (labels == band_id).nonzero(as_tuple=True)[0]
            if len(mask_idx) == 0:
                continue
            for i in mask_idx:
                out[i] = self._apply_freq_mask(img[i], band_id)

        return out, labels

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy

    def __repr__(self) -> str:
        return (
            f"FrequencyBandPrediction("
            f"classes={self.num_classes}, "
            f"low={self.low_ratio}, mid={self.mid_ratio}, "
            f"ls={self.label_smoothing}, "
            f"feat_dim={self.feat_dim})"
        )
