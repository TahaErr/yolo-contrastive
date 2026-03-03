"""FrequencyBandPrediction — Frekans bandi tahmin pretext taski.

Novel katki: Frekans domain pretext tasklar zaman serisi SSL'de
kullanilmis (TF-C, TRLS, FreMixer) ancak goruntu SSL + object
detection baglaminda hic denenmemis.

Motivasyon:
    Object detection backbone'u 3 tur bilgi kullanir:
    - Low frequency  -> genel sekil, kontur (shape)
    - Mid frequency  -> doku, pattern (texture)
    - High frequency -> kenar, ince detay (edge)

Pipeline:
    img -> FFT2D -> frekans maskesi uygula -> IFFT2D -> tahmin et

    4 sinif:
        0: none   - orijinal (maskeleme yok)
        1: low    - dusuk frekanslar silinir (sekil kaybi)
        2: mid    - orta frekanslar silinir (doku kaybi)
        3: high   - yuksek frekanslar silinir (kenar kaybi)

Referanslar:
    - TF-C (Zhang et al. 2022): Time-frequency contrastive
    - TRLS (2024): Spectrogram-based representation learning
    - IE-Rot (Yamaguchi et al. 2019): Multi-task pretext
    - Bu calisma: Frekans domain pretext -> goruntu SSL (ilk kez)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BasePretextTask, register_task


@register_task("freq_band")
class FrequencyBandPrediction(BasePretextTask):
    """Frekans bandi maskeleme tahmini.

    Goruntuye 2D FFT uygular, rastgele bir frekans bandini sifirlar,
    IFFT ile geri donusturur. Model hangi bandin silindigini tahmin eder.

    Siniflar:
        0: none - orijinal goruntu
        1: low  - dusuk frekanslar silinir (r < r_low)
        2: mid  - orta frekanslar silinir (r_low < r < r_high)
        3: high - yuksek frekanslar silinir (r > r_high)
    """

    BAND_NAMES = ["none", "low", "mid", "high"]

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 256,
        low_ratio: float = 0.1,
        mid_ratio: float = 0.4,
        smooth_width: float = 0.02,
    ):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim)
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
        return 4

    @property
    def difficulty(self) -> str:
        return "hard"

    def _get_distance_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Frekans uzayinda merkeze normalize mesafe gridi."""
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

    def _make_band_mask(self, dist: torch.Tensor, band: int) -> torch.Tensor:
        """Belirtilen frekans bandi icin maskeleme tensoru.

        Mask = 1 -> korunur, Mask = 0 -> silinir.
        Smooth sigmoid gecis (ringing artefaktlarini azaltir).
        """
        if band == 0:
            return torch.ones_like(dist)

        w = max(self.smooth_width, 1e-4)

        if band == 1:
            # Low: dusuk frekanslari sil (merkez)
            return torch.sigmoid((dist - self.low_ratio) / w)

        elif band == 2:
            # Mid: orta frekanslari sil
            keep_low = torch.sigmoid((self.low_ratio - dist) / w)
            keep_high = torch.sigmoid((dist - self.mid_ratio) / w)
            return (keep_low + keep_high).clamp(0.0, 1.0)

        elif band == 3:
            # High: yuksek frekanslari sil (kenarlar)
            return torch.sigmoid((self.mid_ratio - dist) / w)

        return torch.ones_like(dist)

    def _apply_freq_mask(self, img: torch.Tensor, band: int) -> torch.Tensor:
        """Tek goruntuye frekans bandi maskeleme uygula.

        Pipeline: img -> FFT2D -> shift -> mask -> ishift -> IFFT2D -> clamp
        """
        if band == 0:
            return img

        C, H, W = img.shape
        freq = torch.fft.fft2(img, dim=(-2, -1))
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

        dist = self._get_distance_grid(H, W, img.device)
        mask = self._make_band_mask(dist, band)

        freq_masked = freq_shifted * mask.unsqueeze(0)
        freq_unshifted = torch.fft.ifftshift(freq_masked, dim=(-2, -1))
        img_back = torch.fft.ifft2(freq_unshifted, dim=(-2, -1))

        return img_back.real.clamp(0.0, 1.0)

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batchteki her goruntuye rastgele frekans maskeleme uygula."""
        B = img.shape[0]
        labels = torch.randint(0, 4, (B,), device=img.device)
        out = img.clone()

        for band in range(1, 4):
            mask_idx = (labels == band).nonzero(as_tuple=True)[0]
            if len(mask_idx) == 0:
                continue
            for i in mask_idx:
                out[i] = self._apply_freq_mask(img[i], band)

        return out, labels

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy

    def __repr__(self) -> str:
        return (
            f"FrequencyBandPrediction("
            f"classes={self.num_classes}, "
            f"low={self.low_ratio}, mid={self.mid_ratio}, "
            f"feat_dim={self.feat_dim})"
        )
