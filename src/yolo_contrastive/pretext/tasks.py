"""Yeni pretext task'lar — IE-Rot ve literatürden esinlenildi.

Task'lar:
    SolarizationTask   — 4-class: solarization seviyesi (IE-Rot paper)
    ColorPermutationTask — 6-class: RGB kanal permütasyonu
    PatchShuffleTask    — 24-class: 2×2 grid permütasyonu (Jigsaw-lite)
    BlurPredictionTask  — 4-class: Gaussian blur seviyesi

Her task BasePretextTask arayüzüne uyar:
    transform(img) → (augmented_img, labels)
    forward(features, labels) → (loss, accuracy)
"""

from __future__ import annotations

import itertools
from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BasePretextTask, register_task


# ═══════════════════════════════════════════════════════════════
# 1) SolarizationTask — IE-Rot paper'dan
# ═══════════════════════════════════════════════════════════════

@register_task("solarization")
class SolarizationTask(BasePretextTask):
    """Solarization seviyesini tahmin et.

    4 seviye:
        0: none (orijinal)
        1: light  — threshold=0.75 (sadece çok parlak pikseller ters)
        2: medium — threshold=0.50
        3: heavy  — threshold=0.25 (neredeyse tüm pikseller ters)

    IE-Rot paper: Rotation şekil öğretir, Solarization doku öğretir.
    Birlikte kullanıldığında her ikisini de yakalar.
    """

    THRESHOLDS = [None, 0.75, 0.50, 0.25]  # None = no solarization

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self._build_head()

    @property
    def task_name(self) -> str:
        return "solarization"

    @property
    def num_classes(self) -> int:
        return 4

    @property
    def difficulty(self) -> str:
        return "medium"

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Her örneğe rastgele solarization seviyesi uygula.

        Solarization: threshold üstü pikseller ters çevrilir → (1 - pixel)
        """
        B = img.shape[0]
        labels = torch.randint(0, 4, (B,), device=img.device)
        out = img.clone()

        for i in range(B):
            level = labels[i].item()
            if level > 0:  # 0 = no change
                thresh = self.THRESHOLDS[level]
                out[i] = torch.where(img[i] > thresh, 1.0 - img[i], img[i])

        return out, labels

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy


# ═══════════════════════════════════════════════════════════════
# 2) ColorPermutationTask
# ═══════════════════════════════════════════════════════════════

@register_task("color_perm")
class ColorPermutationTask(BasePretextTask):
    """RGB kanal permütasyonunu tahmin et.

    6 permütasyon:
        0: RGB (orijinal)
        1: RBG
        2: GRB
        3: GBR
        4: BRG
        5: BGR

    Model hangi kanalların yer değiştirdiğini anlamak için
    renk-şekil ilişkisini öğrenmeli (gökyüzü=mavi, çimen=yeşil, vb.)
    """

    # Tüm 3! = 6 permütasyon
    PERMS = list(itertools.permutations([0, 1, 2]))  # [(0,1,2), (0,2,1), ...]

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self._build_head()

    @property
    def task_name(self) -> str:
        return "color_perm"

    @property
    def num_classes(self) -> int:
        return 6

    @property
    def difficulty(self) -> str:
        return "hard"

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Her örneğe rastgele kanal permütasyonu uygula."""
        B, C, H, W = img.shape
        assert C == 3, f"ColorPermutationTask requires 3-channel input, got {C}"

        labels = torch.randint(0, 6, (B,), device=img.device)
        out = img.clone()

        for i in range(B):
            perm = self.PERMS[labels[i].item()]
            out[i] = img[i, perm, :, :]  # kanal sırasını değiştir

        return out, labels

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy


# ═══════════════════════════════════════════════════════════════
# 3) PatchShuffleTask — Jigsaw-lite
# ═══════════════════════════════════════════════════════════════

@register_task("patch_shuffle")
class PatchShuffleTask(BasePretextTask):
    """2×2 grid patch permütasyonunu tahmin et.

    Görüntü 4 parçaya bölünür, karıştırılır, model orijinal
    düzeni tahmin eder. 4! = 24 olası permütasyon.

    Jigsaw puzzle'ın basitleştirilmiş versiyonu — uzamsal düzen öğretir.
    """

    # Tüm 4! = 24 permütasyon
    PERMS = list(itertools.permutations([0, 1, 2, 3]))

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self._build_head()

    @property
    def task_name(self) -> str:
        return "patch_shuffle"

    @property
    def num_classes(self) -> int:
        return 24

    @property
    def difficulty(self) -> str:
        return "hard"

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Görüntüyü 2×2 grid'e böl, rastgele karıştır."""
        B, C, H, W = img.shape
        h2, w2 = H // 2, W // 2

        # 4 patch çıkar: sol-üst, sağ-üst, sol-alt, sağ-alt
        patches = [
            img[:, :, :h2, :w2],       # 0: sol-üst
            img[:, :, :h2, w2:2*w2],   # 1: sağ-üst
            img[:, :, h2:2*h2, :w2],   # 2: sol-alt
            img[:, :, h2:2*h2, w2:2*w2],  # 3: sağ-alt
        ]

        labels = torch.randint(0, 24, (B,), device=img.device)
        out = torch.zeros(B, C, h2 * 2, w2 * 2, device=img.device, dtype=img.dtype)

        for i in range(B):
            perm = self.PERMS[labels[i].item()]
            # Permütasyona göre patch'leri yerleştir
            out[i, :, :h2, :w2]          = patches[perm[0]][i]
            out[i, :, :h2, w2:2*w2]      = patches[perm[1]][i]
            out[i, :, h2:2*h2, :w2]      = patches[perm[2]][i]
            out[i, :, h2:2*h2, w2:2*w2]  = patches[perm[3]][i]

        return out, labels

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy


# ═══════════════════════════════════════════════════════════════
# 4) BlurPredictionTask
# ═══════════════════════════════════════════════════════════════

@register_task("blur")
class BlurPredictionTask(BasePretextTask):
    """Gaussian blur seviyesini tahmin et.

    4 seviye:
        0: none (orijinal, net görüntü)
        1: light  — sigma=0.5, kernel=3
        2: medium — sigma=1.5, kernel=5
        3: heavy  — sigma=3.0, kernel=7

    Model kenar keskinliğini analiz ederek blur seviyesini anlamalı.
    Doku ve frekans bilgisi öğretir.
    """

    # (sigma, kernel_size) çiftleri
    LEVELS = [
        None,              # 0: no blur
        (0.5, 3),          # 1: light
        (1.5, 5),          # 2: medium
        (3.0, 7),          # 3: heavy
    ]

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self._build_head()
        self._kernel_cache: dict = {}

    @property
    def task_name(self) -> str:
        return "blur"

    @property
    def num_classes(self) -> int:
        return 4

    @property
    def difficulty(self) -> str:
        return "medium"

    def _get_kernel(self, sigma: float, k: int,
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Gaussian kernel oluştur veya cache'den al."""
        cache_key = (sigma, k, str(device), str(dtype))
        if cache_key not in self._kernel_cache:
            coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            kernel_2d = torch.outer(g, g)
            kernel_2d = (kernel_2d / kernel_2d.sum()).view(1, 1, k, k).repeat(3, 1, 1, 1)
            self._kernel_cache[cache_key] = kernel_2d
        return self._kernel_cache[cache_key]

    def _apply_blur(self, img: torch.Tensor, sigma: float, k: int) -> torch.Tensor:
        """Tek bir batch'e Gaussian blur uygula."""
        kernel = self._get_kernel(sigma, k, img.device, img.dtype)
        pad = k // 2
        return F.conv2d(
            F.pad(img, (pad, pad, pad, pad), mode="reflect"),
            kernel, groups=3,
        )

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Her örneğe rastgele blur seviyesi uygula."""
        B = img.shape[0]
        labels = torch.randint(0, 4, (B,), device=img.device)
        out = img.clone()

        # Seviye bazında grupla — aynı seviyedeki örnekleri toplu blur et
        for level in range(1, 4):
            mask = (labels == level)
            if not mask.any():
                continue
            sigma, k = self.LEVELS[level]
            indices = mask.nonzero(as_tuple=True)[0]
            blurred = self._apply_blur(img[indices], sigma, k)
            out[indices] = blurred

        return out, labels

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy
