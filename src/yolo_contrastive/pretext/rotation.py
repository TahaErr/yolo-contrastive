"""Rotation prediction pretext task (RotNet-style).

Refactored: BasePretextTask'tan türetildi + registry'ye kayıtlı.
Backward compatible: rotate_batch() hâlâ çalışır.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BasePretextTask, register_task


@register_task("rotation")
class RotationTask(BasePretextTask):
    """Rotation prediction: 0° / 90° / 180° / 270°."""

    ANGLES = [0, 90, 180, 270]

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self._build_head()

    @property
    def task_name(self) -> str:
        return "rotation"

    @property
    def num_classes(self) -> int:
        return 4

    @property
    def difficulty(self) -> str:
        return "trivial"

    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = img.shape[0]
        labels = torch.randint(0, 4, (B,), device=img.device)
        rotated = img.clone()
        for i in range(B):
            k = labels[i].item()
            if k > 0:
                rotated[i] = torch.rot90(img[i], k, dims=[1, 2])
        return rotated, labels

    def rotate_batch(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy alias → transform()."""
        return self.transform(img)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(features)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        return loss, accuracy
