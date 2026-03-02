"""Rotation prediction pretext task (RotNet-style)."""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import PredictionHead


class RotationTask(nn.Module):
    """Rotation prediction pretext task."""

    ANGLES = [0, 90, 180, 270]  # k values: 0, 1, 2, 3

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.head = PredictionHead(
            feat_dim=feat_dim,
            num_classes=4,
            hidden_dim=hidden_dim,
        )

    def rotate_batch(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch'teki her görüntüyü rastgele açıyla döndür.

        FIX: k değerine göre grupla — Python for loop yerine vectorized rot90.
        """
        B = img.shape[0]
        labels = torch.randint(0, 4, (B,), device=img.device)
        rotated = img.clone()

        for k in range(1, 4):
            mask = labels == k
            if mask.any():
                rotated[mask] = torch.rot90(img[mask], k, dims=[2, 3])

        return rotated, labels

    def forward(
        self,
        backbone_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        logits = self.head(backbone_features)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

        return loss, accuracy
