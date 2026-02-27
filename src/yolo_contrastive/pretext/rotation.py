"""Rotation prediction pretext task (RotNet-style).

Backbone'a döndürülmüş görüntü verilir, model kaç derece döndürüldüğünü tahmin eder.
Bu sayede backbone geometrik feature'lar öğrenir — az etiketli veri ile fine-tune'a hazırlanır.

Akış:
    img → rotate(0°|90°|180°|270°) → backbone → features → PredictionHead → 4-class CE loss
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import PredictionHead


class RotationTask(nn.Module):
    """Rotation prediction pretext task.

    Her batch'ten:
    1. Görüntüleri rastgele 0°/90°/180°/270° döndür
    2. Backbone'dan feature çıkar
    3. PredictionHead ile açıyı tahmin et
    4. Cross-entropy loss

    Kullanım:
        rot_task = RotationTask(feat_dim=256)
        rot_loss, accuracy = rot_task(features_from_backbone, img_batch)
    """

    ANGLES = [0, 90, 180, 270]  # k values: 0, 1, 2, 3

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.head = PredictionHead(
            feat_dim=feat_dim,
            num_classes=4,  # 0°, 90°, 180°, 270°
            hidden_dim=hidden_dim,
        )

    def rotate_batch(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch'teki her görüntüyü rastgele açıyla döndür.

        Returns:
            rotated_img: [B, C, H, W]
            labels: [B] — 0=0°, 1=90°, 2=180°, 3=270°
        """
        B = img.shape[0]
        labels = torch.randint(0, 4, (B,), device=img.device)
        rotated = img.clone()
        for i in range(B):
            k = labels[i].item()
            if k > 0:
                rotated[i] = torch.rot90(img[i], k, dims=[1, 2])
        return rotated, labels

    def forward(
        self,
        backbone_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Rotation loss hesapla.

        Args:
            backbone_features: [B, D] — backbone'dan çıkan embedding
            labels: [B] — rotation labels (0-3)

        Returns:
            loss: scalar
            accuracy: float (0-1)
        """
        logits = self.head(backbone_features)  # [B, 4]
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

        return loss, accuracy
