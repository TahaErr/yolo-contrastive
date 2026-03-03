"""CompositeTask — birden fazla pretext task'ı birleştirir.

IE-Rot paper'dan esinlenildi:
    - Tek görüntüye birden fazla transform sırayla uygulanır
    - Backbone tek forward pass yapar
    - Her task kendi head'i ile tahmin yapar
    - Total loss = Σ weight_i × task_i_loss

Bu tasarım pretrained backbone için trivial olmayan bir sinyal üretir:
    Rotation tek başına → trivial (acc=100%)
    Rotation + Solarization + Blur birlikte → zor (her biri diğerini bozuyor)

Kullanım:
    composite = CompositeTask.from_names(
        ["rotation", "solarization", "blur"],
        feat_dim=256,
        weights=[1.0, 1.0, 0.5],
    )

    # Transform: tüm augmentation'lar sırayla uygulanır
    augmented_img, labels_dict = composite.transform(img)

    # Backbone forward (trainer tarafında):
    _ = model(augmented_img)
    features = feature_tap.get_embedding()

    # Tüm task'lar tek embedding'den tahmin:
    total_loss, avg_acc, details = composite(features, labels_dict)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import BasePretextTask, get_task


class CompositeTask(nn.Module):
    """Birden fazla pretext task'ı birleştirir.

    Transform sırası önemli:
        - Her task görüntüyü sırayla dönüştürür
        - Rotation sonrası solarization → model ikisini de çözmeli
        - Bu, tek task'a göre çok daha zor → trivial değil

    Args:
        tasks: BasePretextTask listesi
        weights: Her task için loss ağırlığı (default: hepsi 1.0)
    """

    def __init__(
        self,
        tasks: List[BasePretextTask],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if len(tasks) == 0:
            raise ValueError("CompositeTask en az 1 task gerektirir")

        self.tasks = nn.ModuleList(tasks)
        self.weights = weights or [1.0] * len(tasks)

        if len(self.weights) != len(self.tasks):
            raise ValueError(
                f"weights ({len(self.weights)}) ve tasks ({len(self.tasks)}) "
                f"uzunlukları eşleşmiyor"
            )

    @classmethod
    def from_names(
        cls,
        names: List[str],
        feat_dim: int,
        hidden_dim: int = 256,
        weights: Optional[List[float]] = None,
    ) -> "CompositeTask":
        """Registry'den task adlarıyla oluştur.

        Args:
            names: task adları (ör. ["rotation", "solarization", "blur"])
            feat_dim: backbone feature boyutu
            hidden_dim: her head'in hidden dim'i
            weights: loss ağırlıkları (default: hepsi 1.0)

        Kullanım:
            composite = CompositeTask.from_names(
                ["rotation", "solarization", "blur"],
                feat_dim=256,
            )
        """
        tasks = [get_task(n, feat_dim=feat_dim, hidden_dim=hidden_dim) for n in names]
        return cls(tasks=tasks, weights=weights)

    @property
    def task_names(self) -> List[str]:
        return [t.task_name for t in self.tasks]

    @property
    def total_classes(self) -> int:
        """Toplam sınıf sayısı (loglama için)."""
        return sum(t.num_classes for t in self.tasks)

    @property
    def num_heads(self) -> int:
        return len(self.tasks)

    def transform(
        self,
        img: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Tüm task transform'larını sırayla uygula.

        Her task görüntüyü alır, dönüştürür, label üretir.
        Sonraki task öncekinin çıktısını alır → kümülatif etki.

        Args:
            img: [B, C, H, W] orijinal görüntü

        Returns:
            augmented: [B, C, H, W] — tüm transform'lar uygulanmış
            labels: {task_name: [B] tensor} — her task'ın label'ları
        """
        labels: Dict[str, torch.Tensor] = {}
        x = img

        for task in self.tasks:
            x, task_labels = task.transform(x)
            labels[task.task_name] = task_labels

        return x, labels

    def forward(
        self,
        features: torch.Tensor,
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, float, Dict[str, dict]]:
        """Tek embedding'den tüm task'ları tahmin et.

        Args:
            features: [B, D] — backbone embedding
            labels: {task_name: [B]} — transform()'dan gelen labels

        Returns:
            total_loss: ağırlıklı toplam loss (scalar, requires_grad=True)
            avg_accuracy: tüm task'ların ortalama accuracy'si
            details: {task_name: {"loss": tensor, "acc": float, "weight": float}}
        """
        total_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype,
                                  requires_grad=True)
        total_acc = 0.0
        details: Dict[str, dict] = {}

        for task, weight in zip(self.tasks, self.weights):
            task_labels = labels.get(task.task_name)
            if task_labels is None:
                continue

            loss, acc = task(features, task_labels)
            total_loss = total_loss + weight * loss
            total_acc += acc

            details[task.task_name] = {
                "loss": loss,
                "acc": acc,
                "weight": weight,
            }

        n = max(1, len(details))
        avg_acc = total_acc / n

        return total_loss, avg_acc, details

    # ── Backward compat: trainer RotationTask arayüzü ──

    def rotate_batch(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Legacy compat — transform()'a yönlendirir.

        NOT: labels artık dict, tek tensor değil.
        Trainer bu değişikliğe adapte edilmeli (Parça 5).
        """
        return self.transform(img)

    # ── Utility ──

    def log_summary(self) -> str:
        """Loglama için özet string."""
        parts = []
        for task, w in zip(self.tasks, self.weights):
            parts.append(f"{task.task_name}({task.num_classes}cls,w={w:.2f})")
        return f"CompositeTask[{'+'.join(parts)}]"

    def __repr__(self) -> str:
        lines = []
        for task, w in zip(self.tasks, self.weights):
            lines.append(f"  {task.task_name}: {task.num_classes} classes, "
                         f"weight={w:.2f}, difficulty={task.difficulty}")
        return (
            f"CompositeTask(\n"
            + "\n".join(lines)
            + f"\n  total: {self.total_classes} classes, {self.num_heads} heads)"
        )
