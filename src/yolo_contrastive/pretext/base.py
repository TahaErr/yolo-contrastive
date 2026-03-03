"""Base pretext task interface + pluggable registry.

Her pretext task iki iş yapar:
    1. transform(img) → (augmented_img, labels)  — görüntüyü dönüştür + etiket üret
    2. forward(features, labels) → (loss, accuracy) — backbone feature'dan tahmin yap

Registry sistemi augmentation modülüyle aynı pattern'i kullanır:
    @register_task("rotation")
    class RotationTask(BasePretextTask): ...

    task = get_task("rotation", feat_dim=256)
    tasks = list_tasks()  # ["rotation", ...]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn


class BasePretextTask(ABC, nn.Module):
    """Tüm pretext task'ların base class'ı.

    Alt sınıflar implement etmeli:
        - task_name (property): registry adı, loglama için
        - num_classes (property): kaç sınıf tahmin edilecek
        - transform(img): augmentation + label üretimi
        - forward(features, labels): loss + accuracy hesabı
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256, label_smoothing: float = 0.15):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.label_smoothing = label_smoothing
        self.head: nn.Module | None = None

    def _build_head(self) -> None:
        """PredictionHead oluştur. __init__ sonunda çağrılmalı."""
        from .heads import PredictionHead
        self.head = PredictionHead(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
        )

    @property
    @abstractmethod
    def task_name(self) -> str:
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    def difficulty(self) -> str:
        return "medium"

    @abstractmethod
    def transform(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task={self.task_name!r}, "
            f"classes={self.num_classes}, "
            f"feat_dim={self.feat_dim}, "
            f"difficulty={self.difficulty!r})"
        )


# ── Global Registry ──

_TASK_REGISTRY: Dict[str, Type[BasePretextTask]] = {}


def register_task(name: str):
    """Decorator: pretext task sınıfını registry'ye ekler."""
    def wrapper(cls: Type[BasePretextTask]) -> Type[BasePretextTask]:
        key = name.lower()
        if key in _TASK_REGISTRY:
            # Re-register izin ver (module reload durumları için)
            pass
        _TASK_REGISTRY[key] = cls
        return cls
    return wrapper


def get_task(name: str, **kwargs) -> BasePretextTask:
    """Registry'den task oluştur."""
    key = name.lower()
    if key not in _TASK_REGISTRY:
        raise KeyError(
            f"Unknown pretext task '{name}'. "
            f"Available: {sorted(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[key](**kwargs)


def list_tasks() -> List[str]:
    """Kayıtlı tüm pretext task adlarını döndür."""
    return sorted(_TASK_REGISTRY.keys())
