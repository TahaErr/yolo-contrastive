"""Pluggable augmentation registry — her augmentation bir sınıf, pipeline compose eder."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Type
import torch


class BaseAugmentation(ABC):
    """Tüm augmentation'ların base class'ı.

    Her augmentation:
        - p: uygulama olasılığı (0=kapalı, 1=her zaman)
        - __call__(img) → img: [B, C, H, W] tensor in, tensor out
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0, f"p must be in [0, 1], got {p}"
        self.p = p

    @abstractmethod
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        """Alt sınıflar bunu implement eder."""
        ...

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return img
        if self.p >= 1.0:
            return self.apply(img)
        if torch.rand((), device=img.device).item() < self.p:
            return self.apply(img)
        return img

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self._params().items())
        return f"{self.__class__.__name__}({params})"

    def _params(self) -> dict:
        return {"p": self.p}


class PerImageAugmentation(BaseAugmentation):
    """Her batch elemanına bağımsız olasılıkla uygulanan augmentation."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return img
        B = img.shape[0]
        mask = torch.rand(B, device=img.device) < self.p  # [B]
        if not mask.any():
            return img
        augmented = self.apply(img)
        # Mask ile seçici uygula
        mask = mask.view(B, 1, 1, 1)
        return torch.where(mask, augmented, img)


class AugmentationPipeline:
    """Augmentation'ları sırayla uygular.

    Kullanım:
        pipeline = AugmentationPipeline([
            RandomHorizontalFlip(p=0.5),
            RandomColorJitter(brightness=0.4, p=0.8),
            RandomGaussianBlur(p=0.5),
        ])
        img2 = pipeline(img)
    """

    def __init__(self, transforms: List[BaseAugmentation]):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "AugmentationPipeline([\n" + "\n".join(lines) + "\n])"

    def __len__(self) -> int:
        return len(self.transforms)


# ── Global Registry ──

_REGISTRY: Dict[str, Type[BaseAugmentation]] = {}


def register(name: str):
    """Decorator: augmentation sınıfını registry'ye ekler."""
    def wrapper(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return wrapper


def get_augmentation(name: str, **kwargs) -> BaseAugmentation:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown augmentation '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key](**kwargs)


def list_augmentations() -> List[str]:
    return sorted(_REGISTRY.keys())
