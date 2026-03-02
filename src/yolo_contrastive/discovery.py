"""Dataset discovery — klasör yapısını analiz edip eğitim modunu belirler.

Üç senaryo:
    data.yaml ✅ + unlabeled/ ✅ → SSL Pretrain → Fine-tune
    data.yaml ✅ + unlabeled/ ❌ → Direkt detection eğitimi
    data.yaml ❌ + unlabeled/ ✅ → Sadece backbone pretrain
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from .exceptions import ConfigError


class TrainMode(Enum):
    """Auto-determined training mode."""
    SSL_FINETUNE = "ssl_finetune"       # unlabeled + labeled
    DETECTION = "detection"              # sadece labeled
    SSL_ONLY = "ssl_only"               # sadece unlabeled


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _count_images(path: str) -> int:
    """Recursively count images in directory."""
    if not os.path.isdir(path):
        return 0
    count = 0
    for f in Path(path).rglob("*"):
        if f.suffix.lower() in IMAGE_EXTS and f.is_file():
            count += 1
    return count


def _resolve_path(base_dir: str, rel_path: str) -> str:
    """Resolve relative path against base_dir."""
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(Path(base_dir) / p)


@dataclass
class DatasetInfo:
    """Dataset analysis result."""

    mode: TrainMode
    data_yaml: Optional[str] = None
    unlabeled_dir: Optional[str] = None

    # Labeled dataset bilgileri
    train_images_dir: Optional[str] = None
    val_images_dir: Optional[str] = None
    test_images_dir: Optional[str] = None
    num_classes: int = 0
    class_names: Optional[list] = None

    # İstatistikler
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_unlabeled: int = 0

    def summary(self) -> str:
        lines = [
            f"Mode:       {self.mode.value}",
            f"Labeled:    train={self.n_train}, val={self.n_val}, test={self.n_test}",
            f"Unlabeled:  {self.n_unlabeled}",
            f"Classes:    {self.num_classes}",
        ]
        if self.mode == TrainMode.SSL_FINETUNE:
            lines.append("Pipeline:   SSL Pretrain → Fine-tune")
        elif self.mode == TrainMode.DETECTION:
            lines.append("Pipeline:   Detection training only")
        elif self.mode == TrainMode.SSL_ONLY:
            lines.append("Pipeline:   SSL backbone pretrain only")
        return "\n".join(lines)


def discover(
    data_yaml: Optional[str] = None,
    unlabeled_dir: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> DatasetInfo:
    """Analyze dataset structure and determine training mode.

    Args:
        data_yaml: YOLO data.yaml yolu (None ise dataset_dir'de arar)
        unlabeled_dir: Etiketsiz görüntü klasörü (None ise otomatik arar)
        dataset_dir: Üst dataset klasörü (data.yaml ve unlabeled/ burada aranır)

    Returns:
        DatasetInfo: Algılanan yapı ve mod

    Raises:
        ConfigError: Ne labeled ne unlabeled veri bulunamazsa
    """
    info = DatasetInfo(mode=TrainMode.DETECTION)

    # --- data.yaml bul ---
    if data_yaml and os.path.isfile(data_yaml):
        info.data_yaml = str(data_yaml)
    elif dataset_dir:
        # dataset_dir içinde data.yaml ara
        for name in ["data.yaml", "data.yml", "dataset.yaml"]:
            candidate = os.path.join(dataset_dir, name)
            if os.path.isfile(candidate):
                info.data_yaml = candidate
                break

    # --- data.yaml parse et ---
    has_labeled = False
    yaml_base_dir = None

    if info.data_yaml:
        yaml_base_dir = str(Path(info.data_yaml).parent)
        try:
            with open(info.data_yaml) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"data.yaml okunamadı: {info.data_yaml}: {e}")

        # Sınıf bilgileri
        names = cfg.get("names", {})
        if isinstance(names, dict):
            info.num_classes = len(names)
            info.class_names = list(names.values())
        elif isinstance(names, list):
            info.num_classes = len(names)
            info.class_names = names

        # Train / val / test yolları
        if "train" in cfg:
            info.train_images_dir = _resolve_path(yaml_base_dir, cfg["train"])
            info.n_train = _count_images(info.train_images_dir)

        if "val" in cfg:
            info.val_images_dir = _resolve_path(yaml_base_dir, cfg["val"])
            info.n_val = _count_images(info.val_images_dir)

        if "test" in cfg:
            info.test_images_dir = _resolve_path(yaml_base_dir, cfg["test"])
            info.n_test = _count_images(info.test_images_dir)

        has_labeled = info.n_train > 0

        # data.yaml içinde unlabeled tanımlı mı?
        if unlabeled_dir is None and "unlabeled" in cfg:
            unlabeled_candidate = _resolve_path(yaml_base_dir, cfg["unlabeled"])
            if os.path.isdir(unlabeled_candidate):
                unlabeled_dir = unlabeled_candidate

    # --- unlabeled klasörünü bul ---
    has_unlabeled = False

    if unlabeled_dir and os.path.isdir(unlabeled_dir):
        info.unlabeled_dir = str(unlabeled_dir)
    elif dataset_dir:
        # dataset_dir/unlabeled/ konvansiyonu
        candidate = os.path.join(dataset_dir, "unlabeled")
        if os.path.isdir(candidate):
            info.unlabeled_dir = candidate
    elif yaml_base_dir:
        # data.yaml yanında unlabeled/ klasörü
        candidate = os.path.join(yaml_base_dir, "unlabeled")
        if os.path.isdir(candidate):
            info.unlabeled_dir = candidate

    if info.unlabeled_dir:
        info.n_unlabeled = _count_images(info.unlabeled_dir)
        has_unlabeled = info.n_unlabeled > 0

    # --- Mod belirle ---
    if has_labeled and has_unlabeled:
        info.mode = TrainMode.SSL_FINETUNE
    elif has_labeled:
        info.mode = TrainMode.DETECTION
    elif has_unlabeled:
        info.mode = TrainMode.SSL_ONLY
    else:
        raise ConfigError(
            "Ne etiketli ne etiketsiz veri bulunamadı. "
            "data_yaml, unlabeled_dir veya dataset_dir parametrelerini kontrol edin."
        )

    return info
