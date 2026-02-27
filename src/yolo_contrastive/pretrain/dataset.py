"""Unlabeled image dataset — etiket gerekmez, sadece görüntü klasörü yeter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import warnings

import torch
from torch.utils.data import Dataset


class UnlabeledImageDataset(Dataset):
    """Klasördeki tüm görüntüleri recursive yükler.

    Kullanım:
        ds = UnlabeledImageDataset("/path/to/images", imgsz=640)
        img = ds[0]  # [3, 640, 640] float32 tensor, range [0, 1]
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root: str, imgsz: int = 640):
        self.imgsz = imgsz
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Image directory not found: {root}")

        self.files = sorted([
            f for f in root_path.rglob("*")
            if f.suffix.lower() in self.EXTS and f.is_file()
        ])
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = str(self.files[idx])
        img = cv2.imread(path)

        if img is None:
            warnings.warn(
                f"Bozuk/okunamayan dosya: {path} — rastgele tensor döndürülüyor.",
                UserWarning, stacklevel=2,
            )
            return torch.rand(3, self.imgsz, self.imgsz)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox yerine basit resize (pretraining için yeterli)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # HWC uint8 → CHW float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def __repr__(self) -> str:
        return f"UnlabeledImageDataset(n={len(self)}, imgsz={self.imgsz})"
