"""Unlabeled image dataset — etiket gerekmez, sadece görüntü klasörü yeter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import warnings

import torch
from torch.utils.data import Dataset


class UnlabeledImageDataset(Dataset):
    """Klasördeki tüm görüntüleri recursive yükler."""

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
        # FIX: Bozuk dosyada random tensor yerine komşu index'e fallback
        max_retries = 3
        for attempt in range(max_retries):
            try_idx = (idx + attempt) % len(self.files)
            path = str(self.files[try_idx])
            img = cv2.imread(path)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.imgsz, self.imgsz),
                                 interpolation=cv2.INTER_LINEAR)
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img).permute(2, 0, 1).contiguous()

            if attempt == 0:
                warnings.warn(
                    f"Corrupt/unreadable file: {path} — trying next image.",
                    UserWarning, stacklevel=2,
                )

        # Tüm retry'lar başarısız — son çare
        warnings.warn(
            f"All retries failed around index {idx}. Returning zero tensor.",
            UserWarning, stacklevel=2,
        )
        return torch.zeros(3, self.imgsz, self.imgsz)

    def __repr__(self) -> str:
        return f"UnlabeledImageDataset(n={len(self)}, imgsz={self.imgsz})"
