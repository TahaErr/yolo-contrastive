"""CSVLogger — CSV dosyasına metrik logging."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional

from .base import BaseLogger


class CSVLogger(BaseLogger):
    """Her step'te metrikleri CSV'ye yazar.

    Kolonlar dinamik: ilk log_scalars çağrısında belirlenir.

    Args:
        save_dir: CSV dosyasının kaydedileceği klasör
        filename: CSV dosya adı (default: metrics.csv)
    """

    def __init__(self, save_dir: str = ".", filename: str = "metrics.csv",
                 project: str = "", name: str = "",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(project=project, name=name, config=config)
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        self._writer = None
        self._file = None
        self._fieldnames = None
        self._buffer: Dict[str, float] = {}

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        self._resolve_step(step)
        self._buffer[key] = value

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        s = self._resolve_step(step)
        self._buffer.update(metrics)
        self._flush(s)

    def _flush(self, step: int) -> None:
        if not self._buffer:
            return

        row = {"step": step, **self._buffer}

        if self._writer is None:
            os.makedirs(self.save_dir, exist_ok=True)
            self._fieldnames = list(row.keys())
            self._file = open(self.filepath, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        # Yeni kolon eklendiyse handle et
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            # Dosyayı yeniden aç (header güncelle)
            self._file.close()
            self._file = open(self.filepath, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)

        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._file.flush()
        self._buffer.clear()

    def finish(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __repr__(self) -> str:
        return f"CSVLogger(path={self.filepath!r})"
