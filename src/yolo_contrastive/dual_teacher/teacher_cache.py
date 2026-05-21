"""TeacherCache — on-disk feature cache for COCO teacher (§2.4, Faz 5.3).

Teacher feature extraction is expensive and deterministic: features are
computed once per image on the un-augmented original (Karar K2, §10.27),
then reused every epoch. This module is the disk layer for that cache.

Layout (§2.4):
    {cache_root}/{teacher_tag}/
        {image_id}.npz       # P3/P4/P5 feature maps, FP16-compressed
        metadata.json        # teacher tag, levels, channel dims, version

image_id may contain '/' (it mirrors the SSL-pool manifest's dataset-relative
path). Slashes are preserved as sub-directories — collision-free and matching
the §2.4 "{image_id}.npz" spec literally.

Features are stored FP16 (half the disk of FP32, ~360 GB → ~180 GB for the
full pool) and restored to FP32 on load. The precision loss is immaterial for
distillation targets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

LOG = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_CACHE_VERSION = 1


class TeacherCache:
    """Read/write teacher feature maps to a per-image npz cache.

    Args:
        cache_root: root directory holding all teacher caches.
        teacher_tag: sub-directory name identifying this teacher + level set
            (e.g. ``"yolov8x_coco_p3p4p5"``). Different teachers / level sets
            get different tags so caches never collide.
        levels: FPN levels stored per image. Must match what the teacher
            produces.
    """

    def __init__(
        self,
        cache_root: str,
        teacher_tag: str = "yolov8x_coco_p3p4p5",
        levels: tuple = ("P3", "P4", "P5"),
    ):
        self.cache_root = Path(cache_root)
        self.teacher_tag = teacher_tag
        self.levels = tuple(levels)
        self.cache_dir = self.cache_root / self.teacher_tag

    # ── path helpers ─────────────────────────────────────────────────────

    def _npz_path(self, image_id: str) -> Path:
        """Cache file path for image_id (slashes → sub-dirs)."""
        return self.cache_dir / f"{image_id}.npz"

    # ── single-image I/O ─────────────────────────────────────────────────

    def has(self, image_id: str) -> bool:
        """True if image_id is already cached."""
        return self._npz_path(image_id).exists()

    def __contains__(self, image_id: str) -> bool:
        return self.has(image_id)

    def save(self, image_id: str, features: Dict[str, torch.Tensor]) -> None:
        """Write per-level feature maps for image_id as FP16-compressed npz.

        Args:
            image_id: cache key.
            features: ``{level: tensor}`` — each tensor any shape; stored as
                FP16. Must contain every configured level.
        """
        missing = [lv for lv in self.levels if lv not in features]
        if missing:
            raise ValueError(
                f"features missing levels {missing}; "
                f"got {sorted(features.keys())}"
            )
        path = self._npz_path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            lv: features[lv].detach().cpu().numpy().astype(np.float16)
            for lv in self.levels
        }
        np.savez_compressed(path, **arrays)

    def load(self, image_id: str) -> Dict[str, torch.Tensor]:
        """Load cached feature maps for image_id, restored to FP32.

        Raises:
            FileNotFoundError: if image_id is not cached.
        """
        path = self._npz_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Not cached: {image_id} ({path})")
        with np.load(path) as data:
            return {
                lv: torch.from_numpy(data[lv].astype(np.float32))
                for lv in self.levels
            }

    def __len__(self) -> int:
        """Number of cached images."""
        if not self.cache_dir.exists():
            return 0
        return sum(1 for _ in self.cache_dir.rglob("*.npz"))

    # ── metadata ─────────────────────────────────────────────────────────

    def save_metadata(self, extra: Dict | None = None) -> None:
        """Write metadata.json describing this cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "teacher_tag": self.teacher_tag,
            "levels": list(self.levels),
            "cache_version": _CACHE_VERSION,
        }
        if extra:
            meta.update(extra)
        with open(self.cache_dir / _METADATA_FILENAME, "w") as f:
            json.dump(meta, f, indent=2)

    def load_metadata(self) -> Dict:
        """Read metadata.json.

        Raises:
            FileNotFoundError: if no metadata has been written.
        """
        path = self.cache_dir / _METADATA_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"No metadata at {path}")
        with open(path) as f:
            return json.load(f)

    # ── bulk build ───────────────────────────────────────────────────────

    def build(
        self,
        teacher,
        images: Iterable[Tuple[str, torch.Tensor]],
        skip_existing: bool = True,
        write_metadata: bool = True,
        log_every: int = 1000,
    ) -> Dict[str, int]:
        """Populate the cache from an (image_id, image) stream.

        Idempotent: with ``skip_existing=True`` already-cached images are
        skipped, so a re-run after an interruption resumes cleanly.

        Args:
            teacher: object exposing ``extract_features(images) -> {level: tensor}``
                (e.g. a CocoTeacher). Accepts a ``[B, 3, H, W]`` batch.
            images: iterable of ``(image_id, image_chw)`` — image_chw is a
                single ``[3, H, W]`` tensor.
            skip_existing: skip image_ids already in the cache.
            write_metadata: write metadata.json after the build.
            log_every: progress log cadence (scanned images).

        Returns:
            ``{"scanned", "skipped", "computed", "errors"}`` counters.
        """
        stats = {"scanned": 0, "skipped": 0, "computed": 0, "errors": 0}
        for image_id, image in images:
            stats["scanned"] += 1
            if skip_existing and self.has(image_id):
                stats["skipped"] += 1
                continue
            try:
                feats = teacher.extract_features(image.unsqueeze(0))
                # Drop the batch dim — cache stores per-image maps.
                feats = {lv: t[0] for lv, t in feats.items()}
                self.save(image_id, feats)
                stats["computed"] += 1
            except Exception as exc:  # noqa: BLE001 — IO / model failures vary
                LOG.warning("Teacher cache failed on %s: %s", image_id, exc)
                stats["errors"] += 1

            if log_every and stats["scanned"] % log_every == 0:
                LOG.info("Teacher cache progress: %s", stats)

        if write_metadata:
            self.save_metadata(extra={"n_cached": len(self)})
        LOG.info("Teacher cache build done: %s", stats)
        return stats

    # ── repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TeacherCache(dir={self.cache_dir}, levels={self.levels}, "
            f"n_cached={len(self)})"
        )
