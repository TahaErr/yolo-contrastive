"""Cityscapes disparity → inverse-depth GT (B1 depth training).

The dense, deterministic, no-GPU real-sensor GT source. Cityscapes ships precomputed SGM
disparity next to every left image:

    disparity/<split>/<city>/<id>_disparity.png   <- 16-bit; disp = (p - 1) / 256 for p > 0

Decoded disparity is proportional to inverse depth (disp = baseline*fx / Z), so for the
**affine-invariant** road-plane consumer (`plane_fit` fits in an arbitrary affine of inverse
depth) we cache ``inv = disp`` directly — no camera intrinsics/baseline needed. The
``p == 0`` sentinel (sky / no-match) becomes ``valid = False`` and is masked, never averaged.

Keyed by the SAME ``image_id`` as the RGB pool (``cityscapes/<split>/<city>/<id>_leftImg8bit``)
so (pool RGB, GT depth) pairs line up. GT is masked-downsampled to the pool's long-side so it
matches the materialized RGB resolution. Output goes to a :class:`~yolo_contrastive.geoteach.depth_gt.DepthGT`.

cv2 is imported lazily (E2).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from ...geoteach.depth_gt import DepthGT, masked_downsample

LOG = logging.getLogger(__name__)

CANONICAL_PREFIX = "disparity/"
KNOWN_SPLITS = ("train", "val", "test", "train_extra")
DEFAULT_LONG_SIDE = 640
#: minimum disparity (px) to trust — below this, depth is far and SGM-noisy.
MIN_DISPARITY = 0.1


def _is_canonical_disparity(name: str) -> bool:
    """True iff ``name`` is ``disparity/<split>/<city>/<id>_disparity.png``."""
    if not name.startswith(CANONICAL_PREFIX) or not name.lower().endswith("_disparity.png"):
        return False
    parts = name[len(CANONICAL_PREFIX):].split("/")
    return len(parts) == 3 and parts[0] in KNOWN_SPLITS


def _image_id(split: str, city: str, disp_stem: str) -> str:
    """Map a disparity stem ``<id>_disparity`` to the RGB pool ``image_id``."""
    frame_id = disp_stem[: -len("_disparity")] if disp_stem.endswith("_disparity") else disp_stem
    return f"cityscapes/{split}/{city}/{frame_id}_leftImg8bit"


def disparity_to_inverse(png_u16: np.ndarray) -> tuple:
    """16-bit Cityscapes disparity PNG → (inv_depth float32, valid bool).

    ``disp = (p - 1) / 256`` for ``p > 0``; ``inv = disp`` (∝ 1/Z, affine-invariant).
    """
    p = np.asarray(png_u16).astype(np.float32)
    disp = (p - 1.0) / 256.0
    valid = (p > 0) & (disp > MIN_DISPARITY)
    inv = np.where(valid, disp, 0.0).astype(np.float32)
    return inv, valid


def _target_hw(h: int, w: int, long_side: int) -> tuple:
    longest = max(h, w)
    if longest <= long_side:
        return h, w
    s = long_side / longest
    return max(1, round(h * s)), max(1, round(w * s))


def ingest_disparity(
    disp_zip: Path,
    gt_store: DepthGT,
    long_side: int = DEFAULT_LONG_SIDE,
    limit: Optional[int] = None,
    resume: bool = True,
    log_every: int = 1000,
) -> dict:
    """Stream a Cityscapes disparity zip → inverse-depth GT into ``gt_store``.

    Works on ``disparity_trainvaltest.zip`` and ``disparity_trainextra.zip``. Idempotent on
    ``image_id`` (skips already-stored). Returns stats.
    """
    import cv2  # lazy
    disp_zip = Path(disp_zip)
    stats = {"scanned": 0, "skipped_existing": 0, "materialized": 0,
             "empty_valid": 0, "errors": 0}

    with zipfile.ZipFile(disp_zip) as z:
        for info in z.infolist():
            if info.is_dir() or not _is_canonical_disparity(info.filename):
                continue
            stats["scanned"] += 1
            split, city, basename = info.filename[len(CANONICAL_PREFIX):].split("/")
            stem = basename.rsplit(".", 1)[0]
            image_id = _image_id(split, city, stem)

            if resume and gt_store.has(image_id):
                stats["skipped_existing"] += 1
                continue
            try:
                raw = np.frombuffer(z.read(info), dtype=np.uint8)
                png = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
                if png is None or png.dtype != np.uint16:
                    raise ValueError(f"not a uint16 PNG (dtype={None if png is None else png.dtype})")
                inv, valid = disparity_to_inverse(png)
                oh, ow = _target_hw(png.shape[0], png.shape[1], long_side)
                inv_ds, valid_ds = masked_downsample(inv, valid, (oh, ow))
                if not valid_ds.any():
                    stats["empty_valid"] += 1
                    continue
                gt_store.save(image_id, inv_ds, valid_ds,
                              meta={"source": "cityscapes_disparity", "split": split,
                                    "kind": "relative_inverse_depth"})
                stats["materialized"] += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("disparity skip %s: %s", info.filename, exc)
                stats["errors"] += 1
                continue

            if log_every and stats["scanned"] % log_every == 0:
                LOG.info("cityscapes disparity: %s", stats)
            if limit is not None and stats["materialized"] >= limit:
                break

    LOG.info("cityscapes disparity done: %s", stats)
    return stats
