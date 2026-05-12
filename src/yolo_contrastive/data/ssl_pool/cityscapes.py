"""Cityscapes adapter.

Supports both Cityscapes packages — the coarse ``leftImg8bit_trainextra.zip``
and the fine ``leftImg8bit_trainvaltest.zip`` — with a single ``ingest``
function. They share the same internal layout; only the split names differ:

    leftImg8bit/<split>/<city>/<id>_leftImg8bit.png   <- canonical RGB image
    README, license.txt                               <- skipped

Splits encountered:
    coarse package:  train_extra  (19,998 images, 1 known corrupt)
    fine package:    train / val / test  (2,975 / 500 / 1,525 images)

We materialize all of them — for SSL the split distinction is just
provenance, not a partitioning constraint. The original split name is
preserved in the manifest so ablation analysis remains possible. Per-city
substructure is also preserved in the materialized path and image_id to
guarantee filename uniqueness and to keep dataset organization legible
inside the pool.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from .common import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LONG_SIDE,
    resize_and_save,
)
from .manifest import (
    ManifestRow,
    append_rows,
    existing_image_ids,
)

LOG = logging.getLogger(__name__)

#: Dataset name written into the manifest's ``dataset`` column.
DATASET_NAME = "cityscapes"

#: Path prefix inside both Cityscapes zips under which RGB images live.
CANONICAL_PREFIX = "leftImg8bit/"

#: All Cityscapes split names we accept across coarse and fine packages.
KNOWN_SPLITS = ("train", "val", "test", "train_extra")

#: How many ``ManifestRow`` instances to buffer before flushing to parquet.
DEFAULT_FLUSH_EVERY = 500


def _is_canonical_image(name: str) -> bool:
    """True iff ``name`` is a ``leftImg8bit`` PNG under a known split.

    Expected path shape (exactly 3 segments after the prefix):
        leftImg8bit/<split>/<city>/<file>.png
    """
    if not name.startswith(CANONICAL_PREFIX):
        return False
    if not name.lower().endswith(".png"):
        return False
    after = name[len(CANONICAL_PREFIX):]
    parts = after.split("/")
    return len(parts) == 3 and parts[0] in KNOWN_SPLITS


def _parse_entry(name: str) -> Tuple[str, str, str]:
    """Split a canonical zip entry into ``(split, city, basename)``."""
    after = name[len(CANONICAL_PREFIX):]
    parts = after.split("/")
    return parts[0], parts[1], parts[2]


def count_canonical_images(zip_path: Path) -> int:
    """Count canonical ``leftImg8bit`` PNG entries inside the archive."""
    with zipfile.ZipFile(zip_path) as z:
        return sum(1 for n in z.namelist() if _is_canonical_image(n))


def ingest(
    zip_path: Path,
    pool_root: Path,
    manifest_path: Path,
    long_side: int = DEFAULT_LONG_SIDE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    limit: Optional[int] = None,
    log_every: int = 1000,
) -> dict:
    """Stream-materialize Cityscapes left-camera images into the SSL pool.

    Works on either the coarse ``trainextra`` or the fine ``trainvaltest``
    package — both share the same internal layout. Output is partitioned by
    split and city, mirroring the source. Idempotent on ``image_id`` so
    running both zips back-to-back into the same manifest is safe and
    crash-resume works.

    Args & returns mirror :func:`yolo_contrastive.data.ssl_pool.bdd100k.ingest`.
    """
    zip_path = Path(zip_path)
    pool_root = Path(pool_root)
    manifest_path = Path(manifest_path)

    already = existing_image_ids(manifest_path)
    LOG.info("Cityscapes ingest start: %d image_ids already in manifest", len(already))

    stats = {
        "scanned": 0,
        "skipped_existing": 0,
        "materialized": 0,
        "errors": 0,
    }
    buffer: list[ManifestRow] = []

    def flush() -> None:
        if not buffer:
            return
        n = append_rows(manifest_path, buffer)
        LOG.info("Flushed %d rows to manifest (+%d new)", len(buffer), n)
        buffer.clear()

    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not _is_canonical_image(info.filename):
                continue

            stats["scanned"] += 1
            split, city, basename = _parse_entry(info.filename)
            stem = basename.rsplit(".", 1)[0]
            image_id = f"{DATASET_NAME}/{split}/{city}/{stem}"

            if image_id in already:
                stats["skipped_existing"] += 1
                continue

            materialized_rel = f"images/{DATASET_NAME}/{split}/{city}/{stem}.jpg"
            dest = pool_root / materialized_rel

            try:
                raw = z.read(info)
                orig_size, mat_size, sha = resize_and_save(
                    io.BytesIO(raw),
                    dest,
                    long_side=long_side,
                    jpeg_quality=jpeg_quality,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping %s: %s", info.filename, exc)
                stats["errors"] += 1
                continue

            buffer.append(
                ManifestRow(
                    image_id=image_id,
                    dataset=DATASET_NAME,
                    original_split=split,
                    materialized_path=materialized_rel,
                    original_h=orig_size[1],
                    original_w=orig_size[0],
                    materialized_h=mat_size[1],
                    materialized_w=mat_size[0],
                    image_hash=sha,
                    original_filename=basename,
                )
            )
            stats["materialized"] += 1

            if len(buffer) >= flush_every:
                flush()

            if log_every and stats["scanned"] % log_every == 0:
                LOG.info("Cityscapes progress: %s", stats)

            if limit is not None and stats["materialized"] >= limit:
                LOG.info("Cityscapes ingest hit limit=%d, stopping", limit)
                break

    flush()
    LOG.info("Cityscapes ingest done: %s", stats)
    return stats
