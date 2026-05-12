"""Mapillary Vistas adapter.

The Vistas zip bundles both v1.2 and v2.0 annotation sets alongside the RGB
images. Top-level layout:

    training/images/<id>.jpg          <- canonical RGB (18,000)
    training/v1.2/labels/<id>.png     <- annotation (skipped)
    training/v1.2/panoptic/<id>.png   <- annotation (skipped)
    training/v1.2/instances/<id>.png  <- annotation (skipped)
    training/v2.0/...                 <- annotation (skipped)
    validation/images/<id>.jpg        <- canonical RGB (2,000)
    validation/v1.2/..., v2.0/...     <- annotation (skipped)
    testing/images/<id>.jpg           <- canonical RGB (5,000)
    config_v1.2.json, config_v2.0.json <- skipped

We materialize only the ~25K RGB images. The filter requires an exact
3-segment ``<split>/images/<file>.jpg`` path, which cleanly rejects every
deeper annotation entry without needing per-version logic.

Note that "testing" images have no annotations in this release — they're
still included in the pool because for SSL we don't care about labels, and
more unlabeled driving images is always better.
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
DATASET_NAME = "mapillary"

#: Top-level splits in the Vistas zip. All three contribute RGB images.
SPLITS = ("training", "validation", "testing")

#: Subdirectory under each split that contains RGB images. Sibling
#: subdirectories (v1.2, v2.0) hold annotation masks and are filtered out.
IMAGE_SUBDIR = "images"

#: How many ``ManifestRow`` instances to buffer before flushing to parquet.
DEFAULT_FLUSH_EVERY = 500


def _is_canonical_image(name: str) -> bool:
    """True iff ``name`` is an RGB image under a known split's ``images/`` dir.

    Expected path shape (exactly 3 segments):
        <split>/images/<file>.jpg

    Rejects: annotation PNGs under ``v1.2``/``v2.0``, config JSONs, and any
    nesting deeper than 3 segments.
    """
    parts = name.split("/")
    if len(parts) != 3:
        return False
    if parts[0] not in SPLITS:
        return False
    if parts[1] != IMAGE_SUBDIR:
        return False
    if not parts[2].lower().endswith(".jpg"):
        return False
    return True


def _parse_entry(name: str) -> Tuple[str, str]:
    """Split a canonical entry into ``(split, basename)``."""
    parts = name.split("/")
    return parts[0], parts[2]


def count_canonical_images(zip_path: Path) -> int:
    """Count canonical RGB image entries inside the archive."""
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
    """Stream-materialize Mapillary Vistas RGB images into the SSL pool.

    Args & returns mirror :func:`yolo_contrastive.data.ssl_pool.bdd100k.ingest`.
    """
    zip_path = Path(zip_path)
    pool_root = Path(pool_root)
    manifest_path = Path(manifest_path)

    already = existing_image_ids(manifest_path)
    LOG.info("Mapillary ingest start: %d image_ids already in manifest", len(already))

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
            split, basename = _parse_entry(info.filename)
            stem = basename.rsplit(".", 1)[0]
            image_id = f"{DATASET_NAME}/{split}/{stem}"

            if image_id in already:
                stats["skipped_existing"] += 1
                continue

            materialized_rel = f"images/{DATASET_NAME}/{split}/{stem}.jpg"
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
                LOG.info("Mapillary progress: %s", stats)

            if limit is not None and stats["materialized"] >= limit:
                LOG.info("Mapillary ingest hit limit=%d, stopping", limit)
                break

    flush()
    LOG.info("Mapillary ingest done: %s", stats)
    return stats
