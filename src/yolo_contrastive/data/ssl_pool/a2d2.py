"""A2D2 adapter: stream front-center camera images from the A2D2 semantic tar.

A2D2 ships its semantic-segmentation release as a single ~160 GB tarball
containing six camera angles per scene plus lidar, label masks, and JSON
metadata. Per the plan we materialize only the ``cam_front_center`` stream:
the other five cameras are redundant views of the same scenes and would
inflate A2D2's share of the pool, skewing it toward Germany driving.

Tar layout (only the relevant branch is materialized):

    camera_lidar_semantic/<scene>/camera/cam_front_center/<ts>_camera_frontcenter_<id>.png
    camera_lidar_semantic/<scene>/camera/cam_front_center/<ts>_camera_frontcenter_<id>.json  <- skipped
    camera_lidar_semantic/<scene>/camera/cam_<other>/...                                       <- skipped
    camera_lidar_semantic/<scene>/label/...                                                    <- skipped
    camera_lidar_semantic/<scene>/lidar/...                                                    <- skipped

Streaming (``tarfile`` mode ``"r|"``) is used because the source typically
lives on Drive: sequential bytewise reads are fast, random-access seeks are
not. Non-target entries are header-skipped — only the ~41K target PNGs are
actually read from the archive, so ingest cost scales with the materialized
count, not the 164 GB archive size.

A2D2 has no inherent train/val/test split — for SSL it's an unlabeled image
source. All rows are tagged ``original_split="unlabeled"``; scene identity is
preserved in ``image_id`` and the materialized path so per-scene analysis
remains possible from the manifest alone.
"""

from __future__ import annotations

import io
import logging
import tarfile
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
DATASET_NAME = "a2d2"

#: Top-level directory inside the A2D2 semantic-segmentation tar.
TAR_ROOT = "camera_lidar_semantic"

#: The single camera position we materialize. The other five (``cam_front_left``,
#: ``cam_front_right``, ``cam_rear_center``, ``cam_side_left``, ``cam_side_right``)
#: are sibling directories inside the tar; we ignore them.
TARGET_CAMERA = "cam_front_center"

#: A2D2 has no train/val/test split — for SSL it's unlabeled images.
ORIGINAL_SPLIT = "unlabeled"

#: How many ``ManifestRow`` instances to buffer before flushing to parquet.
DEFAULT_FLUSH_EVERY = 500


def _is_canonical_image(name: str) -> bool:
    """True iff ``name`` is a target-camera PNG inside the tar.

    Expected path shape (exactly 5 segments):
        camera_lidar_semantic/<scene>/camera/cam_front_center/<file>.png

    Rejects: the ``label``/``lidar`` branches, sibling cameras, ``.json``
    companion metadata, and any nested oddities.
    """
    parts = name.split("/")
    if len(parts) != 5:
        return False
    if parts[0] != TAR_ROOT:
        return False
    if parts[2] != "camera":
        return False
    if parts[3] != TARGET_CAMERA:
        return False
    if not parts[4].lower().endswith(".png"):
        return False
    return True


def _parse_entry(name: str) -> Tuple[str, str]:
    """Split a canonical tar entry into ``(scene, basename_with_ext)``."""
    parts = name.split("/")
    return parts[1], parts[4]  # <scene>, <file>.png


def count_canonical_images(tar_path: Path) -> int:
    """Count canonical front-center PNG entries inside the tar.

    Note: with a ~164 GB tar on Drive this is a full sequential scan, so
    expect tens of minutes. Useful as a one-time pre-ingest sanity check
    against the plan's ~41K target; not something to run casually.
    """
    n = 0
    with tarfile.open(tar_path, "r|") as t:
        for m in t:
            if m.isfile() and _is_canonical_image(m.name):
                n += 1
    return n


def ingest(
    tar_path: Path,
    pool_root: Path,
    manifest_path: Path,
    long_side: int = DEFAULT_LONG_SIDE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    limit: Optional[int] = None,
    log_every: int = 1000,
) -> dict:
    """Stream-materialize A2D2 cam_front_center images into the SSL pool.

    Side effects:
      - writes JPEG files under ``pool_root / "images" / "a2d2" / <scene>``
      - appends rows to the parquet manifest

    Idempotent on ``image_id`` — a re-run skips images already in the
    manifest. Note however that the tar is iterated from the start every
    time (the tar format has no random index), so a re-run still pays the
    full sequential-scan cost. Within a single run, a Colab disconnect
    means losing only the in-flight ``flush_every`` rows since the last
    parquet write.

    Args & returns mirror :func:`yolo_contrastive.data.ssl_pool.bdd100k.ingest`.
    """
    tar_path = Path(tar_path)
    pool_root = Path(pool_root)
    manifest_path = Path(manifest_path)

    already = existing_image_ids(manifest_path)
    LOG.info("A2D2 ingest start: %d image_ids already in manifest", len(already))

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

    # Stream mode "r|": one-pass forward iteration, no random-access seeks.
    # This is the right mode for a tar sitting on Drive — seeks would be
    # ruinously expensive there.
    with tarfile.open(tar_path, "r|") as t:
        for member in t:
            if not member.isfile() or not _is_canonical_image(member.name):
                continue

            stats["scanned"] += 1
            scene, basename = _parse_entry(member.name)
            stem = basename.rsplit(".", 1)[0]
            image_id = f"{DATASET_NAME}/{scene}/{stem}"

            if image_id in already:
                stats["skipped_existing"] += 1
                continue

            materialized_rel = f"images/{DATASET_NAME}/{scene}/{stem}.jpg"
            dest = pool_root / materialized_rel

            try:
                fp = t.extractfile(member)
                if fp is None:
                    # Defensive — shouldn't happen for ``isfile()`` members.
                    LOG.warning("extractfile returned None for %s", member.name)
                    stats["errors"] += 1
                    continue
                raw = fp.read()
                orig_size, mat_size, sha = resize_and_save(
                    io.BytesIO(raw),
                    dest,
                    long_side=long_side,
                    jpeg_quality=jpeg_quality,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping %s: %s", member.name, exc)
                stats["errors"] += 1
                continue

            buffer.append(
                ManifestRow(
                    image_id=image_id,
                    dataset=DATASET_NAME,
                    original_split=ORIGINAL_SPLIT,
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
                LOG.info("A2D2 progress: %s", stats)

            if limit is not None and stats["materialized"] >= limit:
                LOG.info("A2D2 ingest hit limit=%d, stopping", limit)
                break

    flush()
    LOG.info("A2D2 ingest done: %s", stats)
    return stats
