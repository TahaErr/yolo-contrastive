"""BDD100K adapter: stream images from the BDD100K archive into the SSL pool.

The user's archive is the "full bundle" zip with three top-level trees:

    bdd100k/bdd100k/images/100k/{train,val,test}/<id>.jpg   <- canonical 100K
    bdd100k_seg/...                                          <- skipped
    bdd100k_labels_release/...                               <- skipped

We restrict to ``CANONICAL_IMAGE_PREFIX`` so the pool gets exactly the
documented 100K images (70K train + 10K val + 20K test) regardless of what
extras live in the bundle. The ``bdd100k_seg/`` tree overlaps with the 100K
and is dropped to keep image provenance unambiguous.

Images are read from the zip into memory and resized in a single pass — no
full extraction. With ~720p JPEG sources (~300 KB each) this is much cheaper
than extract-then-process and keeps Colab local disk pressure low.
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
DATASET_NAME = "bdd100k"

#: Path prefix inside the zip under which canonical 100K JPEGs live. The
#: double ``bdd100k/bdd100k/`` is a real artifact of the bundle layout — not a
#: typo.
CANONICAL_IMAGE_PREFIX = "bdd100k/bdd100k/images/100k/"

#: Splits the BDD100K release uses inside the 100k tree.
SPLITS = ("train", "val", "test")

#: How many ``ManifestRow`` instances to buffer before flushing to parquet.
#: Each flush rewrites the whole parquet file, so we batch.
DEFAULT_FLUSH_EVERY = 500


def _is_canonical_image(name: str) -> bool:
    """True iff ``name`` is a JPEG entry somewhere under ``<prefix>/<split>/``.

    BDD100K bundles are inconsistent in their internal layout: some entries
    sit flat under ``<split>/<file>.jpg``, others are grouped into sub-buckets
    like ``<split>/<subdir>/<file>.jpg`` (e.g. ``test/testA/``, ``test/testB/``).
    We accept both. The split must still be one of the known three.

    Filters out: the ``bdd100k_seg`` and ``bdd100k_labels_release`` trees,
    label JSONs, and any unknown split names.
    """
    if not name.startswith(CANONICAL_IMAGE_PREFIX):
        return False
    if not name.lower().endswith(".jpg"):
        return False
    after = name[len(CANONICAL_IMAGE_PREFIX):]
    parts = after.split("/")
    # At least "<split>/<basename>" — subdirs under the split are allowed.
    return len(parts) >= 2 and parts[0] in SPLITS


def _parse_entry(name: str) -> Tuple[str, str]:
    """Split a canonical zip entry into ``(split, basename)``.

    ``basename`` is the relative path *under* the split — for a flat entry
    this is just the filename (``"foo.jpg"``); for a nested entry it includes
    the subdir (``"testA/foo.jpg"``). Callers preserve that nesting in both
    ``image_id`` and the materialized path so flat- and nested-named images
    in different sub-buckets cannot collide.
    """
    after = name[len(CANONICAL_IMAGE_PREFIX):]
    split, basename = after.split("/", 1)
    return split, basename


def count_canonical_images(zip_path: Path) -> int:
    """Count canonical 100K image entries inside the archive.

    Useful as a pre-ingest sanity check (expect ~100,000 from the official
    release) and as a post-ingest reconciliation against the manifest.
    """
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
    """Stream-materialize BDD100K canonical 100K images into the SSL pool.

    Side effects:
      - writes JPEG files under ``pool_root / "images" / "bdd100k" / <split>``
      - appends rows to the parquet manifest at ``manifest_path``

    Idempotent: ``image_id`` values already present in the manifest are
    skipped, so a re-run after a Colab disconnect resumes cleanly.

    Args:
        zip_path: path to the user's BDD100K bundle zip in Drive.
        pool_root: SSL pool root (e.g. ``.../ssl_pool/``).
        manifest_path: parquet manifest path.
        long_side: target long-side resolution for the JPEG copy.
        jpeg_quality: JPEG quality for the materialized copy.
        flush_every: how many rows to buffer before writing to parquet.
        limit: optional cap on images materialized this run. Useful for a
            smoke test before committing to the full 100K. Must be ``>= 1``.
        log_every: emit a progress log every N scanned entries.

    Returns:
        dict of counters: ``scanned``, ``skipped_existing``, ``materialized``,
        ``errors``.
    """
    zip_path = Path(zip_path)
    pool_root = Path(pool_root)
    manifest_path = Path(manifest_path)

    already = existing_image_ids(manifest_path)
    LOG.info("BDD100K ingest start: %d image_ids already in manifest", len(already))

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
            # ``basename`` may contain a subdir (e.g. "testA/foo.jpg"). Keep
            # that nesting in both the id and the materialized path so flat
            # and nested entries with the same filename can never collide.
            rel_stem = basename.rsplit(".", 1)[0]
            image_id = f"{DATASET_NAME}/{split}/{rel_stem}"

            if image_id in already:
                stats["skipped_existing"] += 1
                continue

            materialized_rel = f"images/{DATASET_NAME}/{split}/{rel_stem}.jpg"
            dest = pool_root / materialized_rel

            try:
                raw = z.read(info)
                orig_size, mat_size, sha = resize_and_save(
                    io.BytesIO(raw),
                    dest,
                    long_side=long_side,
                    jpeg_quality=jpeg_quality,
                )
            except Exception as exc:  # noqa: BLE001 — image errors take many shapes
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
                LOG.info("BDD100K progress: %s", stats)

            if limit is not None and stats["materialized"] >= limit:
                LOG.info("BDD100K ingest hit limit=%d, stopping", limit)
                break

    flush()
    LOG.info("BDD100K ingest done: %s", stats)
    return stats
