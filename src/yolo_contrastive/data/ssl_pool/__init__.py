"""SSL pretraining image pool: ingestion, materialization, and indexing.

Public API:

- ``ManifestRow``, ``append_rows``, ``read_manifest``, ``existing_image_ids``
  — the parquet-backed registry of materialized images.
- ``resize_and_save``, ``is_readable_image``, ``download_with_resume``
  — adapter-shared helpers.

Per-dataset adapters (``a2d2.py``, ``cityscapes.py``, ``bdd100k.py``,
``mapillary.py``) and the orchestrator (``build_pool.py``) are added in
subsequent phases.
"""

from .common import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LONG_SIDE,
    download_with_resume,
    is_readable_image,
    resize_and_save,
)
from .manifest import (
    MANIFEST_COLUMNS,
    ManifestRow,
    append_rows,
    existing_image_ids,
    read_manifest,
    write_manifest,
)

__all__ = [
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_LONG_SIDE",
    "MANIFEST_COLUMNS",
    "ManifestRow",
    "append_rows",
    "download_with_resume",
    "existing_image_ids",
    "is_readable_image",
    "read_manifest",
    "resize_and_save",
    "write_manifest",
]
