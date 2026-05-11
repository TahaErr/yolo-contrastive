"""SSL pool manifest: parquet-backed registry of materialized images.

The manifest is the authoritative index of every image in the SSL pool. It
records both the materialized form (what we actually train on after resize)
and traceability fields back to the original dataset entry. Per-dataset
adapters call ``append_rows`` after materializing a batch; the orchestrator
uses ``existing_image_ids`` for resume logic.

Schema is intentionally flat (no nested types) so that pandas + pyarrow round
trips cleanly on every platform we expect to run on (Colab, local).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, Set

import pandas as pd

#: Canonical column order. Used both for empty-DataFrame construction and as a
#: schema check on writes.
MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "original_split",
    "materialized_path",
    "original_h",
    "original_w",
    "materialized_h",
    "materialized_w",
    "image_hash",
    "original_filename",
]


@dataclasses.dataclass(frozen=True)
class ManifestRow:
    """One row of the manifest. Field names match ``MANIFEST_COLUMNS``."""

    image_id: str
    dataset: str
    original_split: str
    materialized_path: str
    original_h: int
    original_w: int
    materialized_h: int
    materialized_w: int
    image_hash: str
    original_filename: str


def read_manifest(path: Path) -> pd.DataFrame:
    """Read manifest parquet. Returns empty DataFrame with full schema if the
    file does not yet exist (this is the normal case before the first append).
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_parquet(path)


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Overwrite the manifest at ``path`` with ``df``.

    Validates that all canonical columns are present. Extra columns are
    silently dropped to keep the on-disk schema stable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    df[MANIFEST_COLUMNS].to_parquet(path, index=False)


def append_rows(path: Path, rows: Iterable[ManifestRow]) -> int:
    """Append ``rows`` to the manifest, deduplicating on ``image_id``.

    Returns the number of NEW rows actually written (i.e. those whose
    ``image_id`` was not already present). This makes the operation
    idempotent: re-running an adapter on a partially-completed pool yields
    zero new rows when nothing changed.
    """
    rows = list(rows)
    if not rows:
        return 0
    incoming = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    existing = read_manifest(path)
    if not existing.empty:
        already = set(existing["image_id"])
        incoming = incoming[~incoming["image_id"].isin(already)]
    if incoming.empty:
        return 0
    combined = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty
        else incoming
    )
    write_manifest(combined, path)
    return len(incoming)


def existing_image_ids(path: Path) -> Set[str]:
    """Return the set of ``image_id`` values currently in the manifest.

    Adapters use this for resume logic: skip images whose id is already
    registered. Reads the entire manifest once per call; callers doing many
    lookups should cache the returned set.
    """
    df = read_manifest(path)
    if df.empty:
        return set()
    return set(df["image_id"])
