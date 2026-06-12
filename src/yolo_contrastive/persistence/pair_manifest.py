"""Parquet manifests for the REVISIT cross-traversal pipeline.

Three flat tables (no nested types — pandas + pyarrow round-trip cleanly on
Colab and local Windows), mirroring ``data/ssl_pool/manifest.py``:

``pairs.parquet``
    One row per candidate co-located image pair. Mining writes rows with
    ``status="queued"``; the downloader flips them to ``"downloaded"``; the
    aligner fills the normalized homography (``h00..h21``, with ``h22 := 1``)
    plus trust stats and sets ``status`` to ``"aligned"`` or ``"rejected"``;
    the labeler fills ``n_persistent`` / ``n_transient``.

``proposals.parquet``
    Class-agnostic blob proposals per unique image (normalized xyxy boxes).

``persistence_labels.parquet``
    Cross-traversal persistence labels per proposal that survived matching
    (``label`` in {"persistent", "transient"}).

All appends deduplicate on a key column, so every offline stage is idempotent
and resumable (re-running a stage on a partially-completed pool yields zero
new rows when nothing changed).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Set

import pandas as pd

# ── schemas (canonical column order) ──────────────────────────────────────────

#: pairs.parquet columns. ``h00..h21`` are the first 8 entries of the
#: row-major NORMALIZED homography (x_b = H_norm @ x_a in [0,1]^2, h22 := 1).
PAIRS_COLUMNS = [
    "pair_id",
    "img_a_id", "img_b_id",
    "path_a", "path_b",
    "lon_a", "lat_a", "lon_b", "lat_b",
    "dist_m",
    "heading_a", "heading_b", "heading_diff",
    "captured_at_a", "captured_at_b", "dt_days",
    "seq_a", "seq_b",
    "city", "tile_id",
    "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21",
    "n_inliers", "inlier_ratio", "reproj_rmse", "overlap_frac",
    "align_method", "align_ok",
    "n_persistent", "n_transient",
    "status",
]

PROPOSALS_COLUMNS = [
    "image_id", "prop_id", "x1", "y1", "x2", "y2", "score", "backend",
]

LABELS_COLUMNS = [
    "label_id", "pair_id", "image_id", "side", "x1", "y1", "x2", "y2", "label",
]

#: Defaults for pair columns that later stages fill in.
_PAIR_DEFAULTS: Dict[str, Any] = {
    "path_a": "", "path_b": "",
    "h00": math.nan, "h01": math.nan, "h02": math.nan,
    "h10": math.nan, "h11": math.nan, "h12": math.nan,
    "h20": math.nan, "h21": math.nan,
    "n_inliers": 0, "inlier_ratio": math.nan, "reproj_rmse": math.nan,
    "overlap_frac": math.nan,
    "align_method": "", "align_ok": False,
    "n_persistent": 0, "n_transient": 0,
    "status": "queued",
}

#: Valid pair lifecycle states (mining -> download -> align -> labeled).
PAIR_STATUSES = ("queued", "downloaded", "aligned", "rejected")


def new_pair_row(**kwargs: Any) -> Dict[str, Any]:
    """Build a full pairs-row dict from the mining-stage fields + defaults.

    Raises on unknown keys so schema drift is caught at the call site.
    """
    unknown = set(kwargs) - set(PAIRS_COLUMNS)
    if unknown:
        raise ValueError(f"unknown pair columns: {sorted(unknown)}")
    row = dict(_PAIR_DEFAULTS)
    row.update(kwargs)
    missing = set(PAIRS_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"pair row missing required columns: {sorted(missing)}")
    return row


# ── generic table IO ──────────────────────────────────────────────────────────


def read_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read a manifest parquet; empty DataFrame with full schema if absent."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    return pd.read_parquet(path)


def write_table(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    """Overwrite ``path`` with ``df`` (schema-checked, extra columns dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    df[list(columns)].to_parquet(path, index=False)


def append_rows(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    columns: Sequence[str],
    key: str,
) -> int:
    """Append dict rows to the table at ``path``, deduplicating on ``key``.

    Returns the number of NEW rows written (idempotent re-runs return 0).
    """
    rows = list(rows)
    if not rows:
        return 0
    incoming = pd.DataFrame(rows)
    # also dedup within the incoming batch itself
    incoming = incoming.drop_duplicates(subset=[key], keep="first")
    existing = read_table(path, columns)
    if not existing.empty:
        incoming = incoming[~incoming[key].isin(set(existing[key]))]
    if incoming.empty:
        return 0
    combined = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty else incoming
    )
    write_table(combined, path, columns)
    return len(incoming)


def existing_keys(path: Path, key: str, columns: Sequence[str]) -> Set[str]:
    """Set of ``key`` values currently in the table (for resume logic)."""
    df = read_table(path, columns)
    if df.empty:
        return set()
    return set(df[key])


# ── pairs.parquet ─────────────────────────────────────────────────────────────


def read_pairs(path: Path) -> pd.DataFrame:
    return read_table(path, PAIRS_COLUMNS)


def write_pairs(df: pd.DataFrame, path: Path) -> None:
    write_table(df, path, PAIRS_COLUMNS)


def append_pairs(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append pair rows (dicts from :func:`new_pair_row`); dedup on pair_id."""
    return append_rows(path, rows, PAIRS_COLUMNS, key="pair_id")


def update_pairs(path: Path, updates: Dict[str, Dict[str, Any]]) -> int:
    """Apply per-pair column updates: ``{pair_id: {column: value, ...}}``.

    Unknown pair_ids are ignored (returns the number of rows touched).
    Status values are validated against :data:`PAIR_STATUSES`.
    """
    df = read_pairs(path)
    if df.empty or not updates:
        return 0
    touched = 0
    index_of = {pid: i for i, pid in enumerate(df["pair_id"])}
    for pid, cols in updates.items():
        i = index_of.get(pid)
        if i is None:
            continue
        for col, val in cols.items():
            if col not in PAIRS_COLUMNS:
                raise ValueError(f"unknown pair column {col!r}")
            if col == "status" and val not in PAIR_STATUSES:
                raise ValueError(f"invalid status {val!r}; expected one of {PAIR_STATUSES}")
            df.loc[df.index[i], col] = val
        touched += 1
    if touched:
        write_pairs(df, path)
    return touched


# ── proposals.parquet ─────────────────────────────────────────────────────────


def read_proposals(path: Path) -> pd.DataFrame:
    return read_table(path, PROPOSALS_COLUMNS)


def append_proposals(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append proposal rows; dedup on prop_id (``{image_id}_{i:03d}``)."""
    return append_rows(path, rows, PROPOSALS_COLUMNS, key="prop_id")


def proposed_image_ids(path: Path) -> Set[str]:
    """Image ids that already have proposals (resume logic)."""
    return existing_keys(path, "image_id", PROPOSALS_COLUMNS)


# ── persistence_labels.parquet ────────────────────────────────────────────────


def read_labels(path: Path) -> pd.DataFrame:
    return read_table(path, LABELS_COLUMNS)


def append_labels(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append label rows; dedup on label_id (``{pair_id}_{side}_{i:03d}``)."""
    return append_rows(path, rows, LABELS_COLUMNS, key="label_id")


def labeled_pair_ids(path: Path) -> Set[str]:
    """Pair ids that already have at least one label row (resume logic)."""
    return existing_keys(path, "pair_id", LABELS_COLUMNS)
