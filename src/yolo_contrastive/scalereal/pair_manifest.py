"""Pair manifest — parquet registry of mined GASP-Real scale pairs.

One row per PAIR (flat schema, pandas + pyarrow round-trip, mirroring
data/ssl_pool/manifest.py conventions). Boxes are normalized xyxy in [0, 1]
of the ORIGINAL materialized image; ``log_r = log(Z_A / Z_B)`` is the real
apparent-scale ratio label (antisymmetric under A<->B swap).

``on_road_a`` / ``on_road_b`` come from TERRA's plane-fit inlier mask when
present and are NaN otherwise (graceful null — flat float column).

Also home of:
    * :class:`PairIndex` — per-image indexed lookup used by the training
      dataset hook (``prepare_targets(image_id)``).
    * :func:`is_probe_image` — the deterministic 1% probe holdout (hash of
      image_id), reserved for sentinel evaluation and never sampled in
      training.

pandas/pyarrow are imported lazily at module level by the callers' standards:
this module imports pandas at import time like ssl_pool/manifest.py, but it is
itself only imported lazily from the scalereal package (E2).
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

#: Canonical column order — schema check on writes, empty-frame construction.
PAIR_COLUMNS = [
    "image_id",
    "pair_id",
    "box_a_x1", "box_a_y1", "box_a_x2", "box_a_y2",
    "box_b_x1", "box_b_y1", "box_b_x2", "box_b_y2",
    "log_r",
    "z_a", "z_b",
    "sim",
    "texture_a", "texture_b",
    "depth_iqr_a", "depth_iqr_b",
    "on_road_a", "on_road_b",
    "miner_version",
]

_BOX_COLS_A = ["box_a_x1", "box_a_y1", "box_a_x2", "box_a_y2"]
_BOX_COLS_B = ["box_b_x1", "box_b_y1", "box_b_x2", "box_b_y2"]


@dataclasses.dataclass(frozen=True)
class PairRecord:
    """One mined pair. Field names match ``PAIR_COLUMNS``."""

    image_id: str
    pair_id: str
    box_a_x1: float
    box_a_y1: float
    box_a_x2: float
    box_a_y2: float
    box_b_x1: float
    box_b_y1: float
    box_b_x2: float
    box_b_y2: float
    log_r: float
    z_a: float
    z_b: float
    sim: float
    texture_a: float
    texture_b: float
    depth_iqr_a: float
    depth_iqr_b: float
    on_road_a: float = float("nan")
    on_road_b: float = float("nan")
    miner_version: int = 1


def read_pairs(path: Path) -> pd.DataFrame:
    """Read the pair manifest; empty frame with full schema if absent."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.read_parquet(path)


def write_pairs(df: pd.DataFrame, path: Path) -> None:
    """Overwrite the pair manifest (schema-checked; extra columns dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(PAIR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Pair manifest missing columns: {sorted(missing)}")
    df[PAIR_COLUMNS].to_parquet(path, index=False)


def append_pairs(path: Path, rows: Iterable[PairRecord]) -> int:
    """Append rows, deduplicating on ``pair_id``; returns NEW rows written.

    Idempotent: re-running the miner on a partially-completed pool appends
    zero rows for already-mined pairs (resume by image_id set-difference).
    """
    rows = list(rows)
    if not rows:
        return 0
    incoming = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    existing = read_pairs(path)
    if not existing.empty:
        already = set(existing["pair_id"])
        incoming = incoming[~incoming["pair_id"].isin(already)]
    if incoming.empty:
        return 0
    combined = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty
        else incoming
    )
    write_pairs(combined, path)
    return len(incoming)


def existing_image_ids(path: Path) -> Set[str]:
    """image_ids already present — the miner's resume set."""
    df = read_pairs(path)
    if df.empty:
        return set()
    return set(df["image_id"])


def validate_pairs(df: pd.DataFrame) -> None:
    """Structural validation: columns, box ranges, finite labels.

    Raises ``ValueError`` on the first violation; cheap enough to run on
    every manifest load in the training dataset.
    """
    missing = set(PAIR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Pair manifest missing columns: {sorted(missing)}")
    if df.empty:
        return
    for cols in (_BOX_COLS_A, _BOX_COLS_B):
        box = df[cols].to_numpy(dtype=np.float64)
        if not np.isfinite(box).all():
            raise ValueError("pair boxes contain non-finite values")
        if (box < -1e-6).any() or (box > 1.0 + 1e-6).any():
            raise ValueError("pair boxes must be normalized xyxy in [0, 1]")
        if ((box[:, 2] - box[:, 0]) <= 0).any() or ((box[:, 3] - box[:, 1]) <= 0).any():
            raise ValueError("pair boxes must satisfy x1 < x2 and y1 < y2")
    log_r = df["log_r"].to_numpy(dtype=np.float64)
    if not np.isfinite(log_r).all():
        raise ValueError("log_r contains non-finite values")
    if df["pair_id"].duplicated().any():
        raise ValueError("duplicate pair_id values")


# ── probe holdout ─────────────────────────────────────────────────────────────


def is_probe_image(image_id: str, fraction: float = 0.01) -> bool:
    """Deterministic probe-holdout membership by sha256(image_id).

    A fixed ``fraction`` of images is reserved for sentinel evaluation and
    NEVER sampled in training. Hash-based so mining order, manifest order and
    resume state cannot change the split.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    digest = hashlib.sha256(str(image_id).encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) / float(16 ** 12)) < fraction


# ── per-image indexed lookup (the dataset hook) ───────────────────────────────


class PairIndex:
    """Per-image lookup over a loaded pair manifest.

    ``prepare_targets(image_id)`` returns the (<= max_pairs_per_image) cached
    pairs in original-image normalized coords, ready for the joint label
    transform (pair_transform.py).
    """

    def __init__(self, pairs: pd.DataFrame, validate: bool = True) -> None:
        if validate:
            validate_pairs(pairs)
        self._targets: Dict[str, Dict[str, np.ndarray]] = {}
        if not pairs.empty:
            for image_id, group in pairs.groupby("image_id", sort=False):
                self._targets[str(image_id)] = {
                    "boxes_a": group[_BOX_COLS_A].to_numpy(dtype=np.float32),
                    "boxes_b": group[_BOX_COLS_B].to_numpy(dtype=np.float32),
                    "log_r": group["log_r"].to_numpy(dtype=np.float32),
                    "pair_id": group["pair_id"].to_numpy(dtype=object),
                }

    @classmethod
    def from_parquet(cls, path: Path) -> "PairIndex":
        return cls(read_pairs(path))

    def __len__(self) -> int:
        return len(self._targets)

    def __contains__(self, image_id: str) -> bool:
        return str(image_id) in self._targets

    def image_ids(self) -> List[str]:
        return list(self._targets.keys())

    def prepare_targets(self, image_id: str) -> Optional[Dict[str, np.ndarray]]:
        """The dataset hook: mined targets for one image, or None.

        Returns:
            ``{"boxes_a": [m, 4] float32 normalized xyxy,
               "boxes_b": [m, 4], "log_r": [m] float32,
               "pair_id": [m] object}`` in ORIGINAL-image coords — the
            caller must push them through the SAME geometric transform as
            the image (R5).
        """
        return self._targets.get(str(image_id))

    def eligible_image_ids(
        self,
        min_pairs: int = 2,
        exclude_probe: bool = True,
        probe_fraction: float = 0.01,
    ) -> List[str]:
        """Training-eligible images: >= min_pairs pairs, probe holdout excluded."""
        out = []
        for image_id, t in self._targets.items():
            if len(t["log_r"]) < min_pairs:
                continue
            if exclude_probe and is_probe_image(image_id, probe_fraction):
                continue
            out.append(image_id)
        return out

    def probe_image_ids(self, probe_fraction: float = 0.01) -> List[str]:
        """The reserved sentinel-probe images present in this manifest."""
        return [i for i in self._targets if is_probe_image(i, probe_fraction)]
