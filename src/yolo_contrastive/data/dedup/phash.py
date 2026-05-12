"""Perceptual-hash computation and persistence for SSL pool deduplication.

We use DCT-based pHash because it's stable across our resize + JPEG
re-encoding pipeline — two materializations of the same source frame yield
bit-different JPEGs but the same pHash. That's exactly what we want when
hunting cross-dataset duplicates (BDD ↔ Mapillary scene re-uploads) and
SSL-pool ↔ evaluation-set leakage.

Hashes are 8×8 = 64-bit, stored as 16-character hex strings in a separate
``phash.parquet`` keyed by ``image_id``. We deliberately keep this in a
sidecar file rather than expanding the main manifest schema — pHash is an
augmentation, not part of the canonical pool definition, and downstream
consumers shouldn't need to learn about it unless they care about dedup.

The sha256 ``image_hash`` already in the manifest catches **bit-identical**
duplicates (rare in practice — only if the same file was materialized twice
through the same pipeline). pHash catches **perceptual** duplicates, which
is what actually matters for SSL pool hygiene and eval-set leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import imagehash
import pandas as pd
from PIL import Image

LOG = logging.getLogger(__name__)

#: Default pHash size. 8×8 yields a 64-bit hash, the standard imagehash default.
DEFAULT_HASH_SIZE = 8

#: How many newly-computed pHash rows to buffer before rewriting the parquet.
#: pHash compute is cheap relative to Drive write, so larger batches help.
DEFAULT_FLUSH_EVERY = 1000


def compute_phash(src, hash_size: int = DEFAULT_HASH_SIZE) -> str:
    """Compute pHash of ``src`` as a 16-character hex string.

    ``src`` is anything ``PIL.Image.open`` accepts — a path or a file-like.
    The hex form is chosen so the parquet is grep-able and so equality checks
    work as plain string comparison; convert to int with ``int(h, 16)`` when
    Hamming distance is needed.
    """
    with Image.open(src) as img:
        h = imagehash.phash(img, hash_size=hash_size)
    return str(h)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance (bit count of XOR) between two hex-string pHashes.

    For 64-bit hashes this fits comfortably in a Python int; ``bin(...).count("1")``
    is fast enough that wrapping it in something fancier isn't worth it.
    """
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def compute_pool_phashes(
    pool_root: Path,
    manifest_path: Path,
    output_path: Path,
    hash_size: int = DEFAULT_HASH_SIZE,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    limit: Optional[int] = None,
    log_every: int = 5000,
) -> dict:
    """Compute pHash for every image in ``manifest_path`` and save to parquet.

    Idempotent on ``image_id``: existing rows in ``output_path`` are
    preserved, only manifest entries with no recorded pHash are computed.
    Safe to re-run after a Colab disconnect.

    Args:
        pool_root: SSL pool root — used to resolve relative ``materialized_path``.
        manifest_path: parquet manifest produced by the ssl_pool adapters.
        output_path: parquet sidecar where pHashes are stored.
        hash_size: pHash side length (8 = 64-bit; rarely needs changing).
        flush_every: number of new rows to compute before persisting.
        limit: optional cap on newly-computed rows (for smoke tests).
        log_every: emit progress log every N scanned manifest rows.

    Returns:
        ``{"scanned": int, "skipped_existing": int, "computed": int, "errors": int}``.
    """
    pool_root = Path(pool_root)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_parquet(manifest_path)

    if output_path.exists():
        df_phash = pd.read_parquet(output_path)
        already: Dict[str, str] = dict(zip(df_phash["image_id"], df_phash["phash"]))
    else:
        already = {}

    LOG.info(
        "pHash compute start: %d manifest rows, %d already hashed",
        len(df_manifest),
        len(already),
    )

    stats = {"scanned": 0, "skipped_existing": 0, "computed": 0, "errors": 0}
    # We accumulate into a dict so a flush is just a single parquet write of all rows.
    accumulated: Dict[str, str] = dict(already)
    new_since_flush = 0

    def flush() -> None:
        df_out = pd.DataFrame(
            {"image_id": list(accumulated.keys()), "phash": list(accumulated.values())}
        )
        df_out.to_parquet(output_path, index=False)
        LOG.info("Flushed pHash sidecar: total %d rows", len(df_out))

    for _, row in df_manifest.iterrows():
        stats["scanned"] += 1
        image_id = row["image_id"]
        if image_id in accumulated:
            stats["skipped_existing"] += 1
            continue

        img_path = pool_root / row["materialized_path"]
        try:
            h = compute_phash(img_path, hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 — PIL/IO failures take many shapes
            LOG.warning("pHash failed on %s: %s", image_id, exc)
            stats["errors"] += 1
            continue

        accumulated[image_id] = h
        stats["computed"] += 1
        new_since_flush += 1

        if new_since_flush >= flush_every:
            flush()
            new_since_flush = 0

        if log_every and stats["scanned"] % log_every == 0:
            LOG.info("pHash progress: %s", stats)

        if limit is not None and stats["computed"] >= limit:
            LOG.info("pHash hit limit=%d, stopping", limit)
            break

    if new_since_flush > 0 or not output_path.exists():
        flush()
    LOG.info("pHash compute done: %s", stats)
    return stats


def load_phashes(path: Path) -> Dict[str, str]:
    """Load ``image_id -> phash`` mapping from the parquet sidecar.

    Returns an empty dict if the file doesn't exist yet — the same
    "no-file-is-empty-state" convention the manifest uses.
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["image_id"], df["phash"]))
