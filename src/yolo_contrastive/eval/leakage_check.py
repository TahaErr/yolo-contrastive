"""Cross-set leakage check — SSL pretrain pool vs downstream eval datasets.

Faz 4.2 — CLI runner on top of the data/dedup pHash machinery (§13.3).

The SSL pool and the downstream evaluation datasets must not overlap: an
image that was pretrained on and then evaluated on inflates results. This
module finds such contamination by perceptual hash.

Workflow:
    1. The SSL pool's pHashes are precomputed once (data/dedup.compute_pool_phashes
       → a parquet sidecar) — the pool is large (~181K images).
    2. The downstream datasets are small, so their pHashes are computed live.
    3. Pool and downstream pHashes are cross-matched; collisions are leakage.

Matching is exact (Hamming = 0) by default. A Hamming threshold can be set to
also catch near-duplicates — the same scene re-encoded at a different JPEG
quality differs by a few bits. Since downstream sets are small, the
pairwise near-duplicate scan is cheap.

CLI:
    python -m yolo_contrastive.eval.leakage_check \\
        --pool-phash pool_phash.parquet \\
        --downstream /data/eval/train /data/eval/valid \\
        --hamming-threshold 5 \\
        --output leaking_ids.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data.dedup import (
    compute_phash,
    cross_set_leakage,
    hamming_distance,
    load_phashes,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

#: A leakage rate above this is flagged as actionable (§13.3 — "leakage > 1%").
DEFAULT_LEAKAGE_ALERT = 0.01


def hash_image_dir(image_dir: str, hash_size: int = 8) -> Dict[str, str]:
    """Compute pHash for every image under ``image_dir`` (recursive).

    Returns ``{image_id: phash}`` where image_id is the directory-relative
    path without extension. Unreadable files are skipped.
    """
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    out: Dict[str, str] = {}
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in _IMG_EXTS and f.is_file():
            image_id = str(f.relative_to(root).with_suffix(""))
            try:
                out[image_id] = compute_phash(f, hash_size=hash_size)
            except Exception:  # noqa: BLE001 — corrupt files are skipped
                pass
    return out


def find_leakage(
    pool_phashes: Dict[str, str],
    downstream_phashes: Dict[str, str],
    hamming_threshold: int = 0,
) -> List[Tuple[str, str, int]]:
    """Cross-match two pHash maps. Returns ``(pool_id, downstream_id, hamming)``.

    hamming_threshold = 0 → exact matches only (fast, hash-bucketed).
    hamming_threshold > 0 → also near-duplicates within that bit distance
        (pairwise scan — fine since the downstream set is small).
    """
    if hamming_threshold <= 0:
        return [
            (pool_id, ds_id, 0)
            for pool_id, ds_id, _ in cross_set_leakage(pool_phashes, downstream_phashes)
        ]
    pairs: List[Tuple[str, str, int]] = []
    for pool_id, pool_h in pool_phashes.items():
        for ds_id, ds_h in downstream_phashes.items():
            dist = hamming_distance(pool_h, ds_h)
            if dist <= hamming_threshold:
                pairs.append((pool_id, ds_id, dist))
    return pairs


def run_leakage_check(
    pool_phash_path: str,
    downstream_dirs: Sequence[str],
    hamming_threshold: int = 0,
    output: str = None,
    hash_size: int = 8,
) -> Dict:
    """Run the full pool ↔ downstream leakage check.

    Args:
        pool_phash_path: parquet sidecar of pool pHashes (compute_pool_phashes).
        downstream_dirs: image directories of the downstream eval datasets.
        hamming_threshold: 0 = exact; > 0 also flags near-duplicates.
        output: if set, write the sorted leaking pool image_ids, one per line.
        hash_size: pHash side length.

    Returns:
        A report dict: pool_size, per-downstream counts, total_leaking_pairs,
        leaking_pool_ids, leakage_rate, alert.
    """
    pool = load_phashes(pool_phash_path)
    report: Dict = {
        "pool_size": len(pool),
        "downstream": {},
        "hamming_threshold": hamming_threshold,
    }

    all_pairs: List[Tuple[str, str, int]] = []
    for ddir in downstream_dirs:
        ds_hashes = hash_image_dir(ddir, hash_size=hash_size)
        pairs = find_leakage(pool, ds_hashes, hamming_threshold)
        report["downstream"][str(ddir)] = {
            "n_images": len(ds_hashes),
            "n_leaking_pairs": len(pairs),
        }
        all_pairs.extend(pairs)

    leaking_ids = sorted({p[0] for p in all_pairs})
    report["total_leaking_pairs"] = len(all_pairs)
    report["leaking_pool_ids"] = leaking_ids
    report["leakage_rate"] = len(leaking_ids) / max(1, len(pool))
    report["alert"] = report["leakage_rate"] > DEFAULT_LEAKAGE_ALERT

    if output:
        with open(output, "w") as f:
            for iid in leaking_ids:
                f.write(iid + "\n")

    return report


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m yolo_contrastive.eval.leakage_check",
        description="Cross-set leakage check — SSL pool vs downstream datasets.",
    )
    p.add_argument(
        "--pool-phash", required=True,
        help="parquet sidecar of pool pHashes (from compute_pool_phashes)",
    )
    p.add_argument(
        "--downstream", required=True, nargs="+",
        help="one or more downstream dataset image directories",
    )
    p.add_argument(
        "--hamming-threshold", type=int, default=0,
        help="0 = exact match; > 0 also flags near-duplicates",
    )
    p.add_argument(
        "--output", default=None,
        help="optional path to write leaking pool image_ids (one per line)",
    )
    p.add_argument("--hash-size", type=int, default=8, help="pHash side length")
    return p


def main(argv: Sequence[str] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_leakage_check(
        pool_phash_path=args.pool_phash,
        downstream_dirs=args.downstream,
        hamming_threshold=args.hamming_threshold,
        output=args.output,
        hash_size=args.hash_size,
    )
    # Print a human-readable summary; the report dict is also JSON-dumped.
    print(json.dumps(report, indent=2))
    if report["alert"]:
        print(
            f"\n[LEAKAGE ALERT] {report['leakage_rate']:.2%} of the pool "
            f"leaks into downstream sets — remove the leaking ids before "
            f"paper-grade runs (§13.3)."
        )
    else:
        print(
            f"\n[OK] leakage rate {report['leakage_rate']:.2%} "
            f"(<= {DEFAULT_LEAKAGE_ALERT:.0%} threshold)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
