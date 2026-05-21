"""Example 05 — Cross-set leakage check.

The unlabeled SSL pretraining pool must not overlap the downstream evaluation
datasets — an image pretrained on and then evaluated on inflates results.
This script perceptually hashes the downstream images and cross-matches them
against the precomputed pool pHashes.

Run:
    python examples/05_leakage_check.py \\
        --pool-phash pool_phash.parquet \\
        --downstream /data/eval/train /data/eval/valid

`pool_phash.parquet` is produced once over the full pool with
yolo_contrastive.data.dedup.compute_pool_phashes.

Equivalently, the module has a CLI:
    python -m yolo_contrastive.eval.leakage_check --pool-phash ... --downstream ...
"""

from __future__ import annotations

import argparse

from yolo_contrastive import run_leakage_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-set leakage check")
    parser.add_argument("--pool-phash", required=True,
                        help="parquet sidecar of pool pHashes")
    parser.add_argument("--downstream", required=True, nargs="+",
                        help="downstream dataset image directories")
    parser.add_argument("--hamming-threshold", type=int, default=5,
                        help="0 = exact match; > 0 also flags near-duplicates")
    args = parser.parse_args()

    report = run_leakage_check(
        pool_phash_path=args.pool_phash,
        downstream_dirs=args.downstream,
        hamming_threshold=args.hamming_threshold,
    )

    print(f"Pool size:          {report['pool_size']}")
    print(f"Leaking pairs:      {report['total_leaking_pairs']}")
    print(f"Leaking pool images: {len(report['leaking_pool_ids'])}")
    print(f"Leakage rate:       {report['leakage_rate']:.2%}")
    if report["alert"]:
        print("\nALERT — leakage exceeds 1%. Remove the leaking pool images "
              "before paper-grade pretraining.")
    else:
        print("\nOK — leakage within tolerance.")


if __name__ == "__main__":
    main()
