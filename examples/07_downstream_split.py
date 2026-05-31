"""Example 07 — Build source-disjoint holdout or CV splits from the selection manifest.

Consumes the ``selection_manifest.json`` from example 06 and emits YOLO-native
splits (data.yaml + image-list txt, no image copying). Every regime is
source-disjoint: train and val never share a source, so frame-adjacent
near-duplicates from the same camera cannot leak across the boundary.

Holdout (source-disjoint train/val/test, target ratios 70/15/15):
    python examples/07_downstream_split.py \\
        --manifest datasets/selection_manifest.json --out datasets/splits \\
        --mode holdout --ratios 0.7 0.15 0.15

Group k-fold CV (default; each fold validates on a group of whole sources):
    python examples/07_downstream_split.py \\
        --manifest datasets/selection_manifest.json --out datasets/splits \\
        --mode cv --scheme group_kfold --k 5

Leave-one-source-out (one held-out source per fold; one run per source):
    python examples/07_downstream_split.py \\
        --manifest datasets/selection_manifest.json --out datasets/splits \\
        --mode cv --scheme logo
"""

from __future__ import annotations

import argparse

from yolo_contrastive.data.downstream import build_cv_splits, build_holdout_split


def main() -> None:
    p = argparse.ArgumentParser(description="Build source-disjoint holdout / CV splits")
    p.add_argument("--manifest", required=True, help="path to selection_manifest.json")
    p.add_argument("--out", default="datasets/splits", help="output root for splits")
    p.add_argument("--mode", choices=["holdout", "cv"], required=True)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"), help="holdout ratios (sum to 1)")
    p.add_argument("--scheme", choices=["group_kfold", "logo"], default="group_kfold",
                   help="cv scheme (source-disjoint)")
    p.add_argument("--k", type=int, default=5, help="number of folds for group_kfold")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "holdout":
        build_holdout_split(args.manifest, args.out, ratios=tuple(args.ratios), seed=args.seed)
    else:
        build_cv_splits(args.manifest, args.out, scheme=args.scheme, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
