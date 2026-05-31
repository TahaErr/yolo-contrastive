"""Example 06 — Assemble the downstream eval pool from Roboflow sources.

Download N Roboflow YOLO exports, verify they share one class, consolidate each
into a single train split, and balance them to a target total via water-filling.
Writes a JSON selection manifest that the cross-validation step consumes.

Run (per-source level fixed, total scales with N):
    python examples/06_downstream_prepare.py \\
        --sources sources.txt --root datasets --per-dataset 500

Or a fixed total budget regardless of source count:
    python examples/06_downstream_prepare.py --sources sources.txt --total 5000

`sources.txt` lists one Roboflow export URL per line (see
yolo_contrastive.data.downstream.load_sources for accepted formats). Keep this
file out of version control — the URLs embed download keys. Network is required
for the download step only.
"""

from __future__ import annotations

import argparse

from yolo_contrastive.data.downstream import prepare_downstream


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble the downstream eval pool")
    p.add_argument("--sources", required=True,
                   help="path to a .txt (one URL per line) or .yaml source list")
    p.add_argument("--root", default="datasets", help="download/work root (gitignored)")
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument("--total", type=int, help="fixed total images across all sources")
    budget.add_argument("--per-dataset", type=int,
                        help="per-source level; total = per_dataset * N")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-download", action="store_true",
                   help="re-download even if a source folder already exists")
    args = p.parse_args()

    manifest = prepare_downstream(
        args.sources, root=args.root, total=args.total,
        per_dataset=args.per_dataset, seed=args.seed,
        force_download=args.force_download,
    )
    print(f"\nselected {manifest['total_selected']} images "
          f"from {manifest['n_sources']} sources")


if __name__ == "__main__":
    main()
