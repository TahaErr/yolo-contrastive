"""Example 10 — Prepare the REVISIT cross-traversal pair pool.

Two entry modes producing the same manifest-driven pool layout
(``pairs.parquet`` + ``images/*.jpg``), then the shared offline stages
(align -> propose -> label -> stats):

MINE mode (network; needs MAPILLARY_TOKEN):
    set MAPILLARY_TOKEN=MLY|...
    python examples/10_revisit_prepare.py --root pool --mine \\
        [--cities cities.json] [--max-per-city 2500]

LOCAL mode (fully offline; e.g. your own repeated dashcam captures):
    python examples/10_revisit_prepare.py --root pool --local-dir my_pairs

    ``my_pairs/`` layout: one subdirectory per co-located pair holding
    exactly two images (sorted by name: first = traversal A, second = B):
        my_pairs/corner_05/2023-01.jpg
        my_pairs/corner_05/2024-03.jpg
    GPS metadata is unavailable locally, so gate fields are filled with
    zeros — the homography trust gates in the align stage are the actual
    filter, exactly as in the mined pipeline.

Then train (see the anchored trainer docs):
    from yolo_contrastive.anchored import AnchoredJointTrainer
    from yolo_contrastive.persistence import PersistenceChannel
    ch = PersistenceChannel(pairs_path="pool/pairs.parquet",
                            labels_path="pool/persistence_labels.parquet")
    AnchoredJointTrainer(model="yolov8n.pt", channels=[ch]).train()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_local_manifest(local_dir: Path, pairs_path: Path) -> int:
    """Register every two-image subdirectory of ``local_dir`` as a queued ->
    downloaded pair (paths already local). Idempotent."""
    from yolo_contrastive.persistence import pair_manifest as pm

    rows = []
    for sub in sorted(p for p in Path(local_dir).iterdir() if p.is_dir()):
        imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if len(imgs) != 2:
            print(f"  skipping {sub.name}: expected exactly 2 images, found {len(imgs)}")
            continue
        a, b = imgs
        rows.append(pm.new_pair_row(
            pair_id=f"local_{sub.name}",
            img_a_id=f"local_{sub.name}_a", img_b_id=f"local_{sub.name}_b",
            path_a=str(a), path_b=str(b),
            lon_a=0.0, lat_a=0.0, lon_b=0.0, lat_b=0.0,
            dist_m=0.0, heading_a=0.0, heading_b=0.0, heading_diff=0.0,
            captured_at_a=0, captured_at_b=0, dt_days=0.0,
            seq_a=f"{sub.name}_a", seq_b=f"{sub.name}_b",
            city="local", tile_id="local_0_0",
            status="downloaded",
        ))
    return pm.append_pairs(pairs_path, rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare the REVISIT pair pool")
    p.add_argument("--root", required=True, help="pool root directory")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mine", action="store_true",
                      help="mine + download via the Mapillary Graph API "
                           "(MAPILLARY_TOKEN required)")
    mode.add_argument("--local-dir", default=None,
                      help="offline: directory of two-image pair subdirectories")
    p.add_argument("--cities", default=None,
                   help="JSON city list [{name, lat, lon, radius_km}]; "
                        "default = built-in 12 cities")
    p.add_argument("--max-per-city", type=int, default=2500)
    p.add_argument("--backend", default="auto", choices=("auto", "cheap", "fastsam"),
                   help="proposal backend (fastsam needs --weights)")
    p.add_argument("--weights", default=None, help="LOCAL FastSAM weights path")
    p.add_argument("--workers", type=int, default=0, help="alignment workers")
    p.add_argument("--audit-dir", default=None,
                   help="render GO/NO-GO audit JPGs into this directory")
    args = p.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    pairs_path = root / "pairs.parquet"
    proposals_path = root / "proposals.parquet"
    labels_path = root / "persistence_labels.parquet"

    # ── stage 1+2: candidates + images ────────────────────────────────────
    if args.mine:
        from yolo_contrastive.persistence.mapillary_pairs import (
            download_images, mine_pairs,
        )

        cities = None
        if args.cities:
            with open(args.cities, encoding="utf-8") as f:
                cities = json.load(f)
        n = mine_pairs(pairs_path, cities=cities, max_pairs_per_city=args.max_per_city)
        print(f"[mine] {n} new candidate pairs")
        n = download_images(pairs_path, root)
        print(f"[download] {n} images fetched")
    else:
        n = build_local_manifest(Path(args.local_dir), pairs_path)
        print(f"[local] {n} new pairs registered from {args.local_dir}")

    # ── stage 3: alignment + trust gates ──────────────────────────────────
    from yolo_contrastive.persistence.align import align_manifest

    res = align_manifest(pairs_path, workers=args.workers)
    print(f"[align] aligned={res['aligned']} rejected={res['rejected']}")

    # ── stage 4: class-agnostic proposals ─────────────────────────────────
    from yolo_contrastive.persistence.proposals import propose_manifest

    n = propose_manifest(pairs_path, proposals_path,
                         backend=args.backend, weights=args.weights)
    print(f"[propose] {n} new proposals")

    # ── stage 5: persistence labels (+ optional audit render) ────────────
    from yolo_contrastive.persistence.persistence_labels import (
        label_manifest, render_audit_samples,
    )

    res = label_manifest(pairs_path, proposals_path, labels_path)
    print(f"[label] {res}")
    if args.audit_dir:
        k = render_audit_samples(pairs_path, labels_path, args.audit_dir)
        print(f"[audit] {k} images -> {args.audit_dir} "
              f"(GO gate: >= 80% plausible, >= 10K aligned pairs)")

    # ── summary ───────────────────────────────────────────────────────────
    from yolo_contrastive.persistence import pair_manifest as pm

    pairs = pm.read_pairs(pairs_path)
    ok = int(pairs["align_ok"].sum()) if len(pairs) else 0
    print(f"\npool ready: {ok} aligned pairs at {root}\n"
          f"  pairs:  {pairs_path}\n  labels: {labels_path}")


if __name__ == "__main__":
    main()
