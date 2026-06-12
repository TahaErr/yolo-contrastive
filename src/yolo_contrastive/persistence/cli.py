"""Command-line entrypoint for the REVISIT offline pair factory.

Thin argparse wrapper over the five resumable stages + a stats report:

    python -m yolo_contrastive.persistence.cli mine     --pairs pool/pairs.parquet
    python -m yolo_contrastive.persistence.cli download --pairs pool/pairs.parquet --root pool
    python -m yolo_contrastive.persistence.cli align    --pairs pool/pairs.parquet --workers 8
    python -m yolo_contrastive.persistence.cli propose  --pairs pool/pairs.parquet \
        --proposals pool/proposals.parquet [--backend fastsam --weights FastSAM-s.pt]
    python -m yolo_contrastive.persistence.cli label    --pairs pool/pairs.parquet \
        --proposals pool/proposals.parquet --labels pool/persistence_labels.parquet
    python -m yolo_contrastive.persistence.cli stats    --pairs pool/pairs.parquet ...

Every stage is idempotent and append-dedup resumable; re-running after an
interruption continues where it stopped. Mining/downloading need the
MAPILLARY_TOKEN env var (everything else is offline).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _add_pairs_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pairs", required=True, help="pairs.parquet path")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m yolo_contrastive.persistence.cli",
        description="REVISIT cross-traversal pair factory (mine/download/align/"
                    "propose/label/stats)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("mine", help="mine candidate pairs from the Mapillary Graph API")
    _add_pairs_arg(sp)
    sp.add_argument("--cities", default=None,
                    help="JSON file: [{name, lat, lon, radius_km}, ...]; default = "
                         "built-in 12-city list")
    sp.add_argument("--max-per-city", type=int, default=2500)

    sp = sub.add_parser("download", help="download pair thumbnails (resumable)")
    _add_pairs_arg(sp)
    sp.add_argument("--root", required=True, help="pool root (images go to root/images)")

    sp = sub.add_parser("align", help="ORB/SIFT + MAGSAC homography alignment")
    _add_pairs_arg(sp)
    sp.add_argument("--workers", type=int, default=0)

    sp = sub.add_parser("propose", help="class-agnostic blob proposals per image")
    _add_pairs_arg(sp)
    sp.add_argument("--proposals", required=True, help="proposals.parquet path")
    sp.add_argument("--backend", default="auto", choices=("auto", "cheap", "fastsam"))
    sp.add_argument("--weights", default=None, help="LOCAL FastSAM weights path")

    sp = sub.add_parser("label", help="cross-traversal persistence labeling")
    _add_pairs_arg(sp)
    sp.add_argument("--proposals", required=True)
    sp.add_argument("--labels", required=True, help="persistence_labels.parquet path")
    sp.add_argument("--audit-dir", default=None,
                    help="also render audit JPGs (GO/NO-GO gate) into this dir")
    sp.add_argument("--audit-n", type=int, default=200)

    sp = sub.add_parser("stats", help="print manifest summary (JSON)")
    _add_pairs_arg(sp)
    sp.add_argument("--proposals", default=None)
    sp.add_argument("--labels", default=None)

    return p


def _load_cities(path: Optional[str]):
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "mine":
        from .mapillary_pairs import mine_pairs

        n = mine_pairs(args.pairs, cities=_load_cities(args.cities),
                       max_pairs_per_city=args.max_per_city)
        print(f"mined {n} new candidate pairs -> {args.pairs}")

    elif args.cmd == "download":
        from .mapillary_pairs import download_images

        n = download_images(args.pairs, args.root)
        print(f"downloaded {n} images -> {args.root}/images")

    elif args.cmd == "align":
        from .align import align_manifest

        res = align_manifest(args.pairs, workers=args.workers)
        print(f"aligned {res['aligned']}, rejected {res['rejected']}")

    elif args.cmd == "propose":
        from .proposals import propose_manifest

        n = propose_manifest(args.pairs, args.proposals,
                             backend=args.backend, weights=args.weights)
        print(f"wrote {n} new proposals -> {args.proposals}")

    elif args.cmd == "label":
        from .persistence_labels import label_manifest, render_audit_samples

        res = label_manifest(args.pairs, args.proposals, args.labels)
        print(json.dumps(res))
        if args.audit_dir:
            k = render_audit_samples(args.pairs, args.labels, args.audit_dir,
                                     n=args.audit_n)
            print(f"rendered {k} audit images -> {args.audit_dir}")

    elif args.cmd == "stats":
        from . import pair_manifest as pm

        pairs = pm.read_pairs(args.pairs)
        out = {
            "pairs_total": int(len(pairs)),
            "by_status": {} if pairs.empty
            else {k: int(v) for k, v in pairs["status"].value_counts().items()},
            "by_city": {} if pairs.empty
            else {k: int(v) for k, v in pairs["city"].value_counts().items()},
        }
        if not pairs.empty and (pairs["status"] != "queued").any():
            attempted = pairs[pairs["status"].isin(["aligned", "rejected"])]
            if len(attempted):
                out["align_acceptance"] = float(attempted["align_ok"].mean())
        if args.proposals:
            out["proposals"] = int(len(pm.read_proposals(args.proposals)))
        if args.labels:
            labels = pm.read_labels(args.labels)
            out["labels_total"] = int(len(labels))
            if len(labels):
                out["labels_by_class"] = {
                    k: int(v) for k, v in labels["label"].value_counts().items()
                }
        print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
