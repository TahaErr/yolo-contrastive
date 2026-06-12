"""Example 11 — GASP-Real offline pair mining (natural scale labels).

Mines pairs of similar-content, depth-separated patches from a directory of
images + a METRIC depth cache, writes the pair manifest parquet
(+ mining_stats.json) and renders sample side-by-side visualizations with
log_r overlays for the human audit gate.

Run on real data (after the geoteach depth-cache pass)::

    python examples/11_gasp_real_pairs.py \
        --images /data/pool_images \
        --depth-cache /content/cache \
        --out /content/cache/scalereal/pairs_v1.parquet \
        --embedder dinov2_vits14 --audit 50

Fully-offline demo (synthetic pinhole scenes + exact metric depth + stub
embedder; no downloads, CPU-only)::

    python examples/11_gasp_real_pairs.py --demo --out runs/scalereal_demo

The production miner CLI (manifest-driven, resumable) is
``python -m yolo_contrastive.scalereal.mine_pairs``; this example is the
directory-of-images convenience wrapper around the same machinery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_manifest_from_dir(images_dir: Path):
    """Ad-hoc pool manifest for a flat image directory (image_id = stem)."""
    import pandas as pd

    rows = []
    for path in sorted(images_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            rows.append({
                "image_id": path.stem,
                "dataset": images_dir.name,
                "materialized_path": str(path),
            })
    if not rows:
        raise SystemExit(f"no images found under {images_dir}")
    return pd.DataFrame(rows)


def run_demo(out_dir: Path):
    """Synthetic scenes -> metric depth cache -> mined pairs -> audit PNGs."""
    from yolo_contrastive.scalereal.depth_io import DepthCache
    from yolo_contrastive.scalereal.synthetic import (
        materialize_scene,
        row_decorrelated_scene,
        two_class_scene,
    )

    images_dir = out_dir / "images"
    cache = DepthCache(out_dir / "cache", variant="dav2_metric_outdoor_small")
    scenes = {
        "demo_two_class": two_class_scene(seed=0),
        "demo_row_decorrelated": row_decorrelated_scene(seed=1),
    }
    for image_id, scene in scenes.items():
        materialize_scene(scene, images_dir, image_id, depth_cache=cache)
    cache.save_metadata({"source": "examples/11_gasp_real_pairs.py --demo"})

    # scene-aware stub embedder keyed by geometry (both demo scenes share a
    # square layout class map; dispatch per image via a tiny closure)
    stubs = {i: s.make_stub_embedder() for i, s in scenes.items()}
    sizes = {i: s.image.shape[:2] for i, s in scenes.items()}

    def embed(image, boxes):
        for image_id, hw in sizes.items():
            if image.shape[:2] == hw and _matches(image, scenes[image_id]):
                return stubs[image_id](image, boxes)
        return stubs["demo_two_class"](image, boxes)

    def _matches(image, scene):
        import numpy as np

        return float(np.abs(image - scene.image).mean()) < 0.02

    return images_dir, cache, embed, build_manifest_from_dir(images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="GASP-Real pair mining example")
    parser.add_argument("--images", help="directory of pool images")
    parser.add_argument("--depth-cache",
                        help="shared cache root (depth/{variant}/ under it)")
    parser.add_argument("--variant", default="dav2_metric_outdoor_small")
    parser.add_argument("--out", default="runs/scalereal_pairs",
                        help="output dir (pairs.parquet + stats + audit/)")
    parser.add_argument("--embedder", default="stub",
                        help="'stub' (offline) or a DINOv2 hub name "
                             "(e.g. dinov2_vits14; downloads weights)")
    parser.add_argument("--audit", type=int, default=24,
                        help="number of audit visualizations to render")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="fully-offline synthetic demo (ignores --images)")
    args = parser.parse_args()

    from yolo_contrastive.scalereal.config import ScaleRealConfig
    from yolo_contrastive.scalereal.depth_io import DepthCache
    from yolo_contrastive.scalereal.mine_pairs import (
        build_embedder,
        mine_pool,
        render_audit,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.parquet"
    cfg = ScaleRealConfig()

    if args.demo:
        _, cache, embed_fn, manifest = run_demo(out_dir)
    else:
        if not args.images or not args.depth_cache:
            parser.error("--images and --depth-cache are required without --demo")
        manifest = build_manifest_from_dir(Path(args.images))
        cache = DepthCache(args.depth_cache, variant=args.variant)
        embed_fn = build_embedder(args.embedder)

    stats = mine_pool(manifest, cache, pairs_path, embed_fn, cfg, limit=args.limit)
    d = stats.to_dict()
    print(f"pairs written : {d['counters']['pairs_written']}")
    print(f"image yield   : {d['image_yield']:.1%} (go/no-go gate: >= 40%)")
    print(f"log_r |bins|  : {d['log_r_hist']} (all four bins should be populated)")
    print(f"pair manifest : {pairs_path}")
    print(f"mining stats  : {pairs_path.parent / 'mining_stats.json'}")

    if args.audit:
        files = render_audit(pairs_path, manifest, out_dir / "audit",
                             n=args.audit, cfg=cfg)
        print(f"audit mosaics : {len(files)} files in {out_dir / 'audit'}")
        print("audit gate    : >= 60% of pairs judged same-content with "
              "plausible relative scale")


if __name__ == "__main__":
    main()
