"""Example 13 — B1 zero-GPU smoke test: does the plane-residual machinery work on REAL GT depth?

The cheapest, purest B1 falsification — no model, no training, no GPU. Ingest a few Cityscapes
disparity frames as inverse-depth GT, then run the SAME consumer M1 will use
(`fit_road_plane` → `standardized_residual`) on that perfect stereo depth and check:

    * the road plane fits and is TRUSTED (RANSAC inlier ratio ok),
    * the road is FLAT in the residual (most road pixels have |z| ≈ 0),
    * σ_MAD is sane.

If perfect stereo GT cannot produce a trusted, flat road-plane residual, the geometry premise
is dead before any depth net is trained. (Cityscapes has ~no potholes, so this validates the
MACHINERY, not pothole recall — that comes from the mid-build recall gate on a labeled set.)

Invocation
----------
    # 1) ingest a slice of a Cityscapes disparity zip into a DepthGT store
    # 2) smoke-test the plane fit on it
    python examples/13_b1_depth_gt_smoke.py \
        --disparity-zip /content/drive/MyDrive/SSL_DATA/disparity_trainvaltest.zip \
        --gt-root runs/b1_gt --limit 200 --n-audit 12 --out runs/b1_depth_gt_smoke
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="B1 zero-GPU plane-fit smoke on real Cityscapes GT depth")
    ap.add_argument("--disparity-zip", help="Cityscapes disparity_*.zip (ingested if --gt-root is empty)")
    ap.add_argument("--gt-root", default="runs/b1_gt", help="DepthGT store root")
    ap.add_argument("--tag", default="cityscapes_disp")
    ap.add_argument("--limit", type=int, default=200, help="frames to ingest (if ingesting)")
    ap.add_argument("--n-audit", type=int, default=12, help="overlay PNGs to write")
    ap.add_argument("--flat-z", type=float, default=2.0, help="|z| below which a road pixel is 'flat'")
    ap.add_argument("--out", default="runs/b1_depth_gt_smoke")
    args = ap.parse_args()

    from yolo_contrastive.geoteach.depth_gt import DepthGT
    from yolo_contrastive.geoteach.plane_fit import fit_road_plane, standardized_residual

    gt = DepthGT(args.gt_root, tag=args.tag)

    if len(gt) == 0:
        if not args.disparity_zip:
            raise SystemExit("empty GT store and no --disparity-zip to ingest from")
        from yolo_contrastive.data.ssl_pool.cityscapes_disparity import ingest_disparity
        print(f"[13] ingesting up to {args.limit} disparity frames → {gt.dir} ...")
        stats = ingest_disparity(Path(args.disparity_zip), gt, limit=args.limit)
        print(f"[13] ingest: {stats}")

    ids = gt.image_ids()
    if not ids:
        raise SystemExit("no GT frames available after ingest")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    trusted = 0
    flat_fracs = []
    sigmas = []
    for image_id in ids:
        inv, valid, _ = gt.load(image_id)
        fit = fit_road_plane(inv, valid_mask=valid)
        rec = {"image_id": image_id, "trusted": int(fit.trusted),
               "inlier_ratio": round(fit.inlier_ratio, 4), "sigma_mad": round(fit.sigma_mad, 6)}
        if fit.trusted:
            trusted += 1
            z = standardized_residual(inv, fit)
            road = fit.inlier_mask & valid
            if road.any():
                flat = float((np.abs(z[road]) <= args.flat_z).mean())
                rec["flat_frac"] = round(flat, 4)
                flat_fracs.append(flat)
                sigmas.append(fit.sigma_mad)
        rows.append(rec)

    with open(out / "smoke.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "trusted", "inlier_ratio", "sigma_mad", "flat_frac"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    _write_overlays(gt, rows, out, args.n_audit)

    n = len(ids)
    trusted_frac = trusted / max(1, n)
    med_flat = float(np.median(flat_fracs)) if flat_fracs else 0.0
    print(f"[13] frames={n} | trusted={trusted_frac:.3f} | median flat_frac={med_flat:.3f} "
          f"| median sigma_mad={float(np.median(sigmas)) if sigmas else 0.0:.5f}")
    ok = trusted_frac >= 0.7 and med_flat >= 0.9
    print(f"GATE: {'OK — plane-residual machinery is sound on real GT depth' if ok else 'CHECK — weak plane fit on GT'} "
          f"(need trusted ≥ 0.70 and median flat_frac ≥ 0.90)")


def _write_overlays(gt, rows, out: Path, n_audit: int) -> None:
    if n_audit <= 0:
        return
    try:
        import cv2
    except Exception:
        return
    from yolo_contrastive.geoteach.plane_fit import fit_road_plane, standardized_residual
    (out / "overlays").mkdir(exist_ok=True)
    picked = [r["image_id"] for r in rows if r.get("trusted")][:n_audit]
    for image_id in picked:
        inv, valid, _ = gt.load(image_id)
        fit = fit_road_plane(inv, valid_mask=valid)
        if not fit.trusted:
            continue
        z = standardized_residual(inv, fit)
        vis = np.clip((z + 6) / 12.0, 0, 1)  # depression(blue) .. elevation(red)
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis[~valid] = 0
        cv2.imwrite(str(out / "overlays" / f"{image_id.replace('/', '_')}.png"), vis)


if __name__ == "__main__":
    main()
