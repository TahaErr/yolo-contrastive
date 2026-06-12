"""Example 09 — TERRA E1 teacher signal-fidelity kill-gate.

THE DECISIVE GATE (wf2_ac.md E1), run BEFORE any training GPU is spent: can
the depth teacher actually see road-surface anomalies? On a LABELED YOLO
dataset, this script runs the full Stage-0 label factory (plane fit +
standardized residuals) per image and measures, across an |z| threshold
sweep:

    * GT pothole / speed-bump RECALL at the correct polarity
      (pothole = depression z < -t, bump = elevation z > +t) — reported both
      over ALL GT boxes and over CLOSE-RANGE boxes only (box center below the
      per-image far-field row threshold from ``compute_label_map``; the
      pre-registered gate quantity is the close-range recall);
    * WRONG-POLARITY rate among detected GT boxes;
    * FALSE-ANOMALY rate on flat road (road pixels outside any GT box).

GO (per the pre-registered gate): recall_near >= 50% (close range) with < 10%
wrong polarity at the chosen ROC operating point. HARD KILL -> the ARK
fallback.

It writes a CSV (one row per threshold) plus a handful of overlay PNGs
(image + residual colorization + GT and mined boxes) for visual audit.

Run with a precomputed depth cache (CPU-only is fine):
    python examples/09_terra_e1_fidelity.py \\
        --dataset /data/pothole4cls/val \\
        --depth-cache /cache/depth --tag depth_anything_v2_small \\
        --class-map "0:depression,3:elevation" --out runs/terra_e1

Or compute depth on the fly (GPU + transformers required):
    python examples/09_terra_e1_fidelity.py \\
        --dataset /data/pothole4cls/val --compute \\
        --model depth-anything/Depth-Anything-V2-Small-hf --out runs/terra_e1

Dataset layout: standard YOLO — images under ``{dataset}/images`` (searched
recursively), each with a ``.txt`` label file under the sibling ``labels``
directory ("cls cx cy w h" normalized).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from yolo_contrastive.geoteach import (
    DepthCache,
    PlaneFitConfig,
    ResidualLabelConfig,
    compute_label_map,
    evaluate_surface,
    fit_road_plane,
    mine_boxes,
    standardized_residual,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
POLARITIES = ("depression", "elevation")


@dataclasses.dataclass
class GtBox:
    polarity: str   # "depression" | "elevation"
    cx: float
    cy: float
    w: float
    h: float


def parse_class_map(spec: str) -> Dict[int, str]:
    """Parse ``"0:depression,3:elevation"`` into {class_id: polarity}."""
    out: Dict[int, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cid, _, pol = part.partition(":")
        pol = pol.strip().lower()
        if pol not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES}, got {pol!r}")
        out[int(cid)] = pol
    if not out:
        raise ValueError("class map is empty")
    return out


def find_images(dataset: Path) -> List[Path]:
    images_dir = dataset / "images" if (dataset / "images").is_dir() else dataset
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def load_gt_boxes(img_path: Path, class_map: Dict[int, str]) -> List[GtBox]:
    path = label_path_for(img_path)
    boxes: List[GtBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        cid = int(float(vals[0]))
        if cid in class_map:
            boxes.append(GtBox(class_map[cid], *(float(v) for v in vals[1:5])))
    return boxes


def get_inverse_depth(
    img_path: Path,
    images_root: Path,
    cache: Optional[DepthCache],
    pipe,
) -> Optional[np.ndarray]:
    """Cache lookup by relative path / stem; on-the-fly compute via ``pipe``."""
    if cache is not None:
        try:
            rel = img_path.relative_to(images_root).with_suffix("").as_posix()
        except ValueError:
            rel = img_path.stem
        for image_id in (rel, img_path.stem):
            if cache.has(image_id):
                return cache.load(image_id)[0]
    if pipe is None:
        return None
    import torch  # lazy
    from PIL import Image  # lazy

    pil = Image.open(img_path).convert("RGB")
    out = pipe(pil)
    pred = out["predicted_depth"]
    if not torch.is_tensor(pred):
        pred = torch.as_tensor(np.asarray(pred))
    pred = pred.detach().float().cpu()
    if pred.ndim == 3:
        pred = pred[0]
    h, w = pil.size[1] // 2, pil.size[0] // 2
    pred = torch.nn.functional.interpolate(
        pred[None, None], size=(max(h, 1), max(w, 1)),
        mode="bilinear", align_corners=False)[0, 0]
    return pred.numpy().astype(np.float32)


def box_pixel_slice(b: GtBox, shape: Tuple[int, int]) -> Tuple[slice, slice]:
    h, w = shape
    x0 = max(int((b.cx - b.w / 2) * w), 0)
    x1 = min(max(int(np.ceil((b.cx + b.w / 2) * w)), x0 + 1), w)
    y0 = max(int((b.cy - b.h / 2) * h), 0)
    y1 = min(max(int(np.ceil((b.cy + b.h / 2) * h)), y0 + 1), h)
    return slice(y0, y1), slice(x0, x1)


def box_is_near(b: GtBox, far_field_mask: np.ndarray) -> bool:
    """Close-range test for the pre-registered gate: a GT box counts as
    near-field when its center pixel is NOT in the image's far-field mask
    (the per-image disparity-percentile row threshold from
    ``compute_label_map`` — beyond it depth noise exceeds pothole amplitude)."""
    h, w = far_field_mask.shape
    cx = min(max(int(b.cx * w), 0), w - 1)
    cy = min(max(int(b.cy * h), 0), h - 1)
    return not bool(far_field_mask[cy, cx])


def overlay_png(out_path: Path, img_path: Path, z: np.ndarray,
                road_mask: np.ndarray, gt: List[GtBox], mined,
                z_vis: float = 6.0) -> None:
    """Image + residual colorization (blue depression / red elevation) +
    GT boxes (green) + mined boxes (yellow)."""
    import cv2  # lazy

    h, w = z.shape
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    mag = np.clip(np.abs(z) / z_vis, 0.0, 1.0)
    color = np.zeros_like(bgr)
    color[..., 0] = np.where(z < 0, (mag * 255), 0).astype(np.uint8)  # B: depression
    color[..., 2] = np.where(z > 0, (mag * 255), 0).astype(np.uint8)  # R: elevation
    color[~road_mask] //= 3
    vis = cv2.addWeighted(bgr, 0.55, color, 0.45, 0.0)
    for b in gt:
        ys, xs = box_pixel_slice(b, z.shape)
        cv2.rectangle(vis, (xs.start, ys.start), (xs.stop - 1, ys.stop - 1), (0, 255, 0), 1)
    for m in mined:
        x0 = int((m.cx - m.w / 2) * w)
        y0 = int((m.cy - m.h / 2) * h)
        x1 = int((m.cx + m.w / 2) * w)
        y1 = int((m.cy + m.h / 2) * h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    parser = argparse.ArgumentParser(description="TERRA E1 teacher-fidelity kill-gate")
    parser.add_argument("--dataset", required=True,
                        help="YOLO dataset dir (images/ + labels/)")
    parser.add_argument("--depth-cache", default=None,
                        help="DepthCache root with precomputed inverse depth")
    parser.add_argument("--tag", default="depth_anything_v2_small",
                        help="depth cache tag (sub-directory)")
    parser.add_argument("--compute", action="store_true",
                        help="compute depth on the fly (GPU + transformers)")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--class-map", default="0:depression",
                        help='downstream class -> polarity, e.g. "0:depression,3:elevation"')
    parser.add_argument("--thresholds", default="1.5,2.0,2.5,3.0,4.0,6.0",
                        help="|z| threshold sweep (comma separated)")
    parser.add_argument("--min-cover", type=float, default=0.05,
                        help="min in-box anomaly pixel fraction to count a detection")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--overlays", type=int, default=8,
                        help="number of overlay PNGs to write")
    parser.add_argument("--out", default="runs/terra_e1")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_map = parse_class_map(args.class_map)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    cache = DepthCache(args.depth_cache, tag=args.tag) if args.depth_cache else None
    if cache is None and not args.compute:
        raise SystemExit("provide --depth-cache and/or --compute")

    pipe = None
    if args.compute:
        from yolo_contrastive.geoteach.depth_cache import _build_pipeline

        pipe = _build_pipeline(args.model, device=None, fp16=True)

    images = find_images(dataset)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"no images found under {dataset}")
    images_root = dataset / "images" if (dataset / "images").is_dir() else dataset

    plane_cfg = PlaneFitConfig(seed=0)
    label_cfg = ResidualLabelConfig()

    n_img = n_skipped_depth = n_untrusted = 0
    n_overlays = 0
    # per-threshold accumulators (all-range + close-range stratification)
    acc = {t: {"hit": 0, "wrong": 0, "detected": 0, "flat_anom_px": 0,
               "hit_near": 0, "wrong_near": 0, "detected_near": 0}
           for t in thresholds}
    n_gt_total = 0
    n_gt_near_total = 0
    flat_px_total = 0
    per_image_rows: List[Dict] = []

    for img_path in images:
        inv_depth = get_inverse_depth(img_path, images_root, cache, pipe)
        if inv_depth is None:
            n_skipped_depth += 1
            continue
        gt = load_gt_boxes(img_path, class_map)
        fit = fit_road_plane(inv_depth, plane_cfg)
        row = {"image": img_path.name, "trusted": fit.trusted,
               "inlier_ratio": round(fit.inlier_ratio, 4),
               "sigma_mad": fit.sigma_mad, "n_gt": len(gt)}
        per_image_rows.append(row)
        if not fit.trusted:
            n_untrusted += 1
            continue
        n_img += 1
        z = standardized_residual(inv_depth, fit)
        d_surf = evaluate_surface(fit.params, z.shape)
        lm = compute_label_map(z, fit.inlier_mask, d_surf, label_cfg)

        # flat-road pixels: road region, near field, outside every GT box
        flat = lm.road_region & ~lm.far_field_mask
        for b in gt:
            ys, xs = box_pixel_slice(b, z.shape)
            flat[ys, xs] = False
        flat_px_total += int(flat.sum())
        n_gt_total += len(gt)
        near_flags = [box_is_near(b, lm.far_field_mask) for b in gt]
        n_gt_near_total += sum(near_flags)

        for t in thresholds:
            a = acc[t]
            a["flat_anom_px"] += int((np.abs(z[flat]) >= t).sum())
            for b, near in zip(gt, near_flags):
                ys, xs = box_pixel_slice(b, z.shape)
                zz = z[ys, xs]
                if zz.size == 0:
                    continue
                correct = float((zz <= -t).mean() if b.polarity == "depression"
                                else (zz >= t).mean())
                wrong = float((zz >= t).mean() if b.polarity == "depression"
                              else (zz <= -t).mean())
                if max(correct, wrong) >= args.min_cover:
                    a["detected"] += 1
                    a["detected_near"] += int(near)
                    if correct >= wrong:
                        a["hit"] += 1
                        a["hit_near"] += int(near)
                    else:
                        a["wrong"] += 1
                        a["wrong_near"] += int(near)

        if n_overlays < args.overlays:
            mined = mine_boxes(z, lm, label_cfg)
            overlay_png(out_dir / "overlays" / f"{img_path.stem}.png",
                        img_path, z, lm.road_region, gt, mined)
            n_overlays += 1

    # ── report ────────────────────────────────────────────────────────────
    csv_path = out_dir / "e1_fidelity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["z_threshold", "n_gt", "recall",
                         "n_gt_near", "recall_near", "wrong_polarity_rate",
                         "wrong_polarity_rate_near", "false_anomaly_rate",
                         "n_detected"])
        for t in thresholds:
            a = acc[t]
            recall = a["hit"] / n_gt_total if n_gt_total else float("nan")
            recall_near = (a["hit_near"] / n_gt_near_total
                           if n_gt_near_total else float("nan"))
            wrong_rate = a["wrong"] / a["detected"] if a["detected"] else 0.0
            wrong_rate_near = (a["wrong_near"] / a["detected_near"]
                               if a["detected_near"] else 0.0)
            false_rate = a["flat_anom_px"] / flat_px_total if flat_px_total else float("nan")
            writer.writerow([t, n_gt_total, f"{recall:.4f}",
                             n_gt_near_total, f"{recall_near:.4f}",
                             f"{wrong_rate:.4f}", f"{wrong_rate_near:.4f}",
                             f"{false_rate:.6f}", a["detected"]])

    (out_dir / "e1_per_image.json").write_text(
        json.dumps(per_image_rows, indent=2), encoding="utf-8")

    print(f"Images evaluated:     {n_img}")
    print(f"Untrusted plane fits: {n_untrusted}")
    print(f"Missing depth:        {n_skipped_depth}")
    print(f"GT geometry boxes:    {n_gt_total} (close-range: {n_gt_near_total})")
    print(f"CSV:                  {csv_path}")
    print(f"Overlays:             {out_dir / 'overlays'} ({n_overlays})")
    print()
    print(f"{'|z|>=':>6} {'recall':>8} {'rec-near':>9} {'wrongpol':>9} {'false-anom':>11}")
    for t in thresholds:
        a = acc[t]
        recall = a["hit"] / n_gt_total if n_gt_total else float("nan")
        recall_near = (a["hit_near"] / n_gt_near_total
                       if n_gt_near_total else float("nan"))
        wrong_rate = a["wrong"] / a["detected"] if a["detected"] else 0.0
        false_rate = a["flat_anom_px"] / flat_px_total if flat_px_total else float("nan")
        print(f"{t:6.2f} {recall:8.3f} {recall_near:9.3f} {wrong_rate:9.3f} "
              f"{false_rate:11.6f}")
    print()
    print("GATE: GO if recall_near >= 0.50 (the pre-registered close-range "
          "quantity) with wrong-polarity < 0.10 at the chosen operating point; "
          "otherwise retry with tiling / DA-v2-Base, then fall back to ARK "
          "(wf2_ac.md E1).")


if __name__ == "__main__":
    main()
