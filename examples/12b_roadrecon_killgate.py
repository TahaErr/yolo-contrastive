"""Example 12b — roadrecon kill-gate: is the reconstruction-mined signal worth training on?

Decisive cheap test BEFORE spending GPU on M3. Loads a trained reconstructor (from
example 12), mines anomaly boxes over a LABELED pothole set, matches them to the
ground-truth boxes, and reports:

    * precision  = mined-box **purity** (are the anomalies actually potholes, not
      shadows / lane marks / patches?)
    * recall     = fraction of GT potholes the mined signal covers
    * small_recall = recall on the small/hard tail (the cell that separates beat from tie)

GO if precision >= --min-precision AND small_recall >= --min-small-recall; otherwise
KILL — the appearance-anomaly signal is impure or misses the hard cases, so M3 will
tie at best. (Pair this with an incrementality probe vs. a non-COCO baseline.)

Invocation
----------
    python examples/12b_roadrecon_killgate.py \
        --reconstructor runs/roadrecon/roadrecon_full.pt \
        --dataset /content/datasets/pothole/val \
        --z-thresh 3.0 --iou 0.3 --out runs/roadrecon_killgate
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_yolo_boxes(label_path: Path) -> np.ndarray:
    """Read GT ``[cx, cy, w, h]`` boxes from a YOLO label file (class ignored)."""
    if not label_path.exists():
        return np.zeros((0, 4), np.float32)
    rows = []
    for ln in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[1:5]])
    return np.asarray(rows, np.float32).reshape(-1, 4)


def _label_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="roadrecon anomaly-mining kill-gate")
    ap.add_argument("--reconstructor", required=True, help="roadrecon_full.pt (from example 12)")
    ap.add_argument("--dataset", required=True,
                    help="labeled YOLO set (with images/ + labels/, or a data.yaml's val dir)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--z-thresh", type=float, default=3.0)
    ap.add_argument("--min-box-area", type=int, default=256,
                    help="mining min component area (px); SWEEP this (e.g. 64/128/256) and watch "
                         "small_recall vs precision — the default is inherited from the depth factory "
                         "and may exclude the small-pothole tail")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--small-frac", type=float, default=0.02,
                    help="GT area (fraction of image) below which a box is 'small/hard'")
    ap.add_argument("--min-precision", type=float, default=0.5)
    ap.add_argument("--min-small-recall", type=float, default=0.3)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="runs/roadrecon_killgate")
    args = ap.parse_args()

    ckpt = Path(args.reconstructor)
    if not ckpt.exists():
        raise SystemExit(f"--reconstructor not found: {ckpt}")
    ds = Path(args.dataset)
    images_dir = ds / "images" if (ds / "images").is_dir() else ds
    labels_dir = ds / "labels" if (ds / "labels").is_dir() else ds
    imgs = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images under {images_dir}")
    if args.max_images > 0:
        imgs = imgs[: args.max_images]

    from yolo_contrastive.roadrecon import (
        AnomalyMineConfig, load_reconstructor, mine_image_boxes, box_iou_xywh,
    )
    from yolo_contrastive.roadrecon.mining import _load_square_rgb
    from yolo_contrastive.geoteach import trapezoid_mask
    import torch

    rec = load_reconstructor(str(ckpt), device=args.device)
    imgsz = rec.imgsz
    road = trapezoid_mask((imgsz, imgsz))
    cfg = AnomalyMineConfig(z_thresh=args.z_thresh, min_box_area_px=args.min_box_area)

    tp = fp = fn = 0
    small_total = small_tp = 0
    n_with_gt = 0
    for p in imgs:
        gt = _read_yolo_boxes(_label_for(p, images_dir, labels_dir))
        rgb, _ = _load_square_rgb(str(p), imgsz)
        err = rec.error_map(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))[0].cpu().numpy()
        boxes = mine_image_boxes(err, road, cfg)
        pred = (np.array([[b.cx, b.cy, b.w, b.h] for b in boxes], np.float32)
                if boxes else np.zeros((0, 4), np.float32))

        if len(gt):
            n_with_gt += 1
        iou = box_iou_xywh(pred, gt)
        matched = set()
        for pi in range(len(pred)):
            if len(gt) == 0:
                break
            for gi in np.argsort(-iou[pi]):        # best UNMATCHED GT above threshold
                if iou[pi, gi] < args.iou:
                    break
                if int(gi) not in matched:
                    matched.add(int(gi))
                    break
        tp += len(matched)
        fp += len(pred) - len(matched)
        fn += len(gt) - len(matched)
        small = [gi for gi in range(len(gt)) if gt[gi, 2] * gt[gi, 3] < args.small_frac]
        small_total += len(small)
        small_tp += sum(1 for gi in small if gi in matched)
    rec.cleanup()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    small_recall = small_tp / max(1, small_total)
    go = precision >= args.min_precision and small_recall >= args.min_small_recall

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "killgate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["images", "images_with_gt", "tp", "fp", "fn",
                    "precision", "recall", "small_total", "small_recall", "verdict"])
        w.writerow([len(imgs), n_with_gt, tp, fp, fn,
                    f"{precision:.4f}", f"{recall:.4f}", small_total,
                    f"{small_recall:.4f}", "GO" if go else "KILL"])

    print(f"[12b] images={len(imgs)} (with GT={n_with_gt}) | tp={tp} fp={fp} fn={fn}")
    print(f"[12b] precision(purity)={precision:.3f}  recall={recall:.3f}  "
          f"small_recall={small_recall:.3f} (n_small={small_total})")
    print(f"GATE: {'GO' if go else 'KILL'} "
          f"(need precision>={args.min_precision}, small_recall>={args.min_small_recall})")


if __name__ == "__main__":
    main()
