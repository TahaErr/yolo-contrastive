"""Anomaly-label factory (Stage 0) — reconstruction error → mined pothole boxes.

Offline, resumable, single-image-at-a-time pass turning the B2 reconstructor's
per-pixel error map into a YOLO detection dataset the M3 pretrainer trains on:

    error_map  →  road-region prior  →  robust z-threshold  →  morphology
               →  connected components  →  boxes  →  YOLO images/ + labels/ + data.yaml

Design mirrors the geoteach label factory (``residual_labels``): a small config
dataclass holds every threshold; robust MAD standardization; per-image trust gates
(a global reconstruction failure floods the road with "anomaly" → skipped). Boxes
are single-class (the downstream pothole class).

cv2 and numpy are used for image/label IO and morphology (imported lazily). The
mined images are written at ``imgsz`` so normalized boxes align exactly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # annotations only — MinedBox is imported lazily at call time
    from ..geoteach.residual_labels import MinedBox


@dataclasses.dataclass
class AnomalyMineConfig:
    """Every anomaly-mining threshold in one dataclass (the ablation grid)."""

    z_thresh: float = 3.0            # robust z over road error → anomaly
    morph_kernel: int = 3           # open/close kernel side (px)
    min_box_area_px: int = 256      # 16^2 minimum component area
    min_box_side_px: int = 4        # minimum box side (px)
    max_anomaly_area_frac: float = 0.10   # of road area → global recon failure, skip
    use_road_prior: bool = True     # restrict mining to the trapezoid road region
    class_name: str = "pothole"     # single downstream class


def _morph_open_close(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Morphological open then close (cv2 lazy) — despeckle + fill pinholes."""
    import cv2  # lazy
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m.astype(bool)


def mine_image_boxes(
    error: np.ndarray,
    road_mask: np.ndarray,
    cfg: Optional[AnomalyMineConfig] = None,
) -> List[MinedBox]:
    """Mine single-class anomaly boxes from one reconstruction-error map.

    Args:
        error: ``[H, W]`` per-pixel reconstruction error (>= 0).
        road_mask: ``[H, W]`` bool road-region prior.
        cfg: thresholds.

    Returns:
        List of :class:`~yolo_contrastive.geoteach.residual_labels.MinedBox`
        (``cls=0``, normalized xywh, ``score`` = median error over the component).
        Empty on a trust-gate failure (nothing anomalous, or a flooded road).
    """
    import cv2  # lazy
    from ..geoteach.residual_labels import MinedBox  # lazy (keeps roadrecon import light)
    cfg = cfg or AnomalyMineConfig()
    error = np.asarray(error, dtype=np.float32)
    h, w = error.shape
    region = np.asarray(road_mask, dtype=bool) if cfg.use_road_prior else np.ones_like(error, dtype=bool)

    vals = error[region]
    if vals.size < 16:
        return []
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-8
    z = (error - med) / mad

    anomaly = region & (z > cfg.z_thresh)
    region_px = int(region.sum())
    frac = float(anomaly.sum()) / max(1, region_px)
    if frac == 0.0 or frac > cfg.max_anomaly_area_frac:
        return []  # nothing, or a global reconstruction failure → no trustworthy boxes

    anomaly = _morph_open_close(anomaly, cfg.morph_kernel)
    if not anomaly.any():
        return []

    n, comp, stats, _ = cv2.connectedComponentsWithStats(anomaly.astype(np.uint8), connectivity=8)
    boxes: List[MinedBox] = []
    for ci in range(1, n):
        x0, y0, bw, bh, area = stats[ci]
        if area < cfg.min_box_area_px or bw < cfg.min_box_side_px or bh < cfg.min_box_side_px:
            continue
        sel = comp == ci
        boxes.append(MinedBox(
            cls=0,
            cx=(x0 + bw / 2.0) / w,
            cy=(y0 + bh / 2.0) / h,
            w=bw / w,
            h=bh / h,
            score=float(np.median(error[sel])),
        ))
    return boxes


def _load_square_rgb(path: str, imgsz: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image, resize to ``imgsz`` square. Returns (rgb_float01, bgr_uint8)."""
    import cv2  # lazy
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"unreadable image: {path}")
    bgr = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, bgr


def mine_anomaly_labels(
    reconstructor,
    images: Iterable[Tuple[str, str]],
    out_root: str,
    cfg: Optional[AnomalyMineConfig] = None,
    imgsz: Optional[int] = None,
    resume: bool = True,
    log_every: int = 200,
) -> dict:
    """Mine an anomaly-box YOLO dataset from the pool (offline, resumable).

    Args:
        reconstructor: a trained :class:`~yolo_contrastive.roadrecon.RoadReconstructor`
            (or any object with ``.error_map(imgs)`` and ``.imgsz``/``.device``).
        images: iterable of ``(image_id, path)``.
        out_root: dataset root; ``images/`` + ``labels/`` + ``data.yaml`` are written.
        cfg: mining thresholds.
        imgsz: image size (defaults to the reconstructor's).
        resume: skip images whose label file already exists.
        log_every: progress print cadence.

    Returns:
        Stats dict ``{"scanned", "with_boxes", "skipped", "boxes"}``.
    """
    import cv2  # lazy
    import torch
    from ..geoteach.plane_fit import trapezoid_mask  # lazy; pure numpy
    from ..geoteach.residual_labels import write_yolo_txt  # lazy

    cfg = cfg or AnomalyMineConfig()
    imgsz = int(imgsz or reconstructor.imgsz)
    out = Path(out_root)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    road = trapezoid_mask((imgsz, imgsz))

    stats = {"scanned": 0, "with_boxes": 0, "skipped": 0, "boxes": 0}
    for image_id, path in images:
        lbl_path = out / "labels" / f"{image_id}.txt"
        img_out = out / "images" / f"{image_id}.jpg"
        if resume and lbl_path.exists():
            stats["scanned"] += 1
            continue
        try:
            rgb, bgr = _load_square_rgb(path, imgsz)
        except FileNotFoundError:
            stats["skipped"] += 1
            continue
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        err = reconstructor.error_map(tensor)[0].detach().cpu().numpy()  # [H, W]
        boxes = mine_image_boxes(err, road, cfg)
        stats["scanned"] += 1
        if boxes:
            cv2.imwrite(str(img_out), bgr)
            write_yolo_txt(lbl_path, boxes)
            stats["with_boxes"] += 1
            stats["boxes"] += len(boxes)
        else:
            stats["skipped"] += 1
        if log_every and stats["scanned"] % log_every == 0:
            print(f"[ycl-mine] scanned={stats['scanned']} "
                  f"with_boxes={stats['with_boxes']} boxes={stats['boxes']}")

    _write_data_yaml(out, cfg.class_name)
    return stats


def _write_data_yaml(out: Path, class_name: str) -> Path:
    """Write an ultralytics ``data.yaml`` (nc=1) for the mined dataset."""
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['{class_name}']\n",
        encoding="utf-8",
    )
    return yaml_path


# ── kill-gate fidelity helpers (used by examples/12b_roadrecon_killgate.py) ────


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of normalized ``[cx, cy, w, h]`` boxes → ``[Na, Nb]``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None]
    return (inter / np.clip(area_a + area_b - inter, 1e-9, None)).astype(np.float32)


def mining_fidelity(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.3,
    small_frac: float = 0.02,
) -> dict:
    """Match mined boxes to GT for the kill-gate.

    Returns ``{recall, precision, tp, fp, fn, small_recall}`` where ``small_recall``
    is recall restricted to GT boxes with area below ``small_frac`` of the image
    (the hard tail that separates *beat* from *tie*).
    """
    pred = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    iou = box_iou_xywh(pred, gt)
    matched_gt = set()
    tp = 0
    for pi in range(len(pred)):
        if len(gt) == 0:
            break
        # match to the best UNMATCHED GT above threshold (greedy argmax over ALL GT
        # would drop a valid match when two preds share a global-argmax GT → under-recall).
        for gi in np.argsort(-iou[pi]):
            if iou[pi, gi] < iou_thr:
                break
            if int(gi) not in matched_gt:
                matched_gt.add(int(gi))
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)
    recall = tp / max(1, len(gt))
    precision = tp / max(1, len(pred))
    small = [gi for gi in range(len(gt)) if gt[gi, 2] * gt[gi, 3] < small_frac]
    small_tp = sum(1 for gi in small if gi in matched_gt)
    small_recall = small_tp / max(1, len(small))
    return {
        "recall": recall, "precision": precision,
        "tp": tp, "fp": fp, "fn": fn, "small_recall": small_recall,
        "n_small": len(small),
    }
