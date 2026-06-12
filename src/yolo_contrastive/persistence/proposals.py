"""Class-agnostic blob proposals for REVISIT persistence labeling.

Offline stage 4: per unique image, generate up to ``max_per_image`` candidate
boxes for the cross-traversal persistence matcher. Two backends:

``cheap`` (default, zero extra deps, R7-clean)
    Union of cv2-core MSER regions on the grayscale image AND on its inverse
    (both polarities: dark manholes/potholes, bright patches/markings), plus
    connected components of a 4-level-per-channel color-posterized image.

``fastsam`` (optional upgrade, local weights ONLY — never downloads)
    SAM-family segmentation via ``ultralytics.FastSAM``; used only when the
    import succeeds AND an existing local weights path is configured.

No COCO-class detector ever touches the pool (R7): MSER/CC are bottom-up
blob detectors and FastSAM is class-agnostic segmentation.

All boxes are normalized xyxy in [0, 1]; score is the region fill ratio
(component area / box area) clipped to [0, 1].
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "ProposalConfig",
    "cheap_proposals",
    "fastsam_proposals",
    "fastsam_available",
    "propose_manifest",
    "box_iou_matrix",
    "nms_boxes",
]


@dataclasses.dataclass(frozen=True)
class ProposalConfig:
    """Proposal filtering knobs (locked; see wf2 spec)."""

    min_area_frac: float = 5e-4
    max_area_frac: float = 0.08
    min_aspect: float = 0.2          # w / h
    max_aspect: float = 5.0
    nms_iou: float = 0.7
    max_per_image: int = 60
    posterize_levels: int = 4        # per-channel quantization levels
    mser_delta: int = 5
    max_label_frac: float = 0.3      # posterize labels covering more are background


# ── pure-numpy box utilities ──────────────────────────────────────────────────


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between [N, 4] and [M, 4] xyxy boxes -> [N, M]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    order = np.argsort(-scores)
    keep: List[int] = []
    iou = box_iou_matrix(boxes, boxes)
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= iou[i] > iou_thresh
        suppressed[i] = True
    return np.array(keep, dtype=np.int64)


def _filter_boxes(
    boxes: np.ndarray, areas: np.ndarray, w: int, h: int, cfg: ProposalConfig
) -> np.ndarray:
    """Normalize pixel boxes, apply area/aspect filters + fill-ratio score.

    ``boxes``: [N, 4] pixel xyxy; ``areas``: [N] component pixel areas.
    Returns [M, 5] normalized (x1, y1, x2, y2, score).
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 5))
    boxes = boxes.astype(np.float64)
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    img_area = float(w * h)
    area_frac = (bw * bh) / img_area
    aspect = np.where(bh > 0, bw / np.maximum(bh, 1e-9), np.inf)
    ok = (
        (area_frac >= cfg.min_area_frac) & (area_frac <= cfg.max_area_frac)
        & (aspect >= cfg.min_aspect) & (aspect <= cfg.max_aspect)
        & (bw >= 2) & (bh >= 2)
    )
    boxes, areas = boxes[ok], np.asarray(areas, dtype=np.float64)[ok]
    if boxes.shape[0] == 0:
        return np.zeros((0, 5))
    score = np.clip(areas / np.maximum((boxes[:, 2] - boxes[:, 0])
                                       * (boxes[:, 3] - boxes[:, 1]), 1e-9), 0.0, 1.0)
    norm = boxes / np.array([w, h, w, h], dtype=np.float64)
    norm = np.clip(norm, 0.0, 1.0)
    return np.concatenate([norm, score[:, None]], axis=1)


# ── cheap backend (cv2-core: MSER both polarities + posterized CC) ───────────


def _mser_boxes(gray: np.ndarray, cfg: ProposalConfig):
    import cv2  # lazy (E2)

    mser = cv2.MSER_create(delta=cfg.mser_delta)
    boxes, areas = [], []
    for img in (gray, 255 - gray):
        regions, _ = mser.detectRegions(img)
        for reg in regions:
            x, y, w, h = cv2.boundingRect(reg.reshape(-1, 1, 2))
            boxes.append([x, y, x + w, y + h])
            areas.append(len(reg))
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(areas)


def _posterize_cc_boxes(img: np.ndarray, cfg: ProposalConfig):
    """Connected components of a color-posterized image (both color blobs and
    grayscale blobs when the input is single-channel)."""
    import cv2  # lazy

    if img.ndim == 2:
        chans = img[:, :, None]
    else:
        chans = img
    h, w = chans.shape[:2]
    step = max(1, 256 // cfg.posterize_levels)
    q = (chans.astype(np.int64) // step)
    label = np.zeros((h, w), dtype=np.int64)
    for c in range(q.shape[2]):
        label = label * cfg.posterize_levels + q[:, :, c]

    boxes, areas = [], []
    max_count = cfg.max_label_frac * h * w
    for val, count in zip(*np.unique(label, return_counts=True)):
        if count > max_count or count < 4:
            continue  # dominant background / specks
        mask = (label == val).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            boxes.append([x, y, x + bw, y + bh])
            areas.append(area)
    return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(areas)


def cheap_proposals(
    img: np.ndarray, cfg: Optional[ProposalConfig] = None
) -> np.ndarray:
    """Cheap class-agnostic proposals on a BGR or grayscale uint8 image.

    Returns [N, 5] normalized (x1, y1, x2, y2, score), NMS'd and capped.
    """
    import cv2  # lazy (E2)

    cfg = cfg or ProposalConfig()
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]

    mb, ma = _mser_boxes(gray, cfg)
    pb, pa = _posterize_cc_boxes(img, cfg)
    boxes = np.concatenate([mb, pb], axis=0)
    areas = np.concatenate([ma, pa], axis=0) if boxes.shape[0] else np.zeros(0)

    cand = _filter_boxes(boxes, areas, w, h, cfg)
    if cand.shape[0] == 0:
        return cand
    keep = nms_boxes(cand[:, :4], cand[:, 4], cfg.nms_iou)
    return cand[keep][: cfg.max_per_image]


# ── optional FastSAM backend (local weights only) ─────────────────────────────


def fastsam_available(weights: Optional[str]) -> bool:
    """True iff ``ultralytics.FastSAM`` imports AND ``weights`` is an existing
    local file. Never triggers a download."""
    if not weights or not Path(weights).exists():
        return False
    try:
        from ultralytics import FastSAM  # noqa: F401  lazy

        return True
    except Exception:
        return False


def fastsam_proposals(
    image_path, weights: str, cfg: Optional[ProposalConfig] = None,
    imgsz: int = 1024, conf: float = 0.2,
) -> np.ndarray:
    """FastSAM everything-mode proposals (requires a LOCAL weights file).

    Returns [N, 5] normalized (x1, y1, x2, y2, score), filtered like the
    cheap backend.
    """
    from ultralytics import FastSAM  # lazy; caller guarantees availability

    cfg = cfg or ProposalConfig()
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"FastSAM weights not found at {weights!r} — pass a local file; "
            f"implicit downloads are disabled by design."
        )
    model = FastSAM(weights)
    results = model(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        h, w = r.orig_shape
        xyxy = r.boxes.xyxy.cpu().numpy()
        scr = r.boxes.conf.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) * scr
        cand = _filter_boxes(xyxy, areas, w, h, cfg)
        out.append(cand)
    cand = np.concatenate(out, axis=0) if out else np.zeros((0, 5))
    if cand.shape[0] == 0:
        return cand
    keep = nms_boxes(cand[:, :4], cand[:, 4], cfg.nms_iou)
    return cand[keep][: cfg.max_per_image]


# ── manifest-driven stage runner ──────────────────────────────────────────────


def propose_manifest(
    pairs_path,
    proposals_path,
    backend: str = "auto",
    weights: Optional[str] = None,
    cfg: Optional[ProposalConfig] = None,
    statuses: Sequence[str] = ("aligned",),
) -> int:
    """Generate proposals for every unique image of pairs in ``statuses``.

    ``backend``: "auto" (FastSAM if importable AND local weights exist, else
    cheap), "cheap", or "fastsam". Resumable: images already present in
    ``proposals_path`` are skipped. Returns the number of NEW proposal rows.
    """
    import cv2  # lazy (E2)

    from . import pair_manifest as pm

    cfg = cfg or ProposalConfig()
    if backend == "auto":
        backend = "fastsam" if fastsam_available(weights) else "cheap"
    if backend not in ("cheap", "fastsam"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "fastsam" and not fastsam_available(weights):
        raise RuntimeError(
            "backend='fastsam' requires ultralytics.FastSAM and an existing "
            "local weights file (pass weights=...)."
        )

    df = pm.read_pairs(pairs_path)
    df = df[df["status"].isin(list(statuses))]
    done = pm.proposed_image_ids(proposals_path)
    images = {}
    for _, r in df.iterrows():
        images.setdefault(str(r["img_a_id"]), r["path_a"])
        images.setdefault(str(r["img_b_id"]), r["path_b"])

    rows = []
    for image_id, path in images.items():
        if image_id in done:
            continue
        if backend == "fastsam":
            props = fastsam_proposals(path, weights, cfg)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            props = cheap_proposals(img, cfg)
        for i, (x1, y1, x2, y2, score) in enumerate(props):
            rows.append({
                "image_id": image_id, "prop_id": f"{image_id}_{i:03d}",
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "score": float(score), "backend": backend,
            })
    return pm.append_proposals(proposals_path, rows)
