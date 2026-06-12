"""Cross-traversal persistence labels for REVISIT (offline stage 5).

Two captures of the same street, months apart, share their permanent
infrastructure (potholes, manholes, bumps, markings) but not their transient
content (vehicles, pedestrians, shadows). Matching class-agnostic blob
proposals across the aligned pair therefore yields label-like supervision —
PERSISTENT / TRANSIENT / background — isomorphic to the downstream split.

Locked label rules (per side, symmetric; pure geometry in v1):

    IoU_max >= iou_persistent (0.30)                  -> PERSISTENT
    IoU_max <  iou_transient (0.10) AND the proposal
        lies fully (4 corners + center) inside the
        visible overlap region with a 2% margin       -> TRANSIENT
    iou_transient <= IoU_max < iou_persistent         -> ignore (alignment slop)
    outside / touching the overlap boundary           -> ignore

where IoU_max is the best IoU against the OTHER side's proposals warped into
this frame (4 corners through H^{+/-1}, axis-aligned bbox). The asymmetric
evidence rule means "not matched because off-frame" can never be mislabeled
as transient.

Every threshold lives in the frozen :class:`PersistenceLabelConfig` (single
source of truth for the ablation grid, the audit tooling and the train-time
rasterizer).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np

from .align import (
    h_from_row,
    points_in_quad,
    quad_from_rect,
    roi_rect,
    shrink_quad,
    warp_boxes_h,
    warp_points_h,
)
from .proposals import box_iou_matrix

__all__ = [
    "PersistenceLabelConfig",
    "LABEL_IGNORE",
    "LABEL_PERSISTENT",
    "LABEL_TRANSIENT",
    "match_proposals",
    "overlap_quad_for_side",
    "box_in_overlap",
    "label_pair",
    "label_manifest",
    "render_audit_samples",
]

#: Integer codes returned by :func:`match_proposals`.
LABEL_IGNORE = 0
LABEL_PERSISTENT = 1
LABEL_TRANSIENT = 2

#: Manifest string per integer code.
LABEL_NAMES = {LABEL_PERSISTENT: "persistent", LABEL_TRANSIENT: "transient"}


@dataclasses.dataclass(frozen=True)
class PersistenceLabelConfig:
    """Every label-factory threshold (locked; sweeps replace this object)."""

    # proposal matching (IoU bands with an ignore middle)
    iou_persistent: float = 0.30
    iou_transient: float = 0.10
    overlap_margin: float = 0.02     # transient evidence margin inside overlap
    # geometry shared with alignment
    roi_bottom_frac: float = 0.6
    # correspondence grid (Signal A)
    corr_k: int = 128                # points kept per pair per batch
    corr_min: int = 32               # below this, Signal A skips the pair
    corr_oversample: int = 4
    coord_margin: float = 0.02       # valid coords in [margin, 1-margin]^2
    # dense rasterization (Signal B, at the P3 grid)
    ignore_index: int = 255
    bg_cap_ratio: float = 3.0        # keep <= ratio * n_fg background cells
    bg_floor: int = 64               # bg budget when a view has NO fg cells
    erode_cells: int = 1             # fg border ring -> ignore
    dilate_cells: int = 1            # background exclusion zone around boxes
    # optional appearance check (zero-mean NCC), default OFF in v1
    appearance_check: bool = False


# ── geometry helpers ──────────────────────────────────────────────────────────


def overlap_quad_for_side(h_norm: np.ndarray, side: str, cfg: PersistenceLabelConfig
                          ) -> np.ndarray:
    """The other traversal's ROI warped into THIS side's frame ([4, 2] quad).

    side "a": B's ROI through H^{-1}; side "b": A's ROI through H.
    """
    quad = quad_from_rect(roi_rect(cfg.roi_bottom_frac))
    if side == "a":
        return warp_points_h(np.linalg.inv(h_norm), quad)
    if side == "b":
        return warp_points_h(h_norm, quad)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


def box_in_overlap(
    box: np.ndarray, overlap_quad: np.ndarray, cfg: PersistenceLabelConfig
) -> bool:
    """Positive-evidence test: all 4 corners + center of ``box`` inside
    overlap-quad ∩ own-ROI, both shrunk by the 2% margin."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    pts = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2],
         [(x1 + x2) / 2.0, (y1 + y2) / 2.0]]
    )
    quad = shrink_quad(overlap_quad, cfg.overlap_margin)
    rx1, ry1, rx2, ry2 = roi_rect(cfg.roi_bottom_frac)
    m = cfg.overlap_margin
    in_roi = (
        (pts[:, 0] >= rx1 + m) & (pts[:, 0] <= rx2 - m)
        & (pts[:, 1] >= ry1 + m) & (pts[:, 1] <= ry2 - m)
    )
    return bool(np.all(points_in_quad(pts, quad) & in_roi))


def _label_one_side(
    own_boxes: np.ndarray,
    other_boxes_warped: np.ndarray,
    overlap_quad: np.ndarray,
    cfg: PersistenceLabelConfig,
) -> np.ndarray:
    """Apply the locked label rules to one side. Returns int codes [N]."""
    own_boxes = np.asarray(own_boxes, dtype=np.float64).reshape(-1, 4)
    n = own_boxes.shape[0]
    labels = np.full(n, LABEL_IGNORE, dtype=np.int64)
    if n == 0:
        return labels
    if other_boxes_warped.shape[0] > 0:
        finite = np.all(np.isfinite(other_boxes_warped), axis=1)
        iou_max = box_iou_matrix(own_boxes, other_boxes_warped[finite]).max(axis=1) \
            if finite.any() else np.zeros(n)
    else:
        iou_max = np.zeros(n)

    for i in range(n):
        if iou_max[i] >= cfg.iou_persistent:
            labels[i] = LABEL_PERSISTENT
        elif iou_max[i] < cfg.iou_transient and box_in_overlap(
            own_boxes[i], overlap_quad, cfg
        ):
            labels[i] = LABEL_TRANSIENT
        # else: ignore (slop band, or outside/touching the overlap boundary)
    return labels


def match_proposals(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    h_norm: np.ndarray,
    cfg: Optional[PersistenceLabelConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric cross-traversal persistence matching.

    Args:
        boxes_a / boxes_b: [N, 4] normalized xyxy proposals per side.
        h_norm: normalized homography, x_b = H @ x_a.
        cfg: thresholds (default = locked spec values).

    Returns:
        (labels_a, labels_b): int codes per proposal —
        :data:`LABEL_PERSISTENT`, :data:`LABEL_TRANSIENT` or
        :data:`LABEL_IGNORE`.
    """
    cfg = cfg or PersistenceLabelConfig()
    h_norm = np.asarray(h_norm, dtype=np.float64)
    h_inv = np.linalg.inv(h_norm)
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    b_in_a = warp_boxes_h(h_inv, boxes_b)
    a_in_b = warp_boxes_h(h_norm, boxes_a)
    labels_a = _label_one_side(boxes_a, b_in_a, overlap_quad_for_side(h_norm, "a", cfg), cfg)
    labels_b = _label_one_side(boxes_b, a_in_b, overlap_quad_for_side(h_norm, "b", cfg), cfg)
    return labels_a, labels_b


# ── manifest-driven stage runner ──────────────────────────────────────────────


def label_pair(
    pair_row,
    props_a: np.ndarray,
    props_b: np.ndarray,
    cfg: Optional[PersistenceLabelConfig] = None,
) -> Tuple[list, int, int]:
    """Label one aligned pair. Returns (label_rows, n_persistent, n_transient).

    ``props_*``: [N, 4] normalized xyxy proposal boxes for each side.
    """
    cfg = cfg or PersistenceLabelConfig()
    h_norm = h_from_row(pair_row)
    labels_a, labels_b = match_proposals(props_a, props_b, h_norm, cfg)

    rows, n_p, n_t = [], 0, 0
    for side, boxes, labels, image_id in (
        ("a", props_a, labels_a, pair_row["img_a_id"]),
        ("b", props_b, labels_b, pair_row["img_b_id"]),
    ):
        for i, (box, lab) in enumerate(zip(np.asarray(boxes).reshape(-1, 4), labels)):
            if lab == LABEL_IGNORE:
                continue
            n_p += int(lab == LABEL_PERSISTENT)
            n_t += int(lab == LABEL_TRANSIENT)
            rows.append({
                "label_id": f"{pair_row['pair_id']}_{side}_{i:03d}",
                "pair_id": pair_row["pair_id"],
                "image_id": str(image_id),
                "side": side,
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
                "label": LABEL_NAMES[int(lab)],
            })
    return rows, n_p, n_t


def label_manifest(
    pairs_path,
    proposals_path,
    labels_path,
    cfg: Optional[PersistenceLabelConfig] = None,
) -> Dict[str, int]:
    """Run persistence labeling over every aligned pair; append label rows and
    write n_persistent/n_transient back to the pairs manifest. Resumable
    (pairs that already have labels are skipped). Returns summary counts."""
    from . import pair_manifest as pm

    cfg = cfg or PersistenceLabelConfig()
    pairs = pm.read_pairs(pairs_path)
    props = pm.read_proposals(proposals_path)
    done = pm.labeled_pair_ids(labels_path)
    by_image = (
        {k: v[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
         for k, v in props.groupby("image_id")}
        if not props.empty else {}
    )

    all_rows, updates = [], {}
    n_p_total = n_t_total = n_pairs = 0
    for _, row in pairs[pairs["align_ok"] == True].iterrows():  # noqa: E712
        if row["pair_id"] in done:
            continue
        pa = by_image.get(str(row["img_a_id"]), np.zeros((0, 4)))
        pb = by_image.get(str(row["img_b_id"]), np.zeros((0, 4)))
        rows, n_p, n_t = label_pair(row, pa, pb, cfg)
        all_rows.extend(rows)
        updates[row["pair_id"]] = {"n_persistent": int(n_p), "n_transient": int(n_t)}
        n_p_total += n_p
        n_t_total += n_t
        n_pairs += 1

    n_new = pm.append_labels(labels_path, all_rows)
    pm.update_pairs(pairs_path, updates)
    return {"pairs": n_pairs, "labels": n_new,
            "persistent": n_p_total, "transient": n_t_total}


# ── GO/NO-GO audit renderer ───────────────────────────────────────────────────


def render_audit_samples(
    pairs_path,
    labels_path,
    out_dir,
    n: int = 200,
    seed: int = 0,
    cfg: Optional[PersistenceLabelConfig] = None,
) -> int:
    """Render up to ``n`` randomly sampled labeled pairs side-by-side with
    their boxes (green = persistent, red = transient) for the >= 80%
    visual-plausibility GO gate. Lazy cv2. Returns the number of JPGs written.
    """
    import cv2  # lazy (E2)

    from pathlib import Path

    from . import pair_manifest as pm

    cfg = cfg or PersistenceLabelConfig()
    pairs = pm.read_pairs(pairs_path)
    labels = pm.read_labels(labels_path)
    if labels.empty:
        return 0
    labeled = pairs[pairs["pair_id"].isin(set(labels["pair_id"]))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labeled))[: int(n)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i in idx:
        row = labeled.iloc[int(i)]
        img_a = cv2.imread(str(row["path_a"]), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(row["path_b"]), cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            continue
        h = min(img_a.shape[0], img_b.shape[0], 512)

        def _prep(img, image_id):
            s = h / img.shape[0]
            img = cv2.resize(img, (int(round(img.shape[1] * s)), h))
            sub = labels[(labels["pair_id"] == row["pair_id"])
                         & (labels["image_id"] == str(image_id))]
            for _, lr in sub.iterrows():
                color = (0, 200, 0) if lr["label"] == "persistent" else (0, 0, 230)
                p1 = (int(lr["x1"] * img.shape[1]), int(lr["y1"] * img.shape[0]))
                p2 = (int(lr["x2"] * img.shape[1]), int(lr["y2"] * img.shape[0]))
                cv2.rectangle(img, p1, p2, color, 2)
            return img

        canvas = np.concatenate(
            [_prep(img_a, row["img_a_id"]), _prep(img_b, row["img_b_id"])], axis=1
        )
        cv2.imwrite(str(out_dir / f"audit_{row['pair_id']}.jpg"), canvas)
        written += 1
    return written
