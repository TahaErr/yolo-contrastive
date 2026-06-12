"""Residual → label factory: ordinal label maps + mined polarity boxes (TERRA Stage 0).

Converts the standardized plane residual ``z = (d - d_surf) / sigma_MAD``
(plane_fit.py) into the two supervision granularities used by TerraChannel:

    1. A dense 6-class ordinal label map:
           D2 (z < -6)            deep depression
           D1 (-6 <= z < -2.5)    depression
           F  (|z| <= 2)          flat road
           E1 (2.5 < z <= 6)      elevation
           E2 (z > 6)             strong elevation
           X                      off-road / invalid
       with an ignore band ``2 < |z| < 2.5`` (label 255), far-field
       invalidation (depth noise exceeds pothole amplitude beyond ~10-15 m)
       and an object-vs-surface gate: connected components of ``|z| > 2.5``
       whose v-extent exceeds 0.25x the road v-extent, or whose median |z|
       exceeds 15, are off-plane OBJECTS (cars, pedestrians), not surface
       anomalies → X. Speed bumps are wide but short in v, so they survive.

    2. Mined 2-class polarity boxes (0 = depression, 1 = elevation): 3x3
       morphological open/close on the per-polarity anomaly masks, connected
       components, minimum 16^2 px, score = median |z| — written as YOLO txt.

Per-image trust gates (wf2 spec, step 6): drop geometric supervision entirely
if the plane fit is untrusted (inlier ratio < 40%) or sigma_MAD exceeds the
caller-supplied pool percentile; if the anomaly area exceeds 8% of the road
(specular/puddle suspicion) keep the F/X labels but set anomaly pixels to
ignore and mine no boxes.

Every threshold lives in :class:`ResidualLabelConfig` (the ablation grid).
cv2 is imported lazily inside functions (E2); everything is numpy in/out.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .plane_fit import (
    PlaneFitConfig,
    PlaneFitResult,
    evaluate_surface,
    fit_road_plane,
    standardized_residual,
)

# ── label space ───────────────────────────────────────────────────────────────

CLS_D2, CLS_D1, CLS_F, CLS_E1, CLS_E2, CLS_X = 0, 1, 2, 3, 4, 5
NUM_CLASSES = 6
IGNORE_LABEL = 255
#: Ordinal chain order (X is off-chain).
ORDINAL_CHAIN: Tuple[int, ...] = (CLS_D2, CLS_D1, CLS_F, CLS_E1, CLS_E2)
ANOMALY_CLASSES: Tuple[int, ...] = (CLS_D2, CLS_D1, CLS_E1, CLS_E2)
CLASS_NAMES = ("D2", "D1", "F", "E1", "E2", "X")

#: Mined-box polarity classes.
BOX_DEPRESSION, BOX_ELEVATION = 0, 1
GEOBOX_CLASS_NAMES = ("depression", "elevation")


@dataclasses.dataclass
class ResidualLabelConfig:
    """Every labeling threshold in one dataclass (the ablation grid)."""

    # z-bin edges
    z_flat: float = 2.0           # |z| <= z_flat            -> F
    z_anomaly: float = 2.5        # z_flat < |z| < z_anomaly -> ignore band
    z_strong: float = 6.0         # |z| > z_strong           -> D2 / E2
    # Far-field invalidation: road pixels whose d_surf is below this
    # percentile of road disparity -> X.
    far_field_percentile: float = 35.0
    # Object-vs-surface separation
    v_extent_max_frac: float = 0.25   # component v-extent vs road v-extent
    object_median_z: float = 15.0     # median |z| above this -> object -> X
    # Box mining
    min_box_area_px: int = 256        # 16^2 px minimum component area
    min_box_side_px: int = 4
    morph_kernel: int = 3             # 3x3 open/close
    # Per-image trust gates
    max_anomaly_area_frac: float = 0.08   # of road area -> puddle suspicion
    max_sigma_mad: Optional[float] = None  # pool 95th pct, supplied by caller
    # Road-region hole filling (recovers anomaly pixels excluded from the
    # plane-inlier mask); kernel for the closing pass, in px.
    road_close_kernel: int = 5
    ignore_label: int = IGNORE_LABEL


@dataclasses.dataclass
class MinedBox:
    """One mined geometric box (normalized xywh, YOLO convention)."""

    cls: int          # BOX_DEPRESSION | BOX_ELEVATION
    cx: float
    cy: float
    w: float
    h: float
    score: float      # median |z| over the component


@dataclasses.dataclass
class LabelMapResult:
    """Dense label map + masks/diagnostics for one image."""

    labels: np.ndarray            # [H, W] uint8, classes 0..5 + 255 ignore
    road_region: np.ndarray       # [H, W] bool, hole-filled road support
    anomaly_mask: np.ndarray      # [H, W] bool, surviving anomaly pixels
    far_field_mask: np.ndarray    # [H, W] bool, invalidated far rows
    road_v_extent: int            # row extent of the road region (px)
    suppressed_components: int    # components removed by the object gate
    anomaly_area_frac: float      # anomaly pixels / road-region pixels


@dataclasses.dataclass
class GeoLabels:
    """Full per-image label-factory output (fit + dense + boxes + trust)."""

    fit: PlaneFitResult
    label_map: Optional[LabelMapResult]
    boxes: List[MinedBox]
    use_dense: bool
    use_boxes: bool
    reason: str


# ── helpers ───────────────────────────────────────────────────────────────────


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a boolean mask (anomaly pits inside the road).

    Flood-fills the background from every border pixel; whatever background
    remains unreached is a hole. cv2 imported lazily (E2).
    """
    import cv2  # lazy: optional [pretrain] extra

    m = (mask.astype(np.uint8)) * 255
    h, w = m.shape
    ff = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    # Seed from all four borders that are background.
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[y, x] == 0 and ff_mask[y + 1, x + 1] == 0:
            cv2.floodFill(ff, ff_mask, (x, y), 255)
    # Remaining zeros in ff are enclosed holes.
    holes = ff == 0
    return mask | holes


def _morph_open_close(mask: np.ndarray, ksize: int) -> np.ndarray:
    """3x3 (configurable) morphological open then close. cv2 lazy."""
    import cv2  # lazy

    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m.astype(bool)


def bin_residual(z: np.ndarray, cfg: Optional[ResidualLabelConfig] = None) -> np.ndarray:
    """Pure z-binning into the 6-class ordinal space (no masks, no gates).

    Returns a [H, W] uint8 map with classes {D2, D1, F, E1, E2} and the
    ignore band ``z_flat < |z| < z_anomaly`` set to ``ignore_label``. X is
    never produced here — off-road/invalid is a mask decision, not a z bin.
    """
    cfg = cfg or ResidualLabelConfig()
    z = np.asarray(z, dtype=np.float32)
    labels = np.full(z.shape, cfg.ignore_label, dtype=np.uint8)
    labels[np.abs(z) <= cfg.z_flat] = CLS_F
    labels[(z <= -cfg.z_anomaly) & (z >= -cfg.z_strong)] = CLS_D1
    labels[z < -cfg.z_strong] = CLS_D2
    labels[(z >= cfg.z_anomaly) & (z <= cfg.z_strong)] = CLS_E1
    labels[z > cfg.z_strong] = CLS_E2
    return labels


# ── dense label map ───────────────────────────────────────────────────────────


def compute_label_map(
    z: np.ndarray,
    road_mask: np.ndarray,
    d_surf: np.ndarray,
    cfg: Optional[ResidualLabelConfig] = None,
) -> LabelMapResult:
    """Dense 6-class ordinal label map from the standardized residual field.

    Args:
        z: [H, W] standardized residual (plane_fit.standardized_residual).
        road_mask: [H, W] bool plane-inlier mask (PlaneFitResult.inlier_mask).
        d_surf: [H, W] fitted surface values (plane_fit.evaluate_surface) —
            used for far-field invalidation.
        cfg: thresholds.

    Returns:
        :class:`LabelMapResult` — labels are X outside the (hole-filled) road
        region, far-field rows are X, the ignore band is 255, and connected
        anomaly components failing the object gate are X.
    """
    cfg = cfg or ResidualLabelConfig()
    import cv2  # lazy

    z = np.asarray(z, dtype=np.float32)
    h, w = z.shape
    road_mask = np.asarray(road_mask, dtype=bool)
    d_surf = np.asarray(d_surf, dtype=np.float64)

    # Road support: inlier mask, closed + hole-filled so anomaly pixels
    # (excluded from the plane inliers by construction) get labels.
    road_region = _morph_open_close(road_mask, cfg.road_close_kernel) | road_mask
    road_region = _fill_holes(road_region)

    rows = np.nonzero(road_region.any(axis=1))[0]
    road_v_extent = int(rows[-1] - rows[0] + 1) if rows.size else 0

    labels = np.full((h, w), CLS_X, dtype=np.uint8)

    # Far-field invalidation: road pixels whose surface disparity is below
    # the percentile of road disparity (too far -> depth noise >> pothole
    # amplitude). Computed on the PLANE value, so anomalies don't self-gate.
    far_field = np.zeros((h, w), dtype=bool)
    if road_mask.any():
        thresh = np.percentile(d_surf[road_mask], cfg.far_field_percentile)
        far_field = road_region & (d_surf < thresh)

    near_road = road_region & ~far_field

    # Bin z inside the near road region.
    binned = bin_residual(z, cfg)
    labels[near_road] = binned[near_road]

    # Object-vs-surface gate on connected anomaly components.
    anomaly = near_road & (np.abs(z) >= cfg.z_anomaly)
    suppressed = 0
    if anomaly.any() and road_v_extent > 0:
        n, comp = cv2.connectedComponents(anomaly.astype(np.uint8), connectivity=8)
        for ci in range(1, n):
            sel = comp == ci
            ys = np.nonzero(sel.any(axis=1))[0]
            v_extent = int(ys[-1] - ys[0] + 1)
            med_abs_z = float(np.median(np.abs(z[sel])))
            if v_extent > cfg.v_extent_max_frac * road_v_extent or \
                    med_abs_z > cfg.object_median_z:
                labels[sel] = CLS_X
                anomaly[sel] = False
                suppressed += 1

    road_px = int(road_region.sum())
    frac = float(anomaly.sum()) / road_px if road_px else 0.0

    return LabelMapResult(
        labels=labels,
        road_region=road_region,
        anomaly_mask=anomaly,
        far_field_mask=far_field,
        road_v_extent=road_v_extent,
        suppressed_components=suppressed,
        anomaly_area_frac=frac,
    )


# ── box mining ────────────────────────────────────────────────────────────────


def mine_boxes(
    z: np.ndarray,
    label_map: LabelMapResult,
    cfg: Optional[ResidualLabelConfig] = None,
) -> List[MinedBox]:
    """Mine 2-class polarity boxes from the gated anomaly field.

    Morphology (open/close, 3x3) per polarity, connected components,
    minimum-area and minimum-side filters; score = median |z|. Components
    already removed by the object/far-field gates never reach this stage
    (mining runs on the post-gate label map).
    """
    cfg = cfg or ResidualLabelConfig()
    import cv2  # lazy

    z = np.asarray(z, dtype=np.float32)
    h, w = z.shape
    labels = label_map.labels
    boxes: List[MinedBox] = []
    for box_cls, classes in (
        (BOX_DEPRESSION, (CLS_D2, CLS_D1)),
        (BOX_ELEVATION, (CLS_E1, CLS_E2)),
    ):
        mask = np.isin(labels, classes)
        if not mask.any():
            continue
        mask = _morph_open_close(mask, cfg.morph_kernel)
        if not mask.any():
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        for ci in range(1, n):
            x0, y0, bw, bh, area = stats[ci]
            if area < cfg.min_box_area_px:
                continue
            if bw < cfg.min_box_side_px or bh < cfg.min_box_side_px:
                continue
            sel = comp == ci
            boxes.append(MinedBox(
                cls=box_cls,
                cx=(x0 + bw / 2.0) / w,
                cy=(y0 + bh / 2.0) / h,
                w=bw / w,
                h=bh / h,
                score=float(np.median(np.abs(z[sel]))),
            ))
    return boxes


def boxes_to_yolo_lines(boxes: Sequence[MinedBox]) -> List[str]:
    """YOLO-txt lines ``"cls cx cy w h"`` (score is NOT written — YOLO format)."""
    return [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]


def write_yolo_txt(path, boxes: Sequence[MinedBox]) -> None:
    """Write mined boxes as a YOLO label txt (parent dirs created)."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(boxes_to_yolo_lines(boxes)) + ("\n" if boxes else ""),
                 encoding="utf-8")


# ── full per-image pipeline + trust gates ─────────────────────────────────────


def labels_from_inverse_depth(
    inv_depth: np.ndarray,
    plane_cfg: Optional[PlaneFitConfig] = None,
    cfg: Optional[ResidualLabelConfig] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> GeoLabels:
    """Plane fit → z → dense labels → boxes → trust gates, for one image.

    Trust-gate semantics (wf2 spec step 6):
        * untrusted plane fit (inlier ratio < 40%) or sigma_MAD above
          ``cfg.max_sigma_mad`` → no geometric supervision at all
          (``use_dense=False, use_boxes=False``; label_map is None);
        * anomaly area > ``max_anomaly_area_frac`` of the road → puddle /
          specular suspicion: F/X labels kept, anomaly pixels set to ignore,
          no boxes (``use_dense=True, use_boxes=False``);
        * otherwise full supervision.
    """
    cfg = cfg or ResidualLabelConfig()
    fit = fit_road_plane(inv_depth, cfg=plane_cfg, valid_mask=valid_mask)

    if not fit.trusted:
        return GeoLabels(fit=fit, label_map=None, boxes=[], use_dense=False,
                         use_boxes=False, reason=fit.reason)
    if cfg.max_sigma_mad is not None and fit.sigma_mad > cfg.max_sigma_mad:
        return GeoLabels(fit=fit, label_map=None, boxes=[], use_dense=False,
                         use_boxes=False,
                         reason=f"sigma_mad {fit.sigma_mad:.4g} > cap {cfg.max_sigma_mad:.4g}")

    z = standardized_residual(inv_depth, fit)
    d_surf = evaluate_surface(fit.params, z.shape)
    lm = compute_label_map(z, fit.inlier_mask, d_surf, cfg)

    if lm.anomaly_area_frac > cfg.max_anomaly_area_frac:
        # Specular/puddle suspicion: keep F/X, ignore the anomaly pixels.
        lm.labels[lm.anomaly_mask] = cfg.ignore_label
        lm.anomaly_mask[:] = False
        return GeoLabels(fit=fit, label_map=lm, boxes=[], use_dense=True,
                         use_boxes=False,
                         reason=f"anomaly_area {lm.anomaly_area_frac:.3f} > "
                                f"{cfg.max_anomaly_area_frac}")

    boxes = mine_boxes(z, lm, cfg)
    return GeoLabels(fit=fit, label_map=lm, boxes=boxes, use_dense=True,
                     use_boxes=True, reason="ok")
