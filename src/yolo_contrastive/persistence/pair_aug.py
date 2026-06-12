"""Single-view joint image+coordinate augmentation for REVISIT pairs (R5).

Factored from ``dense/spatial_aug.py``'s affine-theta machinery: one random
resized crop + horizontal flip is expressed as a single 2x3 affine ``theta``
(the exact matrix ``F.affine_grid`` consumes), and the SAME theta transforms

    * the image                      (:func:`apply_theta_to_images`),
    * correspondence points          (:func:`transform_points`),
    * proposal boxes                 (:func:`transform_boxes`),
    * the overlap quadrilateral      (:func:`transform_quad`).

The teacher-cache spatial-misalignment bug class is therefore impossible by
construction: there is exactly one source of truth (normalized original-image
coordinates) and one transform per view. Photometric jitter is applied AFTER
geometry and touches pixels only.

Theta convention (identical to ``SpatialTwoViewAugmentation``): for output
(view) grid coords ``out`` in [-1, 1] (align_corners=False),

    in = A @ out + t,   A = diag(2*half_w*sign, 2*half_h),  t = (2c - 1)

maps into the ORIGINAL image's [-1, 1] grid space. ``transform_points`` is
the exact inverse (original [0,1] -> view [0,1]); the crop is axis-aligned,
so boxes map to boxes and quads to quads.

The dense 3-class label rasterizer (:func:`rasterize_label_map`) also lives
here: labels stay as normalized boxes/points until AFTER augmentation, then
are rasterized on the fly at the P3 grid — no cached label rasters, ever.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "ViewAugConfig",
    "make_theta",
    "sample_view_theta",
    "apply_theta_to_images",
    "transform_points",
    "transform_points_inverse",
    "transform_boxes",
    "transform_quad",
    "valid_points_mask",
    "photometric_jitter",
    "rasterize_label_map",
]


@dataclasses.dataclass(frozen=True)
class ViewAugConfig:
    """Per-view geometric + photometric augmentation knobs.

    The RRC scale floor is deliberately MILDER (0.5) than the SSL default
    (0.2): both views must retain the shared road region or Signal A starves
    (monitored by the skip-rate sentinel).
    """

    scale: Tuple[float, float] = (0.5, 1.0)
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip_prob: float = 0.5
    photometric: float = 0.2     # brightness/contrast/saturation jitter strength


# ── theta construction / sampling ─────────────────────────────────────────────


def make_theta(cx: float, cy: float, half_w: float, half_h: float,
               flip: bool) -> np.ndarray:
    """Build the 2x3 affine theta for a crop box (center + half-extents in
    normalized [0, 1] original-image coords) with optional horizontal flip."""
    sign = -1.0 if flip else 1.0
    theta = np.zeros((2, 3), dtype=np.float64)
    theta[0, 0] = 2.0 * half_w * sign
    theta[1, 1] = 2.0 * half_h
    theta[0, 2] = 2.0 * cx - 1.0
    theta[1, 2] = 2.0 * cy - 1.0
    return theta


def sample_view_theta(
    rng: np.random.Generator, cfg: Optional[ViewAugConfig] = None
) -> np.ndarray:
    """Sample one RRC+flip theta (RandomResizedCrop logic, normalized units).

    Mirrors ``SpatialTwoViewAugmentation._sample_crop_boxes`` for a single
    sample: area fraction ~ U(scale), aspect ~ exp(U(log ratio)), position
    uniform so the box fits; out-of-range boxes are clamped.
    """
    cfg = cfg or ViewAugConfig()
    area = rng.uniform(cfg.scale[0], cfg.scale[1])
    aspect = float(np.exp(rng.uniform(np.log(cfg.ratio[0]), np.log(cfg.ratio[1]))))
    # normalized box dims on a unit square (aspect distorts w vs h)
    w = min(1.0, float(np.sqrt(area * aspect)))
    h = min(1.0, float(np.sqrt(area / aspect)))
    x0 = rng.uniform(0.0, 1.0 - w) if w < 1.0 else 0.0
    y0 = rng.uniform(0.0, 1.0 - h) if h < 1.0 else 0.0
    flip = bool(rng.random() < cfg.hflip_prob)
    return make_theta(x0 + w / 2.0, y0 + h / 2.0, w / 2.0, h / 2.0, flip)


# ── applying theta ────────────────────────────────────────────────────────────


def apply_theta_to_images(
    images: torch.Tensor, theta: torch.Tensor, out_size: Tuple[int, int]
) -> torch.Tensor:
    """Sample views: ``images`` [B, C, H, W], ``theta`` [B, 2, 3] -> views
    [B, C, out_h, out_w] (bilinear, zero padding, align_corners=False)."""
    if images.dim() != 4:
        raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
    b, c = images.shape[:2]
    theta = theta.to(dtype=torch.float32, device=images.device).reshape(b, 2, 3)
    grid = F.affine_grid(theta, size=(b, c, out_size[0], out_size[1]),
                         align_corners=False)
    return F.grid_sample(
        images.float() if not images.is_floating_point() else images,
        grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    ).to(images.dtype)


def transform_points(theta: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Original-image normalized [0,1] points -> VIEW normalized [0,1] coords.

    Exact 2x3 affine inverse: in_grid = 2p - 1; out = A^{-1}(in - t);
    v = (out + 1) / 2. A is diagonal (axis-aligned crop), so the inverse is
    elementwise. Points outside the crop land outside [0, 1].
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(2, 3)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    in_g = 2.0 * pts - 1.0
    ax, ay = theta[0, 0], theta[1, 1]
    if abs(ax) < 1e-12 or abs(ay) < 1e-12:
        raise ValueError("degenerate theta: zero scale")
    out = np.empty_like(in_g)
    out[:, 0] = (in_g[:, 0] - theta[0, 2]) / ax
    out[:, 1] = (in_g[:, 1] - theta[1, 2]) / ay
    return (out + 1.0) / 2.0


def transform_points_inverse(theta: np.ndarray, pts_view: np.ndarray) -> np.ndarray:
    """VIEW normalized coords -> original-image normalized coords (forward
    application of theta; exact inverse of :func:`transform_points`)."""
    theta = np.asarray(theta, dtype=np.float64).reshape(2, 3)
    v = np.asarray(pts_view, dtype=np.float64).reshape(-1, 2)
    out = 2.0 * v - 1.0
    in_g = np.empty_like(out)
    in_g[:, 0] = theta[0, 0] * out[:, 0] + theta[0, 2]
    in_g[:, 1] = theta[1, 1] * out[:, 1] + theta[1, 2]
    return (in_g + 1.0) / 2.0


def transform_boxes(theta: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Transform [N, 4] normalized xyxy boxes into view coords (corner
    transform; the crop is axis-aligned so boxes stay boxes, flips swap
    x1/x2 which is re-sorted)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes.copy()
    c1 = transform_points(theta, boxes[:, [0, 1]])
    c2 = transform_points(theta, boxes[:, [2, 3]])
    out = np.empty_like(boxes)
    out[:, 0] = np.minimum(c1[:, 0], c2[:, 0])
    out[:, 1] = np.minimum(c1[:, 1], c2[:, 1])
    out[:, 2] = np.maximum(c1[:, 0], c2[:, 0])
    out[:, 3] = np.maximum(c1[:, 1], c2[:, 1])
    return out


def transform_quad(theta: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Transform a [4, 2] quad's vertices into view coords."""
    return transform_points(theta, np.asarray(quad).reshape(4, 2))


def valid_points_mask(pts_view: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """Bool [N]: view coords inside [margin, 1 - margin]^2 and finite —
    exactly excludes points cropped out by the view."""
    pts_view = np.asarray(pts_view, dtype=np.float64).reshape(-1, 2)
    finite = np.all(np.isfinite(pts_view), axis=1)
    inside = np.all((pts_view >= margin) & (pts_view <= 1.0 - margin), axis=1)
    return finite & inside


# ── photometric jitter (image-only, after geometry) ───────────────────────────


def photometric_jitter(
    img: torch.Tensor, strength: float, rng: np.random.Generator
) -> torch.Tensor:
    """Mild brightness/contrast/saturation jitter on a [C, H, W] or
    [B, C, H, W] float tensor in [0, 1]. Coords are untouched by design."""
    if strength <= 0:
        return img
    s = float(strength)
    fb, fc, fs = (float(rng.uniform(1.0 - s, 1.0 + s)) for _ in range(3))
    out = img * fb
    mean = out.mean(dim=(-2, -1), keepdim=True)
    out = (out - mean) * fc + mean
    if out.shape[-3] == 3:  # saturation only for RGB
        gray = out.mean(dim=-3, keepdim=True)
        out = (out - gray) * fs + gray
    return out.clamp(0.0, 1.0)


# ── dense label-map rasterization (train-time, AFTER augmentation) ────────────


def _erode(mask: np.ndarray, cells: int) -> np.ndarray:
    """Binary erosion with a (2c+1)^2 structuring element via shifting."""
    out = mask.copy()
    for _ in range(int(cells)):
        m = out
        shifted = m.copy()
        shifted[1:, :] &= m[:-1, :]
        shifted[:-1, :] &= m[1:, :]
        shifted[:, 1:] &= m[:, :-1]
        shifted[:, :-1] &= m[:, 1:]
        # edge cells lose their out-of-bounds neighbor -> eroded away
        shifted[0, :] = False
        shifted[-1, :] = False
        shifted[:, 0] = False
        shifted[:, -1] = False
        out = shifted
    return out


def rasterize_label_map(
    grid: int,
    boxes: np.ndarray,
    classes: np.ndarray,
    overlap_quad: Optional[np.ndarray],
    roi_box: Optional[np.ndarray],
    rng: np.random.Generator,
    bg_cap_ratio: float = 3.0,
    ignore_index: int = 255,
    erode_cells: int = 1,
    dilate_cells: int = 1,
    bg_floor: int = 64,
) -> np.ndarray:
    """Rasterize the 3-class persistence label map at the P3 grid.

    All inputs are in VIEW-normalized coordinates (post-augmentation; the
    caller transformed boxes/quad/ROI with the same theta as the image — R5).

    Args:
        grid: P3 grid size g (imgsz / 8); output is [g, g] int64.
        boxes: [N, 4] labeled proposal boxes (xyxy).
        classes: [N] codes — 1 = persistent, 2 = transient.
        overlap_quad: [4, 2] overlap region in view coords (None -> no
            background cells are labeled).
        roi_box: [4] own-ROI rect in view coords (None -> no background).
        rng: random generator for the background cap.
        bg_cap_ratio: keep at most ``ratio * n_fg`` background cells (rest
            ignored — anti predict-background shortcut).
        ignore_index: the ignore label (255).
        erode_cells: fg boxes lose a border ring of this width to ignore
            (box-vs-object slop).
        dilate_cells: background cells within this distance of ANY proposal
            box are ignored, not background.
        bg_floor: fixed background budget for views with ZERO foreground
            cells (``min(bg_floor, available)`` cells are kept) — clean
            overlap regions still supply negatives instead of an all-ignore
            map starving the background class.

    Label semantics: 0 background, 1 persistent, 2 transient, 255 ignore.
    """
    g = int(grid)
    labels = np.full((g, g), int(ignore_index), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    if boxes.shape[0] != classes.shape[0]:
        raise ValueError("boxes and classes length mismatch")

    centers = (np.arange(g, dtype=np.float64) + 0.5) / g
    cgx, cgy = np.meshgrid(centers, centers)          # [g, g] x and y
    cell = 1.0 / g

    def _box_mask(box: np.ndarray, pad: float = 0.0) -> np.ndarray:
        x1, y1, x2, y2 = box
        return (
            (cgx >= x1 - pad) & (cgx <= x2 + pad)
            & (cgy >= y1 - pad) & (cgy <= y2 + pad)
        )

    # 1) background candidates: overlap ∩ ROI, away from every proposal
    if overlap_quad is not None and roi_box is not None:
        from .align import points_in_quad  # pure numpy

        pts = np.stack([cgx.ravel(), cgy.ravel()], axis=1)
        bg = points_in_quad(pts, overlap_quad).reshape(g, g)
        rx1, ry1, rx2, ry2 = np.asarray(roi_box, dtype=np.float64)
        bg &= (cgx >= rx1) & (cgx <= rx2) & (cgy >= ry1) & (cgy <= ry2)
        for box in boxes:
            bg &= ~_box_mask(box, pad=dilate_cells * cell)
    else:
        bg = np.zeros((g, g), dtype=bool)

    # 2) foreground boxes: interior gets the class, the border ring -> ignore
    for box, cls in zip(boxes, classes):
        if cls not in (1, 2):
            continue
        mask = _box_mask(box)
        interior = _erode(mask, erode_cells)
        labels[mask] = int(ignore_index)
        labels[interior] = int(cls)

    # 3) capped background (after fg so fg always wins); proposal-free views
    #    keep a small fixed budget so they still supply negatives.
    bg &= labels == int(ignore_index)
    n_fg = int(np.isin(labels, (1, 2)).sum())
    cap = int(bg_cap_ratio * n_fg) if n_fg > 0 else int(bg_floor)
    bg_idx = np.flatnonzero(bg.ravel())
    if len(bg_idx) > cap:
        bg_idx = rng.permutation(bg_idx)[:cap]
    flat = labels.ravel()
    flat[bg_idx] = 0
    return labels
