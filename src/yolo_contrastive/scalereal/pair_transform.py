"""Joint image/box augmentation bridge — R5 by construction.

The training loader augments pool images with a RandomResizedCrop + hflip
affine and MUST push the mined pair boxes through the SAME transform. This
module owns both sides of that contract:

    * :func:`sample_rrc_theta`     — samples the affine ``theta`` [B, 2, 3]
      in the exact ``F.affine_grid`` convention used by
      dense/spatial_aug.py (output grid coords -> input grid coords).
    * :func:`warp_images`          — applies theta to images
      (``affine_grid`` + ``grid_sample``).
    * :func:`transform_boxes_theta`— applies the INVERSE mapping to boxes
      (input image coords -> view coords), exact for axis-aligned affines.
    * :func:`filter_transformed_pairs` — validity gating: pairs clipped
      > ``max_clip_frac`` by the crop, scaled below ``min_patch_px``, or left
      without a partner are dropped.
    * :func:`theta_anisotropy` / :func:`assert_aspect_ok` — the runtime
      guard against anisotropic per-axis scaling, the ONE transform family
      that would corrupt ``log_r``.

KEY INVARIANT (asserted by the metamorphic tests): a global crop / resize /
flip multiplies BOTH patches' apparent scales by the same factor, so
``log_r`` is unchanged by augmentation — labels are aug-invariant, validity
is not. ``log_r`` would be invalidated by anisotropic per-axis scaling only,
hence the hard aspect-distortion bound (<= 1.2x by default).

Coordinate conventions (identical to dense/spatial_aug.py):
    * theta maps OUTPUT grid coords (u', v') in [-1, 1] to INPUT grid coords:
      ``[u_in, v_in] = A @ [u', v'] + t`` with diagonal A (no rotation/shear).
    * boxes are normalized xyxy in [0, 1] of the input image; view boxes are
      normalized xyxy in [0, 1] of the output view.

NOTE on non-square images: theta lives in normalized coordinates, so the
per-axis PIXEL scale also depends on the input aspect ratio. The training
dataset therefore letterboxes images to a square canvas first
(:func:`letterbox_to_square`) — after that, theta anisotropy IS pixel
anisotropy and ``assert_aspect_ok`` is exact.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── theta sampling ────────────────────────────────────────────────────────────


def sample_rrc_theta(
    b: int,
    scale: Tuple[float, float] = (0.5, 1.0),
    ratio: Tuple[float, float] = (1.0 / 1.15, 1.15),
    hflip_prob: float = 0.5,
    content_box: Optional[Tuple[float, float, float, float]] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample B RandomResizedCrop(+hflip) affine thetas [B, 2, 3].

    Args:
        b: batch size.
        scale: crop area fraction range (relative to the content box).
        ratio: crop aspect (w/h) range — keep strictly inside the channel's
            ``max_aspect_distortion`` bound.
        hflip_prob: horizontal flip probability.
        content_box: optional normalized xyxy region the crop must stay
            inside (the letterbox content area); default full image.
        generator: optional torch.Generator for determinism.
        device: tensor device.

    Returns:
        theta [B, 2, 3] float32 in the affine_grid convention
        (diagonal scales = crop half-extents * 2, flip as a negated x scale).
    """
    if not 0.0 < scale[0] <= scale[1] <= 1.0:
        raise ValueError(f"scale must satisfy 0 < min <= max <= 1, got {scale}")
    if not 0.0 < ratio[0] <= ratio[1]:
        raise ValueError(f"ratio must satisfy 0 < min <= max, got {ratio}")
    device = device if device is not None else torch.device("cpu")
    cb = content_box if content_box is not None else (0.0, 0.0, 1.0, 1.0)
    cw = float(cb[2] - cb[0])
    ch = float(cb[3] - cb[1])
    if cw <= 0 or ch <= 0:
        raise ValueError(f"content_box must have positive extent, got {cb}")

    def _u(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, device=device).uniform_(lo, hi, generator=generator)

    area = _u(scale[0], scale[1]) * (cw * ch)
    aspect = torch.exp(_u(float(np.log(ratio[0])), float(np.log(ratio[1]))))
    box_w = torch.sqrt(area * aspect).clamp(max=cw)
    box_h = torch.sqrt(area / aspect).clamp(max=ch)
    x0 = cb[0] + _u(0.0, 1.0) * (cw - box_w)
    y0 = cb[1] + _u(0.0, 1.0) * (ch - box_h)
    cx = x0 + box_w / 2.0
    cy = y0 + box_h / 2.0
    flip = (_u(0.0, 1.0) < hflip_prob).float()
    x_sign = 1.0 - 2.0 * flip

    theta = torch.zeros(b, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0] = box_w * x_sign          # = 2 * half_w in grid units
    theta[:, 1, 1] = box_h
    theta[:, 0, 2] = 2.0 * cx - 1.0
    theta[:, 1, 2] = 2.0 * cy - 1.0
    return theta


def identity_theta(b: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """The no-aug theta (used for the deterministic sentinel probe batch)."""
    theta = torch.zeros(b, 2, 3, device=device or torch.device("cpu"), dtype=torch.float32)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return theta


# ── image side ────────────────────────────────────────────────────────────────


def warp_images(images: torch.Tensor, theta: torch.Tensor, out_size: int) -> torch.Tensor:
    """Apply theta to images via affine_grid + grid_sample (bilinear, zeros)."""
    if images.dim() != 4:
        raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
    if theta.shape != (images.shape[0], 2, 3):
        raise ValueError(
            f"theta must be [B, 2, 3] matching the batch, got {tuple(theta.shape)}"
        )
    grid = F.affine_grid(
        theta.to(images.device, torch.float32),
        size=(images.shape[0], images.shape[1], int(out_size), int(out_size)),
        align_corners=False,
    )
    img = images.float() if not images.is_floating_point() else images
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


# ── box side ──────────────────────────────────────────────────────────────────


def transform_boxes_theta(boxes_norm: torch.Tensor, theta_single: torch.Tensor) -> torch.Tensor:
    """Map boxes (input-image normalized xyxy) into view-normalized coords.

    Inverts the theta mapping per axis (theta is diagonal — RRC + flip only):
    ``u' = (u_in - t_x) / s_x``. Flip (negative ``s_x``) is handled by the
    min/max over transformed corners. The result is UNCLIPPED — values may
    leave [0, 1]; :func:`filter_transformed_pairs` does the validity gating.

    Args:
        boxes_norm: [N, 4] normalized xyxy in the input image.
        theta_single: [2, 3] for ONE image.

    Returns:
        [N, 4] view-normalized xyxy (x1 < x2, y1 < y2 restored after flip).
    """
    if boxes_norm.numel() == 0:
        return boxes_norm.reshape(0, 4).clone()
    if theta_single.shape != (2, 3):
        raise ValueError(f"theta_single must be [2, 3], got {tuple(theta_single.shape)}")
    sx = float(theta_single[0, 0])
    sy = float(theta_single[1, 1])
    tx = float(theta_single[0, 2])
    ty = float(theta_single[1, 2])
    if abs(sx) < 1e-9 or abs(sy) < 1e-9:
        raise ValueError(f"degenerate theta scales ({sx}, {sy})")

    b = boxes_norm.to(torch.float32)
    # normalized [0,1] -> grid [-1,1]
    u1, v1 = 2.0 * b[:, 0] - 1.0, 2.0 * b[:, 1] - 1.0
    u2, v2 = 2.0 * b[:, 2] - 1.0, 2.0 * b[:, 3] - 1.0
    # inverse mapping per axis
    pu1, pu2 = (u1 - tx) / sx, (u2 - tx) / sx
    pv1, pv2 = (v1 - ty) / sy, (v2 - ty) / sy
    x1 = (torch.minimum(pu1, pu2) + 1.0) / 2.0
    x2 = (torch.maximum(pu1, pu2) + 1.0) / 2.0
    y1 = (torch.minimum(pv1, pv2) + 1.0) / 2.0
    y2 = (torch.maximum(pv1, pv2) + 1.0) / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)


def _visible_fraction(boxes: torch.Tensor) -> torch.Tensor:
    """Fraction of each (view-normalized) box area inside [0, 1]^2."""
    full = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    clipped = boxes.clamp(0.0, 1.0)
    vis = (clipped[:, 2] - clipped[:, 0]).clamp_min(0) * \
          (clipped[:, 3] - clipped[:, 1]).clamp_min(0)
    return torch.where(full > 0, vis / full.clamp_min(1e-12), torch.zeros_like(full))


def filter_transformed_pairs(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    log_r: torch.Tensor,
    out_size: int,
    max_clip_frac: float = 0.20,
    min_patch_px: float = 24.0,
) -> Dict[str, torch.Tensor]:
    """Validity gating after the joint transform; pairs are dropped JOINTLY.

    A pair survives only if BOTH boxes are clipped by <= ``max_clip_frac`` of
    their area AND both clipped boxes keep a min side >= ``min_patch_px`` in
    view pixels — a box failing either gate drops its partner too (no
    partnerless patches).

    The ``log_r`` payload passes through UNCHANGED for survivors (the
    aug-invariance of the label; see module docstring).

    Returns:
        ``{"boxes_a": [m, 4] view-normalized CLIPPED xyxy, "boxes_b": ...,
           "log_r": [m], "keep": [N] bool mask}``.
    """
    if boxes_a.shape != boxes_b.shape or boxes_a.shape[0] != log_r.shape[0]:
        raise ValueError(
            f"shape mismatch: boxes_a {tuple(boxes_a.shape)}, boxes_b "
            f"{tuple(boxes_b.shape)}, log_r {tuple(log_r.shape)}"
        )
    if boxes_a.numel() == 0:
        return {"boxes_a": boxes_a.reshape(0, 4), "boxes_b": boxes_b.reshape(0, 4),
                "log_r": log_r.reshape(0), "keep": torch.zeros(0, dtype=torch.bool)}

    keep = torch.ones(boxes_a.shape[0], dtype=torch.bool, device=boxes_a.device)
    clipped = []
    for boxes in (boxes_a, boxes_b):
        vis = _visible_fraction(boxes)
        cb = boxes.clamp(0.0, 1.0)
        side_px = torch.minimum(cb[:, 2] - cb[:, 0], cb[:, 3] - cb[:, 1]) * float(out_size)
        keep &= vis >= (1.0 - max_clip_frac)
        keep &= side_px >= float(min_patch_px)
        clipped.append(cb)
    return {
        "boxes_a": clipped[0][keep],
        "boxes_b": clipped[1][keep],
        "log_r": log_r[keep],
        "keep": keep,
    }


def transform_targets(
    targets: Dict[str, np.ndarray],
    theta_single: torch.Tensor,
    out_size: int,
    max_clip_frac: float = 0.20,
    min_patch_px: float = 24.0,
) -> Dict[str, torch.Tensor]:
    """Dataset-side convenience: prepare_targets() output -> view tensors.

    Maps both boxes of every pair through the SAME theta as the image (R5 by
    construction), then applies :func:`filter_transformed_pairs`.

    Args:
        targets: ``{"boxes_a": [m,4], "boxes_b": [m,4], "log_r": [m]}`` in
            original-image normalized coords (numpy or tensor).
        theta_single: [2, 3] — the EXACT affine used to warp this image.
        out_size: view side in pixels.

    Returns:
        ``{"boxes_a", "boxes_b", "log_r", "keep"}`` as in
        :func:`filter_transformed_pairs` (view-normalized survivor boxes).
    """
    boxes_a = torch.as_tensor(np.asarray(targets["boxes_a"], dtype=np.float32))
    boxes_b = torch.as_tensor(np.asarray(targets["boxes_b"], dtype=np.float32))
    log_r = torch.as_tensor(np.asarray(targets["log_r"], dtype=np.float32))
    va = transform_boxes_theta(boxes_a, theta_single)
    vb = transform_boxes_theta(boxes_b, theta_single)
    return filter_transformed_pairs(
        va, vb, log_r, out_size,
        max_clip_frac=max_clip_frac, min_patch_px=min_patch_px,
    )


# ── the aspect guard ──────────────────────────────────────────────────────────


def theta_anisotropy(theta: torch.Tensor) -> torch.Tensor:
    """Per-sample anisotropy max(|sx|/|sy|, |sy|/|sx|) of [B, 2, 3] thetas."""
    if theta.dim() == 2:
        theta = theta.unsqueeze(0)
    sx = theta[:, 0, 0].abs().clamp_min(1e-9)
    sy = theta[:, 1, 1].abs().clamp_min(1e-9)
    return torch.maximum(sx / sy, sy / sx)


def assert_aspect_ok(theta: torch.Tensor, max_ratio: float = 1.2) -> None:
    """Hard guard: anisotropic per-axis scaling corrupts log_r labels.

    Raises ``ValueError`` if any sample's |sx|/|sy| ratio exceeds
    ``max_ratio`` (small numerical slack included). The channel calls this on
    every batch's ``aug_theta`` — if the shared pool augmentation ever
    diverges (e.g. adds mosaic or anisotropic resize), training fails loudly
    instead of learning from silently-broken labels.
    """
    aniso = theta_anisotropy(theta)
    worst = float(aniso.max())
    if worst > max_ratio * (1.0 + 1e-4):
        raise ValueError(
            f"aug_theta aspect distortion {worst:.4f} exceeds the bound {max_ratio} — "
            f"anisotropic scaling invalidates log_r labels (pair_transform.py guard). "
            f"Keep the shared pool augmentation isotropic (RRC ratio inside the bound, "
            f"no mosaic)."
        )


# ── letterbox helper (non-square inputs) ─────────────────────────────────────


def letterbox_to_square(
    image: torch.Tensor,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[float, float, float, float]]:
    """Pad [C, H, W] to a square canvas; return (padded, content_box).

    No resampling, no aspect change — pure padding, so box coords transform
    affinely and exactly. ``content_box`` is the normalized xyxy of the
    original content inside the square canvas; use
    :func:`boxes_to_padded` for boxes, and pass ``content_box`` to
    :func:`sample_rrc_theta` so crops stay on content.
    """
    if image.dim() != 3:
        raise ValueError(f"image must be [C, H, W], got {tuple(image.shape)}")
    _, h, w = image.shape
    side = max(h, w)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    padded = F.pad(
        image, (pad_x, side - w - pad_x, pad_y, side - h - pad_y), value=pad_value
    )
    content = (pad_x / side, pad_y / side, (pad_x + w) / side, (pad_y + h) / side)
    return padded, content


def boxes_to_padded(
    boxes_norm: np.ndarray,
    content_box: Tuple[float, float, float, float],
) -> np.ndarray:
    """Re-normalize original-image boxes into the padded square canvas."""
    boxes = np.asarray(boxes_norm, dtype=np.float32).reshape(-1, 4).copy()
    x0, y0, x1, y1 = content_box
    boxes[:, [0, 2]] = x0 + boxes[:, [0, 2]] * (x1 - x0)
    boxes[:, [1, 3]] = y0 + boxes[:, [1, 3]] * (y1 - y0)
    return boxes
