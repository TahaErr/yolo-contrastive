"""Spatial-aware two-view augmentation with coordinate tracking.

Faz 1.4a — Foundation for Dense CL (WORK_PLAN_v3 §5).

Standard contrastive aug pipelines are pixel-level: a flat [C, H, W] tensor
goes in, a transformed [C, H, W] comes out, and any spatial bookkeeping
(where in the original image did this pixel come from?) is lost. Dense CL
needs that information — without it, the (i, j) position of view1 cannot
be matched to its corresponding (i', j') in view2.

This module produces, for each input image:
    view1, view2: [B, C, H_out, W_out]   augmented crops
    coords1, coords2: [B, 2, H_out, W_out]
        Channel 0 = original-image x in [0, 1]
        Channel 1 = original-image y in [0, 1]

The loss module (Faz 1.4b — dense_loss.py) consumes these coords to find
correspondences.

Scope (intentionally narrow):
    Geometric only: random resized crop + horizontal flip. Photometric
    augmentations (color jitter, blur, grayscale, etc.) are pixel-level
    and don't affect coords; they are applied separately, either before
    this module or as a post-step on view1/view2.

Implementation:
    Fully batch-vectorized via a single affine grid + grid_sample call.
    No per-sample Python loop. Coords are computed by sampling an
    identity-coordinate grid through the same transform as the image.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F


class TwoView(NamedTuple):
    """Output of SpatialTwoViewAugmentation.

    Each tensor has batch dim B. Image tensors carry the input dtype;
    coord tensors are always float32.

    view1, view2:        [B, C, H_out, W_out]
    coords1, coords2:    [B, 2, H_out, W_out] in original-image normalized
                         coordinates [0, 1]. Channel 0 = x, channel 1 = y.
    """
    view1: torch.Tensor
    view2: torch.Tensor
    coords1: torch.Tensor
    coords2: torch.Tensor


class SpatialTwoViewAugmentation:
    """Two-view geometric augmentation with per-pixel coordinate tracking.

    Args:
        out_size: (H_out, W_out) of each output view.
        scale: (min, max) area fraction for random resized crop.
        ratio: (min, max) aspect ratio for random resized crop.
        hflip_prob: probability of horizontal flip per view.
        seed: optional integer seed for deterministic sampling (testing).

    Usage:
        aug = SpatialTwoViewAugmentation(out_size=(640, 640))
        out = aug(images)         # images: [B, 3, H, W]
        # out.view1, out.coords1, out.view2, out.coords2

    Notes:
        - Each view is sampled INDEPENDENTLY (different crop & flip per view).
        - Both views per sample share the same input image; correspondence
          is encoded entirely in coords.
        - Sampling parameters can be regenerated each forward (default) or
          reused via seed (testing/debugging).
    """

    def __init__(
        self,
        out_size: Tuple[int, int] = (640, 640),
        scale: Tuple[float, float] = (0.2, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        hflip_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if len(out_size) != 2 or out_size[0] <= 0 or out_size[1] <= 0:
            raise ValueError(f"out_size must be (H, W) positive, got {out_size}")
        if not 0.0 < scale[0] <= scale[1] <= 1.0:
            raise ValueError(f"scale must satisfy 0 < min <= max <= 1, got {scale}")
        if not 0.0 < ratio[0] <= ratio[1]:
            raise ValueError(f"ratio must satisfy 0 < min <= max, got {ratio}")
        if not 0.0 <= hflip_prob <= 1.0:
            raise ValueError(f"hflip_prob must be in [0, 1], got {hflip_prob}")

        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.hflip_prob = float(hflip_prob)
        self._seed = seed
        self._gen: Optional[torch.Generator] = None  # initialized lazily on device

    # ── public API ───────────────────────────────────────────────────────

    def __call__(self, images: torch.Tensor) -> TwoView:
        if images.dim() != 4:
            raise ValueError(f"images must be [B, C, H, W], got shape {tuple(images.shape)}")
        view1, coords1 = self._make_view(images)
        view2, coords2 = self._make_view(images)
        return TwoView(view1, view2, coords1, coords2)

    # ── single-view sampling ─────────────────────────────────────────────

    def _make_view(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one augmented view + its coord map. Fully batch-vectorized."""
        B, _, H_in, W_in = images.shape
        device = images.device
        gen = self._get_gen(device)

        # 1) Sample a crop box per sample in normalized coords [0, 1].
        crop = self._sample_crop_boxes(B, H_in, W_in, device, gen)
        # crop: dict with cx, cy, half_w, half_h, all [B] in [0, 1]
        cx, cy, hw, hh = crop["cx"], crop["cy"], crop["half_w"], crop["half_h"]

        # 2) Sample flip per sample.
        flip = (torch.rand(B, device=device, generator=gen) < self.hflip_prob).float()
        # flip ∈ {0, 1} per sample → multiplier ∈ {+1, -1} for x scale
        x_sign = 1.0 - 2.0 * flip  # [B]

        # 3) Build affine matrix per sample.
        # grid_sample uses [-1, 1] coord space. We want to sample the crop
        # box [(cx-hw, cy-hh), (cx+hw, cy+hh)] (in [0,1] of input) into the
        # output grid. Convert center & half-extent from [0,1] to [-1,1]:
        #     center_grid = 2*center - 1
        #     halfext_grid = 2*halfext (since [0,1] range → [-1,1] range)
        cx_g = 2.0 * cx - 1.0       # [B]
        cy_g = 2.0 * cy - 1.0
        hw_g = 2.0 * hw             # [B]
        hh_g = 2.0 * hh

        # Affine matrix theta [B, 2, 3] for affine_grid:
        #   [u, v]_in = theta @ [u', v', 1]_out, where u', v' ∈ [-1, 1] over output.
        # Diagonal scales = (hw_g * x_sign, hh_g); translation = (cx_g, cy_g).
        theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0] = hw_g * x_sign
        theta[:, 1, 1] = hh_g
        theta[:, 0, 2] = cx_g
        theta[:, 1, 2] = cy_g

        # 4) Sample image.
        grid = F.affine_grid(
            theta, size=(B, images.shape[1], self.out_h, self.out_w),
            align_corners=False,
        )
        view = F.grid_sample(
            images.float() if not images.is_floating_point() else images,
            grid, mode="bilinear", padding_mode="zeros", align_corners=False,
        ).to(images.dtype)

        # 5) Compute coords by sampling an identity grid through the SAME
        # transform. Identity grid encodes (x_in, y_in) ∈ [0, 1] at each
        # input pixel; sampling it tells us "for output pixel (i, j),
        # what was the input coord?".
        coords = self._sample_coord_map(B, H_in, W_in, theta, device)
        return view, coords

    # ── crop sampling ────────────────────────────────────────────────────

    def _sample_crop_boxes(
        self,
        B: int,
        H_in: int,
        W_in: int,
        device: torch.device,
        gen: torch.Generator,
    ) -> dict:
        """Sample B crop boxes (RandomResizedCrop logic), batched.

        Returns center & half-extents in normalized [0, 1] input-image coords.

        We use a vectorized batched version of torchvision.transforms.RandomResizedCrop
        sampling logic. No fallback retry loop; if a sampled box is invalid
        for that sample, we clamp it to the largest valid box at the same
        center (rare with default scale/ratio).
        """
        # Areas relative to input area
        log_ratio = torch.log(torch.tensor(self.ratio, device=device))
        scale_min, scale_max = self.scale

        target_area = (
            torch.empty(B, device=device).uniform_(scale_min, scale_max, generator=gen)
            * (H_in * W_in)
        )
        aspect = torch.exp(
            torch.empty(B, device=device).uniform_(
                log_ratio[0].item(), log_ratio[1].item(), generator=gen
            )
        )
        # box w, h in pixels
        box_w = torch.sqrt(target_area * aspect)
        box_h = torch.sqrt(target_area / aspect)

        # Clamp to image dimensions (rare overflow when scale_max=1.0 + extreme ratio)
        box_w = box_w.clamp(min=1.0, max=float(W_in))
        box_h = box_h.clamp(min=1.0, max=float(H_in))

        # Sample top-left so the box fits
        max_x = (W_in - box_w).clamp(min=0.0)
        max_y = (H_in - box_h).clamp(min=0.0)
        x0 = torch.empty(B, device=device).uniform_(0.0, 1.0, generator=gen) * max_x
        y0 = torch.empty(B, device=device).uniform_(0.0, 1.0, generator=gen) * max_y

        # Convert to normalized center + half-extents in [0, 1]
        cx = (x0 + box_w / 2.0) / W_in
        cy = (y0 + box_h / 2.0) / H_in
        half_w = (box_w / 2.0) / W_in
        half_h = (box_h / 2.0) / H_in
        return {"cx": cx, "cy": cy, "half_w": half_w, "half_h": half_h}

    # ── coord map ────────────────────────────────────────────────────────

    def _sample_coord_map(
        self,
        B: int,
        H_in: int,
        W_in: int,
        theta: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a coord map for each output pixel → input [0, 1] coords.

        We do this analytically (no grid_sample on a coord image). For each
        output pixel (i, j) ∈ [0, H_out)×[0, W_out), we have:
            grid (normalized) = ((j+0.5)/W_out * 2 - 1, (i+0.5)/H_out * 2 - 1)
        Apply theta:
            input_grid = theta[:, :, :2] @ grid + theta[:, :, 2]
        Convert input_grid (in [-1, 1]) back to [0, 1]:
            input_norm = (input_grid + 1) / 2
        """
        # Output grid in [-1, 1] (align_corners=False semantics)
        ys = (torch.arange(self.out_h, device=device, dtype=torch.float32) + 0.5)
        xs = (torch.arange(self.out_w, device=device, dtype=torch.float32) + 0.5)
        ys = ys / self.out_h * 2.0 - 1.0
        xs = xs / self.out_w * 2.0 - 1.0
        # Mesh [H_out, W_out, 2] (last dim = (x, y))
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        out_grid = torch.stack([gx, gy], dim=-1)  # [H_out, W_out, 2]
        out_grid = out_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H_out, W_out, 2]

        # Apply per-sample theta
        # theta[:, :, :2] is [B, 2, 2]; theta[:, :, 2] is [B, 2]
        A = theta[:, :, :2]  # [B, 2, 2]
        t = theta[:, :, 2]   # [B, 2]
        # input_grid[b, i, j, :] = A[b] @ out_grid[b, i, j, :] + t[b]
        in_grid = torch.einsum("bxy,bijy->bijx", A, out_grid) + t.view(B, 1, 1, 2)

        # Map [-1, 1] → [0, 1]
        in_norm = (in_grid + 1.0) / 2.0  # [B, H_out, W_out, 2] (x, y)
        # Permute to [B, 2, H_out, W_out] with channel 0 = x, channel 1 = y
        coords = in_norm.permute(0, 3, 1, 2).contiguous()
        return coords

    # ── rng ──────────────────────────────────────────────────────────────

    def _get_gen(self, device: torch.device) -> torch.Generator:
        if self._gen is not None and self._gen.device == device:
            return self._gen
        gen = torch.Generator(device=device)
        if self._seed is not None:
            gen.manual_seed(self._seed)
        else:
            # Seed from default RNG so behaviour follows global torch.manual_seed
            gen.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        self._gen = gen
        return gen

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SpatialTwoViewAugmentation(out_size=({self.out_h}, {self.out_w}), "
            f"scale={self.scale}, ratio={self.ratio}, hflip_prob={self.hflip_prob})"
        )
