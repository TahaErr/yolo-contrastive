"""ScaleRealConfig — every GASP-Real threshold in one dataclass.

GASP-Real mines pairs of spatially disjoint patches whose frozen-embedder
similarity is high and whose METRIC depths differ by a bounded ratio; the pair
label is the real apparent-scale ratio ``log_r = log(Z_A / Z_B)``. All mining
gates, per-image selection budgets, training-time loss weights and probe
settings live here so the ablation grid (sim threshold {0.5, 0.6, 0.7},
lambda_inv {0, 0.5, 1}, lambda_channel {0.15, 0.3, 0.6}) is a one-field sweep.

Calibration provenance (see wf2_designs.md / wf2_ac.md):
    * sim_threshold 0.60      — calibrated on the 200-pair visual audit.
    * ratio band [1.5x, 6x]   — lower bound keeps labels above the depth-noise
                                 floor, upper bound excludes far-field junk.
    * iqr_ratio_max 0.30      — drops patches straddling depth discontinuities.
    * texture_std_min 0.04    — kills featureless asphalt/sky false matches.
    * stratified 4x4 budget   — transplants GASP's measured mutual-NN
                                 distribution-skew lesson (easy near-scale
                                 pairs dominate without stratification).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ScaleRealConfig:
    """All GASP-Real thresholds, budgets and loss weights."""

    # ── candidate grid (mining) ────────────────────────────────────────────
    #: Square candidate side as a fraction of min(H, W), one entry per level.
    grid_fractions: Tuple[float, ...] = (0.10, 0.16, 0.25)
    #: Grid stride as a fraction of the candidate side.
    grid_stride_frac: float = 0.5
    #: Horizontal margin (fraction of W) excluded on each side — bounds lens
    #: distortion at the image periphery (pinhole s ∝ 1/Z breaks off-center).
    central_margin: float = 0.15

    # ── per-patch gates (mining) ───────────────────────────────────────────
    #: Central sub-box fraction used for the patch depth statistics.
    patch_central_fraction: float = 0.5
    #: Depth-coherence gate: IQR(1/Z) / median(1/Z) must be <= this.
    iqr_ratio_max: float = 0.30
    #: Texture gate: grayscale std (image in [0, 1]) must be >= this in BOTH
    #: patches of a pair.
    texture_std_min: float = 0.04
    #: Metric-depth validity band [m] — excludes sky / far-field junk.
    z_min_m: float = 1.0
    z_max_m: float = 60.0

    # ── pair gates (mining) ────────────────────────────────────────────────
    #: DINO cosine similarity threshold (ablate {0.5, 0.6, 0.7}).
    sim_threshold: float = 0.60
    #: |log_r| band: [log 1.5, log 6] = [0.405, 1.792].
    log_ratio_min: float = math.log(1.5)
    log_ratio_max: float = math.log(6.0)
    #: Spatial disjointness: boxes expanded by this factor must NOT intersect.
    box_expand: float = 1.25

    # ── per-image stratified selection (mining) ────────────────────────────
    #: |log_r| bin edges for stratified greedy selection (4 bins).
    ratio_bin_edges: Tuple[float, ...] = (
        math.log(1.5), math.log(2.0), math.log(3.0), math.log(4.5), math.log(6.0)
    )
    #: Max pairs kept per |log_r| bin per image.
    max_pairs_per_bin: int = 4
    #: Max pairs kept per image.
    max_pairs_per_image: int = 16
    #: Images contributing fewer surviving pairs than this yield no rows.
    min_pairs_per_image: int = 2

    # ── training-time loss weights ─────────────────────────────────────────
    #: Smooth-L1 beta for the scale-equivariance term (~28% scale error).
    smooth_l1_beta: float = 0.25
    #: Within-channel weight of the SimSiam content-invariance term
    #: (ablate {0, 0.5, 1}).
    lambda_inv: float = 0.5
    #: Documented channel weight vs COCO det 1.0 (sweep {0.15, 0.3, 0.6}).
    #: NOTE: applied by the AnchoredJointTrainer via ``lambda_aux``, NOT by
    #: the channel — recorded here for the run matrix.
    lambda_channel: float = 0.3
    #: Pairs surviving augmentation are capped per batch (uniform subsample).
    max_pairs_per_batch: int = 256

    # ── heads ──────────────────────────────────────────────────────────────
    #: RoIAlign output size on the P4 tap (3x3 cells, aligned=True).
    roi_output_size: int = 3
    #: PatchDescriptor MLP widths: C4*9 -> descriptor_hidden -> descriptor_dim.
    descriptor_hidden: int = 256
    descriptor_dim: int = 128
    #: ScaleHead hidden width (descriptor_dim -> scale_hidden -> 1).
    scale_hidden: int = 64
    #: ContentProjector output dim (descriptor_dim -> descriptor_dim -> proj_dim).
    proj_dim: int = 64
    #: Expected P4 stride (asserted from tap shape at the first batch).
    expected_stride: int = 16

    # ── joint augmentation validity (training loader) ──────────────────────
    #: RandomResizedCrop area-fraction range on pool batches (no mosaic).
    rrc_scale: Tuple[float, float] = (0.5, 1.0)
    #: Crop pixel-aspect range. Kept strictly inside the assert bound below —
    #: anisotropic scaling is the one transform that would corrupt log_r.
    rrc_ratio: Tuple[float, float] = (1.0 / 1.15, 1.15)
    hflip_prob: float = 0.5
    #: Hard bound on per-axis aspect distortion, asserted from aug_theta.
    max_aspect_distortion: float = 1.2
    #: Pairs clipped by more than this area fraction by the crop are dropped.
    max_clip_frac: float = 0.20
    #: Boxes smaller than this (min side, view pixels) after aug are dropped.
    min_patch_px: float = 24.0

    # ── sentinel probe ─────────────────────────────────────────────────────
    #: Deterministic fraction of images held out for the sentinel probe set
    #: (hash of image_id) — never sampled in training.
    probe_fraction: float = 0.01
    #: Number of fixed probe pairs evaluated per epoch.
    probe_pairs: int = 512
    #: Collapse flag: prediction std below this is flagged.
    pred_std_flag: float = 0.05
    #: Row-shortcut probe: the head must beat the row-only regressor's R^2 by
    #: this epoch, else the channel is flagged.
    row_probe_deadline_epoch: int = 4

    # ── plane-fit trust gate (TERRA composability) ─────────────────────────
    #: Per-image depth-model self-consistency gate (R^2 of the affine fit of
    #: metric 1/Z against the relative cache on plane inliers). Applied only
    #: when both caches exist; graceful skip otherwise.
    plane_consistency_r2_min: float = 0.85

    # ── bookkeeping ────────────────────────────────────────────────────────
    miner_version: int = 1
    #: Pool sources excluded at mining time (panorama / fisheye captures
    #: break the pinhole s ∝ 1/Z law).
    exclude_sources: Tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not 0.0 < self.log_ratio_min < self.log_ratio_max:
            raise ValueError(
                f"need 0 < log_ratio_min < log_ratio_max, got "
                f"{self.log_ratio_min}, {self.log_ratio_max}"
            )
        if not 0.0 < self.z_min_m < self.z_max_m:
            raise ValueError(f"need 0 < z_min_m < z_max_m, got {self.z_min_m}, {self.z_max_m}")
        if not 0.0 < self.patch_central_fraction <= 1.0:
            raise ValueError(
                f"patch_central_fraction must be in (0, 1], got {self.patch_central_fraction}"
            )
        if self.box_expand < 1.0:
            raise ValueError(f"box_expand must be >= 1, got {self.box_expand}")
        if len(self.ratio_bin_edges) < 2:
            raise ValueError("ratio_bin_edges needs at least 2 edges")
        if list(self.ratio_bin_edges) != sorted(self.ratio_bin_edges):
            raise ValueError("ratio_bin_edges must be ascending")
        if self.max_aspect_distortion < max(self.rrc_ratio[1], 1.0 / self.rrc_ratio[0]):
            raise ValueError(
                "rrc_ratio must stay inside max_aspect_distortion — anisotropic "
                "scaling beyond the bound silently corrupts log_r labels"
            )
        if not 0.0 < self.probe_fraction < 1.0:
            raise ValueError(f"probe_fraction must be in (0, 1), got {self.probe_fraction}")
