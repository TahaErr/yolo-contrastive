"""Dense (per-position) NT-Xent loss with coordinate-based positive matching.

Faz 1.4b — Foundation for Dense CL (WORK_PLAN_v3 §5).

Standard NT-Xent operates on one [B, D] embedding per image. Dense CL
operates on [B, D, H, W] feature maps and computes a contrastive loss
per spatial position. Without coordinate tracking the (i,j) of view1
cannot be matched to view2; spatial_aug.py (Faz 1.4a) solves that by
emitting per-pixel original-image coordinates.

Pipeline:
    1. Subsample N_q query positions from view1's feature map (saves memory).
    2. For each query, find positive matches in view2 by coord proximity:
          threshold mode  → all k positions within `pos_radius` (multi-positive)
          nearest mode    → single closest k position (1-1)
    3. NT-Xent over [logits_pos, logits_neg]:
          neg = remaining k positions of same image + queue (if any)
    4. Mask out queries with no positive match.

Numerical:
    All loss math in fp32 (autocast disabled). Subtract-max trick on logits.

Memory:
    With H=W=80, B=8: full pairwise sim is 6400×6400×8 ≈ 327M floats — too big.
    With n_query=256, B=8 and queue K=65536: ≈ 256 × (6400 + 65536) × 8
    ≈ 147M floats peak — manageable.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


MatchMode = Literal["threshold", "nearest"]


# ─────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────


def coords_to_feature_map(
    coords: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resample a per-pixel coord map to match a feature map's spatial size.

    Args:
        coords: [B, 2, H_in, W_in] from SpatialTwoViewAugmentation.
        target_h, target_w: spatial size of the feature map (e.g. P3 = H/8).

    Returns:
        [B, 2, target_h, target_w] coords aligned to feature map cells.
    """
    if coords.dim() != 4 or coords.shape[1] != 2:
        raise ValueError(
            f"coords must be [B, 2, H, W], got {tuple(coords.shape)}"
        )
    if coords.shape[-2:] == (target_h, target_w):
        return coords
    return F.interpolate(
        coords.float(), size=(target_h, target_w),
        mode="bilinear", align_corners=False,
    )


# ─────────────────────────────────────────────────────────────────────────
# Subsampling
# ─────────────────────────────────────────────────────────────────────────


def _subsample_positions(
    features: torch.Tensor,
    coords: torch.Tensor,
    n: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample N positions per batch from [B, D, H, W] feature + coord maps.

    Returns:
        feats_sub:  [B, n, D]
        coords_sub: [B, n, 2]   (last dim: (x, y))
    """
    B, D, H, W = features.shape
    HW = H * W

    feats_flat = features.flatten(2).permute(0, 2, 1)         # [B, HW, D]
    coords_flat = coords.flatten(2).permute(0, 2, 1)          # [B, HW, 2]

    if n >= HW:
        return feats_flat, coords_flat

    # Per-sample random index selection (with replacement is OK for HW >> n;
    # we use without-replacement for cleaner signal)
    idx = torch.stack(
        [torch.randperm(HW, generator=generator, device=features.device)[:n]
         for _ in range(B)],
        dim=0,
    )  # [B, n]

    feats_sub = torch.gather(
        feats_flat, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D),
    )  # [B, n, D]
    coords_sub = torch.gather(
        coords_flat, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, 2),
    )  # [B, n, 2]
    return feats_sub, coords_sub


# ─────────────────────────────────────────────────────────────────────────
# Main loss
# ─────────────────────────────────────────────────────────────────────────


def dense_ntxent_loss(
    q_features: torch.Tensor,           # [B, D, Hq, Wq]
    k_features: torch.Tensor,           # [B, D, Hk, Wk]
    q_coords: torch.Tensor,             # [B, 2, Hq, Wq] (in feature-map size)
    k_coords: torch.Tensor,             # [B, 2, Hk, Wk] (in feature-map size)
    queue: Optional[torch.Tensor] = None,
    temperature: float = 0.2,
    n_query: int = 256,
    pos_radius: float = 0.07,
    match_mode: MatchMode = "threshold",
    generator: Optional[torch.Generator] = None,
    return_info: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute dense NT-Xent loss with coord-based positive matching.

    Args:
        q_features, k_features: L2-normalized along dim=1.
        q_coords, k_coords:     coord maps already at feature-map resolution
                                (use coords_to_feature_map() if not).
        queue:                  optional [N, D] cross-image negatives.
        temperature:            NT-Xent temperature.
        n_query:                positions to subsample per sample.
        pos_radius:             positive threshold in normalized image coords
                                (only used when match_mode="threshold").
        match_mode:             "threshold" → multi-positive (PixPro style)
                                "nearest"   → single closest (DenseCL style)
        generator:              optional RNG for deterministic subsampling.
        return_info:            if False, info dict is empty.

    Returns:
        (loss, info)
            loss: scalar fp32 tensor with grad
            info: {"matched_frac", "mean_pos_sim", "mean_neg_sim",
                   "acc_top1", "n_used"}
    """
    if q_features.dim() != 4 or k_features.dim() != 4:
        raise ValueError("q_features and k_features must be [B, D, H, W]")
    if q_features.shape[0] != k_features.shape[0]:
        raise ValueError("Batch dims of q and k must match")
    if q_features.shape[1] != k_features.shape[1]:
        raise ValueError("Feature dims of q and k must match")
    if match_mode not in ("threshold", "nearest"):
        raise ValueError(f"match_mode must be 'threshold' or 'nearest', got {match_mode!r}")
    if queue is not None and queue.dim() != 2:
        raise ValueError(f"queue must be 2-D [N, D], got shape {tuple(queue.shape)}")
    if queue is not None and queue.shape[1] != q_features.shape[1]:
        raise ValueError("queue dim must match feature dim")

    B, D = q_features.shape[0], q_features.shape[1]

    # All math in fp32 — autocast off
    device_type = "cuda" if q_features.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        q_features = q_features.float()
        k_features = k_features.float()
        q_coords = q_coords.float()
        k_coords = k_coords.float()
        queue_f = queue.float() if queue is not None else None

        # 1) Subsample queries (view1)
        q_sub, qc_sub = _subsample_positions(q_features, q_coords, n_query, generator)
        # q_sub: [B, n, D], qc_sub: [B, n, 2]

        # 2) Flatten k side completely (no subsampling on key side — use all positions)
        k_flat = k_features.flatten(2).permute(0, 2, 1)         # [B, HWk, D]
        kc_flat = k_coords.flatten(2).permute(0, 2, 1)          # [B, HWk, 2]
        HWk = k_flat.shape[1]

        # 3) Per-query coord distance to all k positions
        # qc: [B, n, 1, 2]; kc: [B, 1, HWk, 2]
        diff = qc_sub.unsqueeze(2) - kc_flat.unsqueeze(1)       # [B, n, HWk, 2]
        dist = diff.pow(2).sum(-1).sqrt()                       # [B, n, HWk]

        # 4) Build positive mask
        if match_mode == "threshold":
            pos_mask = dist <= pos_radius                       # [B, n, HWk]
        else:  # "nearest"
            min_idx = dist.argmin(dim=-1, keepdim=True)         # [B, n, 1]
            pos_mask = torch.zeros_like(dist, dtype=torch.bool)
            pos_mask.scatter_(-1, min_idx, True)
            # Optional: also enforce a max distance for nearest mode
            within = dist <= pos_radius
            pos_mask = pos_mask & within

        has_pos = pos_mask.any(dim=-1)                          # [B, n]
        n_used = int(has_pos.sum().item())

        # If no query has any positive, return zero loss with grad path
        if n_used == 0:
            zero = (q_sub.sum() * 0.0)  # keeps gradient graph
            info = {} if not return_info else {
                "matched_frac": 0.0, "mean_pos_sim": 0.0,
                "mean_neg_sim": 0.0, "acc_top1": 0.0, "n_used": 0,
            }
            return zero, info

        # 5) Compute similarities
        # In-image similarities: q_sub [B, n, D] · k_flat^T [B, D, HWk] → [B, n, HWk]
        sim_kk = torch.bmm(q_sub, k_flat.transpose(1, 2))       # [B, n, HWk]

        # Queue similarities (cross-image, all negative)
        # q flatten across batch: [B*n, D], queue: [N, D]
        if queue_f is not None and queue_f.shape[0] > 0:
            sim_q = (q_sub.reshape(B * q_sub.shape[1], D) @ queue_f.t())
            # [B*n, N] → [B, n, N]
            sim_q = sim_q.view(B, q_sub.shape[1], queue_f.shape[0])
        else:
            sim_q = q_sub.new_zeros((B, q_sub.shape[1], 0))

        # 6) Build logits per (b, q)
        # Positive logits: only those entries in sim_kk where pos_mask is True
        # Negative logits: rest of sim_kk + sim_q
        # Use log-sum-exp formulation:
        #
        #   loss = -log( sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))) )
        #        = -log_sum_exp(pos) + log_sum_exp(all)
        #
        # where neg = (sim_kk where ~pos_mask) ∪ sim_q
        T = float(temperature)
        sim_kk_t = sim_kk / T
        sim_q_t = sim_q / T

        # For numerical stability: shift by global max per (b, q)
        # Compute combined max across pos and neg for each query
        all_logits = torch.cat([sim_kk_t, sim_q_t], dim=-1)     # [B, n, HWk + N]
        max_logit = all_logits.max(dim=-1, keepdim=True).values  # [B, n, 1]

        sim_kk_shift = sim_kk_t - max_logit
        sim_q_shift = sim_q_t - max_logit.expand_as(sim_q_t)

        # log_sum_exp over positives (mask)
        # exp(shifted) for entries; sum over True entries; log
        exp_kk = torch.exp(sim_kk_shift)
        # log-sum-exp over positives only (per query)
        pos_sum = (exp_kk * pos_mask.float()).sum(dim=-1)        # [B, n]
        # log-sum-exp over everything
        denom_kk = exp_kk.sum(dim=-1)                            # [B, n]
        denom_q = torch.exp(sim_q_shift).sum(dim=-1) if sim_q_t.shape[-1] > 0 else \
                  torch.zeros_like(denom_kk)
        denom = denom_kk + denom_q                               # [B, n]

        # log(pos_sum) - log(denom), but pos_sum could be 0 when no positives
        # We've already handled global no-positive case; per-query no-positive
        # is masked out below.
        # Add tiny epsilon to avoid log(0) for masked-out queries (safe under mask)
        eps = 1e-20
        per_q_log_ratio = torch.log(pos_sum + eps) - torch.log(denom + eps)
        per_q_loss = -per_q_log_ratio                            # [B, n]

        # Mask out queries with no positive
        per_q_loss = per_q_loss * has_pos.float()
        loss = per_q_loss.sum() / max(n_used, 1)

        # ── info ────────────────────────────────────────────────────────
        info: Dict[str, float] = {}
        if return_info:
            with torch.no_grad():
                # Mean positive similarity over matched (q, k) pairs
                pos_sim_sum = (sim_kk * pos_mask.float()).sum()
                pos_count = pos_mask.float().sum()
                mean_pos = (pos_sim_sum / pos_count.clamp(min=1.0)).item() \
                           if pos_count.item() > 0 else 0.0

                # Mean negative similarity (all queue + all in-image where ~pos_mask)
                neg_sim_kk = sim_kk * (~pos_mask).float()
                neg_count_kk = (~pos_mask).float().sum()
                neg_sum = neg_sim_kk.sum()
                neg_count = neg_count_kk.clone()
                if sim_q.shape[-1] > 0:
                    # Use only valid queries' queue sims
                    neg_sum = neg_sum + (sim_q * has_pos.float().unsqueeze(-1)).sum()
                    neg_count = neg_count + has_pos.float().sum() * sim_q.shape[-1]
                mean_neg = (neg_sum / neg_count.clamp(min=1.0)).item() \
                           if neg_count.item() > 0 else 0.0

                # Top-1 accuracy: argmax over (in-image keys + queue) hits a positive
                logits_concat = torch.cat([sim_kk, sim_q], dim=-1)  # [B, n, HWk+N]
                # Build full pos mask on concat (queue is never positive)
                pos_full = torch.cat([
                    pos_mask,
                    torch.zeros(*sim_q.shape, device=pos_mask.device, dtype=torch.bool),
                ], dim=-1)
                argmax = logits_concat.argmax(dim=-1, keepdim=True)
                hit = pos_full.gather(-1, argmax).squeeze(-1)        # [B, n]
                hit = hit & has_pos
                acc_top1 = (hit.float().sum() / max(n_used, 1)).item()

                info = {
                    "matched_frac": float(n_used) / float(B * q_sub.shape[1]),
                    "mean_pos_sim": mean_pos,
                    "mean_neg_sim": mean_neg,
                    "acc_top1": acc_top1,
                    "n_used": n_used,
                }

    return loss, info
