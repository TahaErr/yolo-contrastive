"""Scale-Aware Positive Sampling (SAPS) — within-image variant.

Faz 2.1 — NOVEL CORE CONTRIBUTION (WORK_PLAN_v4 §1.2 D).

Standard multi-scale dense CL processes each FPN level INDEPENDENTLY:
P3 queries see only P3 keys (in-image + queue), never P4/P5. SAPS-within
breaks this independence by treating cross-scale positions of the SAME
image as additional NEGATIVES:

    P3 query  →  positives:  P3 key positions within radius (PixPro/DenseCL)
              →  negatives:  P3 in-image (non-positive)  ← already in dense_loss
                           + P3 queue                    ← already in dense_loss
                           + P4 in-image (ALL positions) ← NEW (cross-scale)
                           + P5 in-image (ALL positions) ← NEW (cross-scale)

Hypothesis: forcing "same image, different FPN level → must be negative"
teaches the encoder to discriminate scales explicitly. Traffic scenes
have extreme scale variation (distant pothole vs near vehicle) where
this is hypothesized to help.

Module independence:
    This file does NOT modify multi_scale_dense_loss. It re-implements
    the NT-Xent core from scratch over a multi-scale feature dict so
    cross-scale negatives can be added cleanly. Helper functions
    (`coords_to_feature_map`, `_subsample_positions`) are imported from
    dense_loss but the loss math is local.

Cost:
    Per query, additional negatives = sum over OTHER levels of HW. For
    YOLOv8 at 640×640: P3=80², P4=40², P5=20² → P3 query gets +1600+400
    = +2000 cross-scale negatives. Negligible vs queue (65K).

Ablation:
    `strict_negatives=True` filters cross-scale candidates by coord
    proximity: a P4 position is a valid negative for a P3 query only if
    their coords are NOT within pos_radius. This avoids treating "same
    object at different scale" as negative. Default False (saf SAPS).
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch

from .dense_loss import (
    coords_to_feature_map,
    _subsample_positions,
    MatchMode,
)


def saps_within_loss(
    q_features: Dict[str, torch.Tensor],          # online proj, L2-normalized
    k_features: Dict[str, torch.Tensor],          # momentum proj, L2-normalized, detached
    q_coords: torch.Tensor,                       # [B, 2, H_view, W_view]
    k_coords: torch.Tensor,                       # [B, 2, H_view, W_view]
    queues: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    weights: Optional[Dict[str, float]] = None,
    temperature: float = 0.2,
    n_query: int = 256,
    pos_radius: float = 0.07,
    match_mode: MatchMode = "threshold",
    strict_negatives: bool = False,
    generator: Optional[torch.Generator] = None,
    return_info: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Within-image SAPS dense NT-Xent loss across FPN levels.

    Args:
        q_features: dict {level: [B, D, H_l, W_l]} online encoder output.
                    Caller normalizes (dim=1).
        k_features: dict {level: [B, D, H_l, W_l]} momentum output.
                    Caller normalizes + detaches.
        q_coords:   [B, 2, H_view, W_view] view-resolution coord map.
        k_coords:   [B, 2, H_view, W_view] view-resolution coord map.
        queues:     optional dict {level: [N, D] or None} cross-image negatives.
        weights:    optional dict {level: float} loss weights, auto-normalized.
        temperature: NT-Xent temperature.
        n_query:    queries subsampled per (batch, level).
        pos_radius: positive-match coord threshold (normalized image distance).
        match_mode: "threshold" (multi-positive) or "nearest" (single).
        strict_negatives: if True, exclude cross-scale candidates whose coord
                          is within pos_radius of the query (avoids treating
                          "same object different scale" as negative).
        generator:  optional RNG for deterministic subsampling.
        return_info: if False, info dict is empty.

    Returns:
        (loss, info)
            loss: scalar fp32 tensor with grad
            info: per-level dict + "total" — same schema as
                  multi_scale_dense_loss but each level's stats include
                  "cross_scale_negs" (number of cross-scale negs per query).
    """
    if not q_features:
        raise ValueError("q_features is empty")
    if set(q_features.keys()) != set(k_features.keys()):
        raise ValueError(
            f"q_features keys {sorted(q_features.keys())} != "
            f"k_features keys {sorted(k_features.keys())}"
        )
    if q_coords.dim() != 4 or q_coords.shape[1] != 2:
        raise ValueError(f"q_coords must be [B, 2, H, W], got {tuple(q_coords.shape)}")
    if k_coords.dim() != 4 or k_coords.shape[1] != 2:
        raise ValueError(f"k_coords must be [B, 2, H, W], got {tuple(k_coords.shape)}")
    if match_mode not in ("threshold", "nearest"):
        raise ValueError(f"match_mode must be 'threshold' or 'nearest', got {match_mode!r}")

    levels = list(q_features.keys())

    # ── weights ──────────────────────────────────────────────────────────
    if weights is None:
        w = {lv: 1.0 / len(levels) for lv in levels}
    else:
        unknown = [k for k in weights if k not in q_features]
        if unknown:
            raise ValueError(f"weights references unknown levels: {unknown}")
        w_raw = {lv: float(weights.get(lv, 0.0)) for lv in levels}
        total = sum(w_raw.values())
        if total <= 0.0:
            raise ValueError(f"weights sum to non-positive value: {total}")
        w = {lv: v / total for lv, v in w_raw.items()}

    queues = queues or {}

    # ── precompute resampled coords + flat key features per level ────────
    # We need:
    #   - resampled q_coords / k_coords at each level's feature-map size
    #   - flat key features [B, HWk, D] per level (used both as own-level
    #     keys AND as cross-scale negatives for other levels)
    B, D = q_features[levels[0]].shape[0], q_features[levels[0]].shape[1]
    device_type = "cuda" if q_features[levels[0]].is_cuda else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        # Cast everything to fp32 for stable loss math
        q_feat_fp32 = {lv: q_features[lv].float() for lv in levels}
        k_feat_fp32 = {lv: k_features[lv].float() for lv in levels}

        # Per-level pre-flatten of keys + coords at feature resolution
        k_flat: Dict[str, torch.Tensor] = {}
        kc_flat: Dict[str, torch.Tensor] = {}
        for lv in levels:
            H_lv, W_lv = q_feat_fp32[lv].shape[2], q_feat_fp32[lv].shape[3]
            kc_lv = coords_to_feature_map(k_coords, H_lv, W_lv).float()
            k_flat[lv] = k_feat_fp32[lv].flatten(2).permute(0, 2, 1)        # [B, HWk_lv, D]
            kc_flat[lv] = kc_lv.flatten(2).permute(0, 2, 1)                  # [B, HWk_lv, 2]

        info: Dict = {}
        total_loss = q_coords.new_zeros((), dtype=torch.float32)
        active_levels = 0
        n_used_total = 0

        for lv in levels:
            if w[lv] == 0.0:
                info[lv] = {"loss": 0.0, "weight": 0.0, "skipped": True}
                continue

            q_feat = q_feat_fp32[lv]
            H_lv, W_lv = q_feat.shape[2], q_feat.shape[3]

            # 1) Subsample queries from this level
            qc_lv = coords_to_feature_map(q_coords, H_lv, W_lv).float()
            q_sub, qc_sub = _subsample_positions(q_feat, qc_lv, n_query, generator)
            # q_sub: [B, n, D], qc_sub: [B, n, 2]
            n = q_sub.shape[1]

            # 2) Same-level keys & positive matching (PixPro/DenseCL convention)
            k_lv = k_flat[lv]                  # [B, HWk_lv, D]
            kc_lv_flat = kc_flat[lv]           # [B, HWk_lv, 2]
            HWk_lv = k_lv.shape[1]

            # Coord distance qc_sub (B,n,2) ↔ kc_lv_flat (B,HWk,2)
            diff = qc_sub.unsqueeze(2) - kc_lv_flat.unsqueeze(1)    # [B, n, HWk]
            dist_self = diff.pow(2).sum(-1).sqrt()

            if match_mode == "threshold":
                pos_mask = dist_self <= pos_radius
            else:  # "nearest"
                min_idx = dist_self.argmin(dim=-1, keepdim=True)
                pos_mask = torch.zeros_like(dist_self, dtype=torch.bool)
                pos_mask.scatter_(-1, min_idx, True)
                pos_mask = pos_mask & (dist_self <= pos_radius)

            has_pos = pos_mask.any(dim=-1)             # [B, n]
            n_used_lv = int(has_pos.sum().item())

            if n_used_lv == 0:
                lv_loss = q_sub.sum() * 0.0
                lv_info = {
                    "loss": 0.0, "weight": w[lv],
                    "matched_frac": 0.0, "mean_pos_sim": 0.0,
                    "mean_neg_sim": 0.0, "acc_top1": 0.0, "n_used": 0,
                    "cross_scale_negs": 0,
                }
                info[lv] = lv_info
                total_loss = total_loss + w[lv] * lv_loss
                active_levels += 1
                continue

            # 3) Same-level similarities (positives + own-level negatives)
            sim_self = torch.bmm(q_sub, k_lv.transpose(1, 2))         # [B, n, HWk_lv]

            # 4) Cross-scale similarities — KEY INNOVATION OF SAPS-WITHIN
            cross_sim_list = []
            cross_count = 0
            for other_lv in levels:
                if other_lv == lv:
                    continue
                k_other = k_flat[other_lv]                  # [B, HWk_other, D]
                # sim: [B, n, HWk_other]
                sim_cross = torch.bmm(q_sub, k_other.transpose(1, 2))

                if strict_negatives:
                    # Exclude cross-scale candidates within pos_radius of query
                    kc_other = kc_flat[other_lv]            # [B, HWk_other, 2]
                    diff_cross = qc_sub.unsqueeze(2) - kc_other.unsqueeze(1)
                    dist_cross = diff_cross.pow(2).sum(-1).sqrt()
                    too_close = dist_cross <= pos_radius     # [B, n, HWk_other]
                    # Set similarities of "too close" entries to a very low
                    # value so they don't contribute to denom (effective
                    # exclusion via masking — we keep tensor shape stable
                    # by using -inf logits; safer than gather).
                    # We'll handle this via a multiplicative mask below
                    # using exp() so set NaN-safe sentinel.
                    sim_cross = sim_cross.masked_fill(too_close, float("-inf"))

                cross_sim_list.append(sim_cross)
                cross_count += sim_cross.shape[-1]

            if cross_sim_list:
                sim_cross_all = torch.cat(cross_sim_list, dim=-1)   # [B, n, sum_HWk_other]
            else:
                sim_cross_all = q_sub.new_zeros((B, n, 0))

            # 5) Queue similarities (own-level queue, all-negative)
            queue_lv = queues.get(lv)
            if queue_lv is not None and queue_lv.shape[0] > 0:
                queue_lv_fp32 = queue_lv.float()
                # [B*n, D] · [N, D].T → [B*n, N] → [B, n, N]
                sim_q = (q_sub.reshape(B * n, D) @ queue_lv_fp32.t()).view(
                    B, n, queue_lv_fp32.shape[0]
                )
            else:
                sim_q = q_sub.new_zeros((B, n, 0))

            # 6) Build logits and apply log-sum-exp NT-Xent
            T = float(temperature)
            sim_self_t = sim_self / T
            sim_cross_t = sim_cross_all / T
            sim_q_t = sim_q / T

            # Combined logits for max stabilization
            all_logits = torch.cat([sim_self_t, sim_cross_t, sim_q_t], dim=-1)
            # Note: -inf entries (from strict_negatives mask) properly handled
            # by max + exp (they contribute 0 to sum)
            max_logit = all_logits.max(dim=-1, keepdim=True).values
            # If a row is all -inf (degenerate), max is -inf — replace with 0
            # so subsequent exp doesn't NaN. has_pos masking will zero these
            # rows anyway.
            max_logit = torch.where(
                torch.isfinite(max_logit), max_logit, torch.zeros_like(max_logit)
            )

            sim_self_shift = sim_self_t - max_logit
            sim_cross_shift = sim_cross_t - max_logit.expand_as(sim_cross_t)
            sim_q_shift = sim_q_t - max_logit.expand_as(sim_q_t)

            # Positives: from same-level only (cross-scale and queue all
            # contribute to denominator only)
            exp_self = torch.exp(sim_self_shift)
            pos_sum = (exp_self * pos_mask.float()).sum(dim=-1)        # [B, n]
            denom_self = exp_self.sum(dim=-1)                          # [B, n]

            denom_cross = torch.exp(sim_cross_shift).sum(dim=-1) \
                          if sim_cross_t.shape[-1] > 0 \
                          else torch.zeros_like(denom_self)
            denom_q = torch.exp(sim_q_shift).sum(dim=-1) \
                      if sim_q_t.shape[-1] > 0 \
                      else torch.zeros_like(denom_self)

            denom = denom_self + denom_cross + denom_q

            eps = 1e-20
            per_q_log_ratio = torch.log(pos_sum + eps) - torch.log(denom + eps)
            per_q_loss = -per_q_log_ratio                              # [B, n]
            per_q_loss = per_q_loss * has_pos.float()
            lv_loss = per_q_loss.sum() / max(n_used_lv, 1)

            # ── stats ──────────────────────────────────────────────────
            lv_info: Dict = {
                "loss": float(lv_loss.detach().item()),
                "weight": w[lv],
                "n_used": n_used_lv,
                "cross_scale_negs": cross_count,
            }
            if return_info:
                with torch.no_grad():
                    pos_count = pos_mask.float().sum()
                    pos_sim_sum = (sim_self * pos_mask.float()).sum()
                    mean_pos = (pos_sim_sum / pos_count.clamp(min=1.0)).item() \
                               if pos_count.item() > 0 else 0.0

                    # neg sim: own-level non-pos + cross + queue, weighted by has_pos
                    neg_self_sum = (sim_self * (~pos_mask).float()).sum()
                    neg_self_count = (~pos_mask).float().sum()
                    neg_total = neg_self_sum
                    neg_count = neg_self_count.clone()

                    if sim_cross_all.shape[-1] > 0:
                        valid_cross = torch.isfinite(sim_cross_all)
                        neg_total = neg_total + (
                            sim_cross_all.masked_fill(~valid_cross, 0.0)
                            * has_pos.float().unsqueeze(-1)
                            * valid_cross.float()
                        ).sum()
                        neg_count = neg_count + (
                            valid_cross.float() * has_pos.float().unsqueeze(-1)
                        ).sum()

                    if sim_q.shape[-1] > 0:
                        neg_total = neg_total + (sim_q * has_pos.float().unsqueeze(-1)).sum()
                        neg_count = neg_count + has_pos.float().sum() * sim_q.shape[-1]

                    mean_neg = (neg_total / neg_count.clamp(min=1.0)).item() \
                               if neg_count.item() > 0 else 0.0

                    # Top-1 acc — argmax over (self + cross + queue), check if positive
                    logits_concat = torch.cat([sim_self, sim_cross_all, sim_q], dim=-1)
                    pos_full = torch.cat([
                        pos_mask,
                        torch.zeros(*sim_cross_all.shape, dtype=torch.bool,
                                    device=pos_mask.device),
                        torch.zeros(*sim_q.shape, dtype=torch.bool,
                                    device=pos_mask.device),
                    ], dim=-1)
                    argmax = logits_concat.argmax(dim=-1, keepdim=True)
                    hit = pos_full.gather(-1, argmax).squeeze(-1)
                    hit = hit & has_pos
                    acc = (hit.float().sum() / max(n_used_lv, 1)).item()

                    lv_info.update({
                        "matched_frac": float(n_used_lv) / float(B * n),
                        "mean_pos_sim": mean_pos,
                        "mean_neg_sim": mean_neg,
                        "acc_top1": acc,
                    })

            info[lv] = lv_info
            total_loss = total_loss + w[lv] * lv_loss
            active_levels += 1
            n_used_total += n_used_lv

        info["total"] = {
            "loss": float(total_loss.item()),
            "active_levels": active_levels,
            "n_used_total": n_used_total,
        }

    return total_loss, info


# ─────────────────────────────────────────────────────────────────────────
# Cross-image SAPS (Faz 2.2)
# ─────────────────────────────────────────────────────────────────────────


def saps_cross_loss(
    q_features: Dict[str, torch.Tensor],
    k_features: Dict[str, torch.Tensor],
    q_coords: torch.Tensor,
    k_coords: torch.Tensor,
    queue_keys: torch.Tensor,                  # [N_total, D] from combine_queues()
    queue_tags: torch.Tensor,                  # [N_total]    from combine_queues()
    level_to_id: Dict[str, int],               # {"P3":0, "P4":1, "P5":2}
    weights: Optional[Dict[str, float]] = None,
    temperature: float = 0.2,
    n_query: int = 256,
    pos_radius: float = 0.07,
    match_mode: MatchMode = "threshold",
    t_scale: float = 1.0,                      # scale-similarity bandwidth
    generator: Optional[torch.Generator] = None,
    return_info: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Cross-image SAPS dense NT-Xent loss with scale-aware queue weighting.

    Unlike standard MoCo where every queue entry is equally weighted as a
    negative, this function reweights queue negatives by FPN-level similarity:

        w(q_level, k_tag) = exp(-|q_level - k_tag| / t_scale)

    A P3 query (level=0) sees P3 queue entries (tag=0) with weight 1, P4
    entries (tag=1) with weight exp(-1/t_scale), P5 entries (tag=2) with
    weight exp(-2/t_scale). Limit cases:
        t_scale → ∞  ⟹ weights → 1 (uniform — equivalent to MoCo)
        t_scale → 0  ⟹ weights → 1 only for matching tag (level-isolated)

    The reweighting is applied to the queue's denominator contribution:
        denom_queue = Σ_k w(q_lv, tag_k) · exp(sim(q, k) / T)

    Same-level in-image negatives (PixPro-style "queries see neighbours but
    not their positive matches") still contribute with weight 1, since
    they're spatially co-located with the query and their scale is by
    definition matched.

    Args:
        q_features, k_features: dict {level: [B, D, H, W]}, normalized.
        q_coords, k_coords:     [B, 2, H_view, W_view] coord maps.
        queue_keys:             [N_total, D] flat key pool (use combine_queues()).
        queue_tags:             [N_total] long tensor of FPN level ids.
        level_to_id:            mapping from FPN level name → id (must match
                                what was passed to combine_queues()).
        weights:                per-level loss weights (auto-normalized).
        temperature:            NT-Xent τ.
        n_query:                queries subsampled per (batch, level).
        pos_radius:             positive-match coord threshold.
        match_mode:             "threshold" (multi-positive) or "nearest".
        t_scale:                scale-similarity bandwidth, must be positive.
        generator:              optional RNG for deterministic subsampling.
        return_info:            if False, info dict is empty.

    Returns:
        (loss, info)
            loss: scalar fp32 tensor with grad
            info: per-level dict + "total"; per-level includes
                  "queue_neg_count" (effective queue negatives = N_total).
    """
    if not q_features:
        raise ValueError("q_features is empty")
    if set(q_features.keys()) != set(k_features.keys()):
        raise ValueError(
            f"q_features keys {sorted(q_features.keys())} != "
            f"k_features keys {sorted(k_features.keys())}"
        )
    if q_coords.dim() != 4 or q_coords.shape[1] != 2:
        raise ValueError(f"q_coords must be [B, 2, H, W], got {tuple(q_coords.shape)}")
    if k_coords.dim() != 4 or k_coords.shape[1] != 2:
        raise ValueError(f"k_coords must be [B, 2, H, W], got {tuple(k_coords.shape)}")
    if match_mode not in ("threshold", "nearest"):
        raise ValueError(f"match_mode must be 'threshold' or 'nearest', got {match_mode!r}")
    if t_scale <= 0:
        raise ValueError(f"t_scale must be positive, got {t_scale}")
    if queue_keys.dim() != 2:
        raise ValueError(f"queue_keys must be [N, D], got {tuple(queue_keys.shape)}")
    if queue_tags.dim() != 1 or queue_tags.shape[0] != queue_keys.shape[0]:
        raise ValueError(
            f"queue_tags must be [N] matching queue_keys[0], "
            f"got tags {tuple(queue_tags.shape)} vs keys {tuple(queue_keys.shape)}"
        )
    missing_ids = [lv for lv in q_features if lv not in level_to_id]
    if missing_ids:
        raise ValueError(f"level_to_id missing levels: {missing_ids}")

    levels = list(q_features.keys())
    B, D = q_features[levels[0]].shape[0], q_features[levels[0]].shape[1]
    if queue_keys.shape[1] != D:
        raise ValueError(
            f"queue_keys dim {queue_keys.shape[1]} != feature dim {D}"
        )

    # ── weights ──────────────────────────────────────────────────────────
    if weights is None:
        w = {lv: 1.0 / len(levels) for lv in levels}
    else:
        unknown = [k for k in weights if k not in q_features]
        if unknown:
            raise ValueError(f"weights references unknown levels: {unknown}")
        w_raw = {lv: float(weights.get(lv, 0.0)) for lv in levels}
        total = sum(w_raw.values())
        if total <= 0.0:
            raise ValueError(f"weights sum to non-positive value: {total}")
        w = {lv: v / total for lv, v in w_raw.items()}

    device_type = "cuda" if q_features[levels[0]].is_cuda else "cpu"
    N_queue = queue_keys.shape[0]

    with torch.autocast(device_type=device_type, enabled=False):
        # Cast everything to fp32
        q_feat_fp32 = {lv: q_features[lv].float() for lv in levels}
        k_feat_fp32 = {lv: k_features[lv].float() for lv in levels}
        queue_keys_fp32 = queue_keys.float()
        queue_tags_long = queue_tags.long()

        info: Dict = {}
        total_loss = q_coords.new_zeros((), dtype=torch.float32)
        active_levels = 0
        n_used_total = 0

        for lv in levels:
            if w[lv] == 0.0:
                info[lv] = {"loss": 0.0, "weight": 0.0, "skipped": True}
                continue

            q_feat = q_feat_fp32[lv]
            k_feat = k_feat_fp32[lv]
            H_lv, W_lv = q_feat.shape[2], q_feat.shape[3]

            # 1) Subsample queries
            qc_lv = coords_to_feature_map(q_coords, H_lv, W_lv).float()
            kc_lv = coords_to_feature_map(k_coords, H_lv, W_lv).float()
            q_sub, qc_sub = _subsample_positions(q_feat, qc_lv, n_query, generator)
            n = q_sub.shape[1]

            # 2) Same-level keys (positives + own-level in-image negatives)
            k_flat = k_feat.flatten(2).permute(0, 2, 1)              # [B, HWk, D]
            kc_flat = kc_lv.flatten(2).permute(0, 2, 1)              # [B, HWk, 2]

            diff = qc_sub.unsqueeze(2) - kc_flat.unsqueeze(1)
            dist = diff.pow(2).sum(-1).sqrt()                        # [B, n, HWk]

            if match_mode == "threshold":
                pos_mask = dist <= pos_radius
            else:
                min_idx = dist.argmin(dim=-1, keepdim=True)
                pos_mask = torch.zeros_like(dist, dtype=torch.bool)
                pos_mask.scatter_(-1, min_idx, True)
                pos_mask = pos_mask & (dist <= pos_radius)

            has_pos = pos_mask.any(dim=-1)
            n_used_lv = int(has_pos.sum().item())

            if n_used_lv == 0:
                lv_loss = q_sub.sum() * 0.0
                info[lv] = {
                    "loss": 0.0, "weight": w[lv],
                    "matched_frac": 0.0, "mean_pos_sim": 0.0,
                    "mean_neg_sim": 0.0, "acc_top1": 0.0, "n_used": 0,
                    "queue_neg_count": N_queue,
                }
                total_loss = total_loss + w[lv] * lv_loss
                active_levels += 1
                continue

            sim_self = torch.bmm(q_sub, k_flat.transpose(1, 2))      # [B, n, HWk]

            # 3) Queue similarities (this is where SAPS-cross applies)
            if N_queue > 0:
                # [B*n, D] · [N, D].T = [B*n, N] → [B, n, N]
                sim_q = (q_sub.reshape(B * n, D) @ queue_keys_fp32.t()).view(
                    B, n, N_queue,
                )
                # Scale-similarity weight per queue entry, given query level
                q_lv_id = float(level_to_id[lv])
                # |q_lv - tag| as float, broadcasted
                level_diff = (queue_tags_long.float() - q_lv_id).abs()    # [N]
                w_per_key = torch.exp(-level_diff / float(t_scale))       # [N]
                # broadcast to [B, n, N]: outer product with ones
                w_broadcast = w_per_key.view(1, 1, N_queue).to(sim_q.device)
            else:
                sim_q = q_sub.new_zeros((B, n, 0))
                w_broadcast = q_sub.new_zeros((1, 1, 0))

            # 4) NT-Xent with weighted queue denominator
            T = float(temperature)
            sim_self_t = sim_self / T
            sim_q_t = sim_q / T

            all_logits = torch.cat([sim_self_t, sim_q_t], dim=-1)
            max_logit = all_logits.max(dim=-1, keepdim=True).values
            max_logit = torch.where(
                torch.isfinite(max_logit), max_logit, torch.zeros_like(max_logit),
            )
            sim_self_shift = sim_self_t - max_logit
            sim_q_shift = sim_q_t - max_logit.expand_as(sim_q_t)

            exp_self = torch.exp(sim_self_shift)
            pos_sum = (exp_self * pos_mask.float()).sum(dim=-1)
            denom_self = exp_self.sum(dim=-1)

            if N_queue > 0:
                # Weighted queue: each negative scaled by w(q_lv, tag_k)
                exp_q = torch.exp(sim_q_shift) * w_broadcast
                denom_q = exp_q.sum(dim=-1)
            else:
                denom_q = torch.zeros_like(denom_self)

            denom = denom_self + denom_q

            eps = 1e-20
            per_q_log_ratio = torch.log(pos_sum + eps) - torch.log(denom + eps)
            per_q_loss = -per_q_log_ratio
            per_q_loss = per_q_loss * has_pos.float()
            lv_loss = per_q_loss.sum() / max(n_used_lv, 1)

            # ── stats ──────────────────────────────────────────────────
            lv_info: Dict = {
                "loss": float(lv_loss.detach().item()),
                "weight": w[lv],
                "n_used": n_used_lv,
                "queue_neg_count": N_queue,
            }
            if return_info:
                with torch.no_grad():
                    pos_count = pos_mask.float().sum()
                    mean_pos = ((sim_self * pos_mask.float()).sum()
                                / pos_count.clamp(min=1.0)).item() \
                               if pos_count.item() > 0 else 0.0

                    neg_self_sum = (sim_self * (~pos_mask).float()).sum()
                    neg_self_count = (~pos_mask).float().sum()
                    neg_total = neg_self_sum
                    neg_count = neg_self_count.clone()

                    if sim_q.shape[-1] > 0:
                        # Weighted mean: sum(w_k * sim) / sum(w_k * 1)
                        # over valid (has_pos) queries
                        w_full = w_broadcast.expand_as(sim_q)
                        neg_total = neg_total + (
                            sim_q * w_full * has_pos.float().unsqueeze(-1)
                        ).sum()
                        neg_count = neg_count + (
                            w_full * has_pos.float().unsqueeze(-1)
                        ).sum()

                    mean_neg = (neg_total / neg_count.clamp(min=1.0)).item() \
                               if neg_count.item() > 0 else 0.0

                    # Top-1 acc — argmax over (self + queue)
                    logits_concat = torch.cat([sim_self, sim_q], dim=-1)
                    pos_full = torch.cat([
                        pos_mask,
                        torch.zeros(*sim_q.shape, dtype=torch.bool,
                                    device=pos_mask.device),
                    ], dim=-1)
                    argmax = logits_concat.argmax(dim=-1, keepdim=True)
                    hit = pos_full.gather(-1, argmax).squeeze(-1) & has_pos
                    acc = (hit.float().sum() / max(n_used_lv, 1)).item()

                    lv_info.update({
                        "matched_frac": float(n_used_lv) / float(B * n),
                        "mean_pos_sim": mean_pos,
                        "mean_neg_sim": mean_neg,
                        "acc_top1": acc,
                    })

            info[lv] = lv_info
            total_loss = total_loss + w[lv] * lv_loss
            active_levels += 1
            n_used_total += n_used_lv

        info["total"] = {
            "loss": float(total_loss.item()),
            "active_levels": active_levels,
            "n_used_total": n_used_total,
            "t_scale": float(t_scale),
        }

    return total_loss, info
