"""Multi-scale dense contrastive loss across FPN levels.

Faz 1.5 — Foundation for Dense + Multi-scale CL (WORK_PLAN_v3 §5).

Wraps dense_ntxent_loss (Faz 1.4b) over multiple FPN levels (P3/P4/P5).
Each level computes its own dense CL with its own queue, and the final
loss is a weighted sum across levels.

Design choices:
    - Caller manages queues. This module does NOT update them. The trainer
      enqueues new keys after loss is computed (loss before queue update).
    - Coord resampling is done internally per level (each level has a
      different feature-map size). Caller passes view-resolution coords.
    - No projection head here — caller projects features before calling
      this loss. (Projection head lives in heads.py — Faz 1.6.)
    - Default weights: equal across levels (1/3 each for 3 levels).
    - Ablation friendly: pass weights={"P3": 1.0} for P3-only;
      pass queues={"P3": q3, "P4": None, "P5": None} for selective queues.

Returns:
    loss: scalar
    info: per-level dict + a "total" entry with aggregate stats.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .dense_loss import (
    coords_to_feature_map,
    dense_ntxent_loss,
    MatchMode,
)


def multi_scale_dense_loss(
    q_features: Dict[str, torch.Tensor],
    k_features: Dict[str, torch.Tensor],
    q_coords: torch.Tensor,
    k_coords: torch.Tensor,
    queues: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    weights: Optional[Dict[str, float]] = None,
    temperature: float = 0.2,
    n_query: int = 256,
    pos_radius: float = 0.07,
    match_mode: MatchMode = "threshold",
    generator: Optional[torch.Generator] = None,
    return_info: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Compute weighted-sum dense NT-Xent across FPN levels.

    Args:
        q_features: dict {level_name: [B, D, H_l, W_l]} from online encoder.
                    L2-normalized along dim=1 by caller.
        k_features: dict {level_name: [B, D, H_l, W_l]} from momentum encoder.
        q_coords:   [B, 2, H_view, W_view] — original-image coords (any res,
                    will be resampled to each level's feature-map size).
        k_coords:   [B, 2, H_view, W_view] — same convention as q_coords.
        queues:     optional dict {level_name: [N, D] or None}. Missing
                    levels or None values mean "no queue for that level".
        weights:    optional dict {level_name: float}. Default: equal weight
                    1/L per level where L = len(q_features). Auto-normalized
                    if doesn't sum to 1 (caller can supply unnormalized
                    floats and they'll be rescaled).
        temperature, n_query, pos_radius, match_mode, generator, return_info:
                    forwarded to dense_ntxent_loss for each level.

    Returns:
        (loss, info)
            loss: scalar fp32 tensor with grad
            info: {
                level_name: {dense_loss info dict for that level} | {"loss": ...},
                ...,
                "total": {
                    "loss": float,
                    "weighted_sum": float,
                    "active_levels": int,
                    "n_used_total": int,
                }
            }

    Raises:
        ValueError: q/k feature dict keys don't match; weights reference
                    a missing level; coords have wrong shape.
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

    levels = list(q_features.keys())

    # ── weights ──────────────────────────────────────────────────────────
    if weights is None:
        w = {lv: 1.0 / len(levels) for lv in levels}
    else:
        unknown = [k for k in weights if k not in q_features]
        if unknown:
            raise ValueError(f"weights references unknown levels: {unknown}")
        # Use only levels present in q_features; missing → weight 0
        w_raw = {lv: float(weights.get(lv, 0.0)) for lv in levels}
        total = sum(w_raw.values())
        if total <= 0.0:
            raise ValueError(f"weights sum to non-positive value: {total}")
        w = {lv: v / total for lv, v in w_raw.items()}

    # ── queues ───────────────────────────────────────────────────────────
    queues = queues or {}

    # ── per-level dense loss ─────────────────────────────────────────────
    info: Dict = {}
    total_loss = q_coords.new_zeros((), dtype=torch.float32)
    active_levels = 0
    n_used_total = 0

    for lv in levels:
        if w[lv] == 0.0:
            info[lv] = {"loss": 0.0, "weight": 0.0, "skipped": True}
            continue

        q_feat = q_features[lv]
        k_feat = k_features[lv]
        if q_feat.shape[2:] != k_feat.shape[2:]:
            raise ValueError(
                f"Level {lv}: q feature spatial size {tuple(q_feat.shape[2:])} "
                f"!= k feature spatial size {tuple(k_feat.shape[2:])}"
            )

        H_lv, W_lv = q_feat.shape[2], q_feat.shape[3]
        qc_lv = coords_to_feature_map(q_coords, H_lv, W_lv)
        kc_lv = coords_to_feature_map(k_coords, H_lv, W_lv)

        lv_loss, lv_info = dense_ntxent_loss(
            q_features=q_feat,
            k_features=k_feat,
            q_coords=qc_lv,
            k_coords=kc_lv,
            queue=queues.get(lv),
            temperature=temperature,
            n_query=n_query,
            pos_radius=pos_radius,
            match_mode=match_mode,
            generator=generator,
            return_info=return_info,
        )
        total_loss = total_loss + w[lv] * lv_loss
        active_levels += 1
        n_used_total += lv_info.get("n_used", 0) if lv_info else 0

        info[lv] = {**lv_info, "loss": lv_loss.item(), "weight": w[lv]} \
                   if return_info else {"loss": lv_loss.item(), "weight": w[lv]}

    info["total"] = {
        "loss": total_loss.item(),
        "active_levels": active_levels,
        "n_used_total": n_used_total,
    }
    return total_loss, info
