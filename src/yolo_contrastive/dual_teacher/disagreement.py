"""DisagreementWeighter — per-position teacher-disagreement sample weighting.

Faz 5.3 — DT-SAPS dual-teacher framework (WORK_PLAN_v9 §10.29, Risk 20).

Two teachers (COCO YOLOv8x + SSL momentum encoder) rarely agree everywhere.
Where they disagree, there is a choice: amplify the distillation signal
(treat disagreement as "hard regions worth learning") or damp it (treat
disagreement as noise, trust consensus). The literature is split — UEKD-style
work amplifies, PAD-style work shows hard-mining can hurt distillation.

This module does NOT commit to either side. It computes a per-position weight

    w = clamp( exp(alpha_d * d), max=clamp_max )

where d = 1 - cosine_similarity(f_a, f_b) is the per-position disagreement.
The SIGN of alpha_d decides the regime:
    alpha_d > 0  → amplify disagreement (hard-region mining; UEKD direction)
    alpha_d = 0  → uniform weighting (classic distillation)
    alpha_d < 0  → amplify agreement (consensus trust; PAD / CoMAD direction)
The Faz 5.3 ablation sweeps alpha_d across negative AND positive values, so
the data — not an a-priori assumption — decides the regime.

Two original extensions beyond the literature (paper contribution):
  1. LEARNABLE alpha_d. Instead of a fixed hyperparameter, alpha_d can be an
     nn.Parameter the model tunes during training. The learned trajectory
     (alpha_d vs epoch) is itself a result: convergence to a negative value
     empirically endorses consensus, a positive value endorses hard-mining.
  2. PER-SCALE alpha_d. One alpha_d per FPN level (P3/P4/P5) rather than a
     single shared scalar. Disagreement may play a different role for small
     distant objects (P3) vs large near surfaces (P5) — making the weighter
     scale-aware, consistent with the SAPS core of the framework.

The 2x2 ablation {fixed, learnable} x {shared, per_scale} positions the
literature's setting (fixed-shared) against this paper's extension
(learnable-per_scale).

Cosine (not L2) is the disagreement metric: the two teachers live in very
different feature-magnitude regimes, and feature-distillation literature
consistently favours cosine — it captures semantic/directional disagreement,
not a scale artefact.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_disagreement(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-position cosine disagreement between two feature maps.

    Args:
        feat_a, feat_b: ``[B, C, H, W]`` feature maps (same shape).
        eps: normalization epsilon.

    Returns:
        ``[B, H, W]`` disagreement in ``[0, 2]`` — 0 identical direction,
        1 orthogonal, 2 opposite.
    """
    if feat_a.shape != feat_b.shape:
        raise ValueError(
            f"feat_a {tuple(feat_a.shape)} and feat_b {tuple(feat_b.shape)} "
            f"must have the same shape"
        )
    if feat_a.dim() != 4:
        raise ValueError(f"features must be [B, C, H, W], got {feat_a.dim()}-D")
    a = F.normalize(feat_a.float(), dim=1, eps=eps)
    b = F.normalize(feat_b.float(), dim=1, eps=eps)
    cos = (a * b).sum(dim=1)            # [B, H, W]
    return 1.0 - cos


class DisagreementWeighter(nn.Module):
    """Per-position disagreement weighting with fixed/learnable, shared/per-scale alpha_d.

    Args:
        levels: FPN levels handled. Default ``("P3", "P4", "P5")``.
        mode: ``"fixed"`` — alpha_d is a constant buffer (the literature
            setting); ``"learnable"`` — alpha_d is an nn.Parameter tuned during
            training (this paper's extension).
        per_scale: if True, one alpha_d per level (scale-aware extension); if
            False, a single shared alpha_d.
        init_alpha: initial / fixed value of alpha_d.
        clamp_max: upper clamp on the weight ``exp(alpha_d * d)`` (Risk 20 —
            prevents signal explosion).
        alpha_clamp: alpha_d is clamped to ``[-alpha_clamp, alpha_clamp]``
            before use — bounds a learnable alpha_d's range.
    """

    def __init__(
        self,
        levels: tuple = ("P3", "P4", "P5"),
        mode: str = "learnable",
        per_scale: bool = True,
        init_alpha: float = 1.0,
        clamp_max: float = 10.0,
        alpha_clamp: float = 3.0,
    ):
        super().__init__()
        if mode not in ("fixed", "learnable"):
            raise ValueError(f"mode must be 'fixed' or 'learnable', got {mode!r}")
        if clamp_max <= 0:
            raise ValueError(f"clamp_max must be positive, got {clamp_max}")
        if alpha_clamp <= 0:
            raise ValueError(f"alpha_clamp must be positive, got {alpha_clamp}")

        self.levels = tuple(levels)
        self.mode = mode
        self.per_scale = bool(per_scale)
        self.clamp_max = float(clamp_max)
        self.alpha_clamp = float(alpha_clamp)

        n = len(self.levels) if self.per_scale else 1
        alpha_init = torch.full((n,), float(init_alpha))
        if mode == "learnable":
            self.alpha = nn.Parameter(alpha_init)
        else:
            self.register_buffer("alpha", alpha_init)

    # ── alpha access ─────────────────────────────────────────────────────

    def _alpha_for(self, level_idx: int) -> torch.Tensor:
        """Clamped alpha_d for a given level index."""
        alpha_eff = self.alpha.clamp(-self.alpha_clamp, self.alpha_clamp)
        return alpha_eff[level_idx] if self.per_scale else alpha_eff[0]

    def get_alpha(self) -> Dict[str, float]:
        """Current (clamped) alpha_d per level — for logging the trajectory."""
        with torch.no_grad():
            return {
                lv: float(self._alpha_for(i))
                for i, lv in enumerate(self.levels)
            }

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        features_a: Dict[str, torch.Tensor],
        features_b: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-position disagreement weights for each level.

        Args:
            features_a, features_b: ``{level: [B, C, H, W]}`` — typically the
                COCO teacher and the SSL teacher features. Disagreement is
                symmetric, so the order does not matter.

        Returns:
            ``{level: [B, H, W]}`` weight maps. A consensus loss multiplies its
            per-position loss map by these.
        """
        missing_a = [lv for lv in self.levels if lv not in features_a]
        missing_b = [lv for lv in self.levels if lv not in features_b]
        if missing_a or missing_b:
            raise ValueError(
                f"features missing levels — a:{missing_a} b:{missing_b}"
            )

        out: Dict[str, torch.Tensor] = {}
        for i, lv in enumerate(self.levels):
            d = cosine_disagreement(features_a[lv], features_b[lv])  # [B,H,W]
            alpha = self._alpha_for(i)
            w = torch.exp(alpha * d).clamp(max=self.clamp_max)
            out[lv] = w
        return out

    # ── repr ─────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"levels={self.levels}, mode={self.mode}, "
            f"per_scale={self.per_scale}, clamp_max={self.clamp_max}, "
            f"alpha_clamp={self.alpha_clamp}"
        )
