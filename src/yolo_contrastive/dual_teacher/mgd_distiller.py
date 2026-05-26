"""MGDDistiller — Masked Generative Distillation (DT-SAPS Improved, Aşama 0).

Implements vanilla MGD (Yang et al., "Masked Generative Distillation",
ECCV 2022, arXiv:2205.01529) as a standalone distillation module — kept
separate from ConsensusLoss by design (Karar: DT-SAPS Improved Aşama 0),
so the v1 ConsensusLoss / DisagreementWeighter code and tests stay
untouched, and Aşama 2's adversarial mask scheduling (ADS) has a clean
home here later.

Why MGD — the early-saturation diagnosis:
    DT-SAPS v1's distillation loss (Form B = weighted L2, Form C = CWD-KL)
    saturated early: a controlled probe measured ep0 distill = 1.02 on a
    random-init student, collapsing to ~0.08 after a single epoch. A frozen
    teacher's L2/CWD feature target is a STATIC point estimate — once the
    student matches it, the target is exhausted and there is nothing left
    to learn. MGD replaces "copy the teacher feature" with "regenerate the
    teacher feature from a randomly masked student feature": the mask is
    re-sampled every batch, so the target is a combinatorially large set of
    reconstruction tasks rather than one fixed point. It cannot be exhausted.

Mechanism (per scale level, per batch):
    1. mask  — a fresh per-position Bernoulli mask, mask=1 with prob (1-lambda);
               lambda (lambda_mask) is the fraction of positions hidden.
    2. masked_student = student_feat * mask        (lambda fraction zeroed out)
    3. gen   = generator[level](masked_student)    (2x 3x3 conv + ReLU)
    4. L     = MSE(gen, teacher_feat.detach())     (teacher is the target)
    loss = mean over levels.

The mask is spatial (B,1,H,W) — whole positions are hidden across all
channels, matching MGD's "mask random pixels of the student's feature".
The teacher feature is detached: it is the target, no gradient flows into
it; gradient reaches the student backbone through masked_student and the
generator blocks.

Asama 0 scope: lambda is a fixed float (default 0.65, MGD's detection
setting). Asama 2 will replace it with a GRL-driven learnable module (ADS)
— the constructor signature is kept compatible with that future change.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDDistiller(nn.Module):
    """Masked Generative Distillation across multiple feature scales.

    Args:
        levels: scale level names, e.g. ("P3", "P4", "P5").
        channels: per-level student channel count, e.g.
            {"P3": 64, "P4": 128, "P5": 256}. The generator for a level
            maps C -> C, so student and teacher must share this count.
        lambda_mask: fraction of feature positions to hide (MGD's lambda).
            0.65 is MGD's detection default. Fixed in Asama 0.
    """

    def __init__(
        self,
        levels: Tuple[str, ...],
        channels: Dict[str, int],
        lambda_mask: float = 0.65,
    ):
        super().__init__()
        if not 0.0 < lambda_mask < 1.0:
            raise ValueError(
                f"lambda_mask must be in (0, 1), got {lambda_mask!r}"
            )
        missing = [lv for lv in levels if lv not in channels]
        if missing:
            raise ValueError(f"channels missing for levels {missing}")

        self.levels = tuple(levels)
        self.lambda_mask = float(lambda_mask)

        self.generators = nn.ModuleDict()
        for lv in self.levels:
            c = channels[lv]
            self.generators[lv] = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
            )

    def forward(
        self,
        student_feats: Dict[str, torch.Tensor],
        teacher_feats: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the MGD loss.

        Args:
            student_feats: {level: (B, C, H, W)} raw student features.
            teacher_feats: {level: (B, C, H, W)} teacher features — the
                regeneration target. Detached internally.

        Returns:
            (loss, info) where loss is a scalar tensor (mean over levels)
            and info is {"mgd_loss": float, "per_level": {level: float}}.
        """
        per_level: Dict[str, float] = {}
        level_losses = []

        for lv in self.levels:
            s = student_feats[lv]
            t = teacher_feats[lv].detach()
            if s.shape != t.shape:
                raise ValueError(
                    f"level {lv}: student/teacher shape mismatch "
                    f"{tuple(s.shape)} vs {tuple(t.shape)}"
                )

            b, _, h, w = s.shape
            mask = (
                torch.rand(b, 1, h, w, device=s.device)
                > self.lambda_mask
            ).to(s.dtype)

            masked_student = s * mask
            gen = self.generators[lv](masked_student)
            loss_lv = F.mse_loss(gen, t)

            level_losses.append(loss_lv)
            per_level[lv] = float(loss_lv.detach())

        loss = torch.stack(level_losses).mean()
        return loss, {"mgd_loss": float(loss.detach()), "per_level": per_level}
