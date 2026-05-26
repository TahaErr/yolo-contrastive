"""MGDDistiller — Masked Generative Distillation (DT-SAPS Improved).

Implements MGD (Yang et al., "Masked Generative Distillation", ECCV 2022,
arXiv:2205.01529) as a standalone module — separate from ConsensusLoss by
design, so the v1 ConsensusLoss / DisagreementWeighter code and tests stay
untouched.

Why MGD — the early-saturation diagnosis:
    DT-SAPS v1's distillation loss (Form B = weighted L2, Form C = CWD-KL)
    saturated early: a controlled probe measured ep0 distill = 1.02 on a
    random-init student, collapsing to ~0.08 after one epoch. A frozen
    teacher's L2/CWD feature target is a STATIC point estimate — once the
    student matches it, the target is exhausted. MGD replaces "copy the
    teacher feature" with "regenerate the teacher feature from a randomly
    masked student feature": the mask is re-sampled every batch, so the
    target is a combinatorially large set of reconstruction tasks.

Two modes:
    single-teacher (Asama 0)  — forward(student, teacher); uniform random
        mask. Vanilla MGD. Measured: removes v1's ep2 wall, but a soft
        plateau remains after ep~5.
    dual-teacher + DAMS (Asama 1) — forward(student, coco, ssl); two
        generator sets (one per teacher) and Disagreement-Aware Mask
        Sampling: the mask is sampled from the two teachers' per-position
        cosine-disagreement map instead of uniformly. Positions where the
        teachers disagree most — the highest-uncertainty, most informative
        regions — are masked preferentially, so the student is forced to
        regenerate exactly there.

DAMS mechanism (per level, per batch, non-differentiable sampling):
    D(i,j)  = cosine_disagreement(f_coco, f_ssl)      in [0, 2]
    P(i,j)  = softmax(D / tau_mask)                   sampling distribution
    mask    = multinomial-sample floor(lambda * H * W) positions from P,
              set them to 0 (hidden); the rest stay 1 (kept).
    Non-differentiable by design (Karar: Asama 1) — torch.multinomial,
    standard MGD practice. Asama 2 (ADS) will make the mask RATIO learnable
    via a GRL; tau_mask stays a fixed float here.

The mask is spatial (B,1,H,W). The teacher features are detached — they are
the regeneration target; gradient reaches the student backbone through the
masked student feature and the generator blocks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .disagreement import cosine_disagreement


def _simple_block(c: int) -> nn.Sequential:
    """MGD's generator: two 3x3 convs with a ReLU between, C -> C."""
    return nn.Sequential(
        nn.Conv2d(c, c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c, kernel_size=3, padding=1),
    )


class MGDDistiller(nn.Module):
    """Masked Generative Distillation, single- or dual-teacher.

    Args:
        levels: scale level names, e.g. ("P3", "P4", "P5").
        channels: per-level student channel count, e.g.
            {"P3": 64, "P4": 128, "P5": 256}. Generators map C -> C.
        lambda_mask: fraction of feature positions to hide (MGD's lambda).
            0.65 is MGD's detection default. Fixed in Asama 0/1.
        dual_teacher: if True, build a second generator set and enable DAMS.
            forward() then expects two teacher dicts.
        tau_mask: DAMS softmax temperature over the disagreement map.
            Lower -> sharper (mask concentrates on top-disagreement
            positions); higher -> closer to uniform. Fixed in Asama 1.
    """

    def __init__(
        self,
        levels: Tuple[str, ...],
        channels: Dict[str, int],
        lambda_mask: float = 0.65,
        dual_teacher: bool = False,
        tau_mask: float = 0.5,
    ):
        super().__init__()
        if not 0.0 < lambda_mask < 1.0:
            raise ValueError(f"lambda_mask must be in (0, 1), got {lambda_mask!r}")
        if tau_mask <= 0.0:
            raise ValueError(f"tau_mask must be positive, got {tau_mask!r}")
        missing = [lv for lv in levels if lv not in channels]
        if missing:
            raise ValueError(f"channels missing for levels {missing}")

        self.levels = tuple(levels)
        self.lambda_mask = float(lambda_mask)
        self.dual_teacher = bool(dual_teacher)
        self.tau_mask = float(tau_mask)

        # Generator set for the first teacher (the only one in single-teacher
        # mode; the COCO teacher in dual-teacher mode).
        self.generators = nn.ModuleDict(
            {lv: _simple_block(channels[lv]) for lv in self.levels}
        )
        # Second generator set — SSL teacher — only in dual-teacher mode.
        self.generators_ssl: Optional[nn.ModuleDict] = None
        if self.dual_teacher:
            self.generators_ssl = nn.ModuleDict(
                {lv: _simple_block(channels[lv]) for lv in self.levels}
            )

    # ── mask sampling ────────────────────────────────────────────────────

    def _uniform_mask(self, b: int, h: int, w: int, device, dtype):
        """Asama 0 — per-position Bernoulli, mask=1 kept with prob (1-lambda)."""
        return (torch.rand(b, 1, h, w, device=device) > self.lambda_mask).to(dtype)

    def _dams_mask(self, disagreement: torch.Tensor, dtype):
        """Asama 1 — Disagreement-Aware Mask Sampling (non-differentiable).

        Args:
            disagreement: [B, H, W] per-position cosine disagreement.

        Returns:
            [B, 1, H, W] mask — floor(lambda*H*W) positions hidden (0),
            sampled without replacement from softmax(D / tau_mask).
        """
        b, h, w = disagreement.shape
        n = h * w
        k = int(self.lambda_mask * n)            # positions to hide
        flat_d = disagreement.reshape(b, n)      # [B, HW]
        # Sampling distribution — sharper where teachers disagree.
        probs = F.softmax(flat_d / self.tau_mask, dim=1)   # [B, HW]
        mask = torch.ones(b, n, device=disagreement.device, dtype=dtype)
        if k > 0:
            # Sample k distinct positions per image (non-differentiable).
            idx = torch.multinomial(probs, num_samples=k, replacement=False)
            mask.scatter_(1, idx, 0.0)
        return mask.reshape(b, 1, h, w)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        student_feats: Dict[str, torch.Tensor],
        teacher_feats: Dict[str, torch.Tensor],
        ssl_feats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute the MGD loss.

        single-teacher: forward(student, teacher) — uniform mask.
        dual-teacher:   forward(student, coco, ssl) — two generator sets,
                        DAMS mask from the COCO/SSL disagreement map.

        Returns:
            (loss, info). info carries "mgd_loss", "per_level", and in
            dual-teacher mode "coco_loss"/"ssl_loss" and the mean masked
            disagreement (a DAMS health signal).
        """
        if self.dual_teacher and ssl_feats is None:
            raise ValueError("dual_teacher=True requires ssl_feats")
        if not self.dual_teacher and ssl_feats is not None:
            raise ValueError("single-teacher mode got an unexpected ssl_feats")

        per_level: Dict[str, float] = {}
        level_losses = []
        coco_losses, ssl_losses, disag_means = [], [], []

        for lv in self.levels:
            s = student_feats[lv]
            t_coco = teacher_feats[lv].detach()
            if s.shape != t_coco.shape:
                raise ValueError(
                    f"level {lv}: student/teacher shape mismatch "
                    f"{tuple(s.shape)} vs {tuple(t_coco.shape)}"
                )
            b, _, h, w = s.shape

            if self.dual_teacher:
                t_ssl = ssl_feats[lv].detach()
                if s.shape != t_ssl.shape:
                    raise ValueError(
                        f"level {lv}: student/ssl shape mismatch "
                        f"{tuple(s.shape)} vs {tuple(t_ssl.shape)}"
                    )
                # DAMS — mask from the teacher disagreement map.
                d = cosine_disagreement(t_coco, t_ssl)        # [B,H,W]
                mask = self._dams_mask(d, s.dtype)
                disag_means.append(float(d.mean().detach()))

                masked = s * mask
                gen_coco = self.generators[lv](masked)
                gen_ssl = self.generators_ssl[lv](masked)
                l_coco = F.mse_loss(gen_coco, t_coco)
                l_ssl = F.mse_loss(gen_ssl, t_ssl)
                loss_lv = l_coco + l_ssl
                coco_losses.append(l_coco)
                ssl_losses.append(l_ssl)
            else:
                # Asama 0 — single teacher, uniform mask.
                mask = self._uniform_mask(b, h, w, s.device, s.dtype)
                masked = s * mask
                gen = self.generators[lv](masked)
                loss_lv = F.mse_loss(gen, t_coco)

            level_losses.append(loss_lv)
            per_level[lv] = float(loss_lv.detach())

        loss = torch.stack(level_losses).mean()
        info: Dict = {"mgd_loss": float(loss.detach()), "per_level": per_level}
        if self.dual_teacher:
            info["coco_loss"] = float(torch.stack(coco_losses).mean().detach())
            info["ssl_loss"] = float(torch.stack(ssl_losses).mean().detach())
            info["disagreement_mean"] = sum(disag_means) / len(disag_means)
        return loss, info
