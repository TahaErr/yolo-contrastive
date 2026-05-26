"""MGDDistiller — Masked Generative Distillation with RAMS (DT-SAPS Improved).

Implements MGD (Yang et al., "Masked Generative Distillation", ECCV 2022,
arXiv:2205.01529) as a standalone module — separate from ConsensusLoss by
design, so the v1 ConsensusLoss / DisagreementWeighter code and tests stay
untouched.

Why MGD — the early-saturation diagnosis:
    DT-SAPS v1's distillation loss saturated early — a frozen teacher's
    L2/CWD feature target is a STATIC point estimate; once matched, it is
    exhausted. MGD replaces "copy the teacher feature" with "regenerate the
    teacher feature from a randomly masked student feature": the mask is
    re-sampled every batch, so the target is a combinatorially large set of
    reconstruction tasks.

Two modes:
    single-teacher (Asama 0)  — forward(student, teacher); uniform random
        mask. Vanilla MGD. Measured: removes v1's ep2 wall, but a soft
        plateau remains after ep~5.
    dual-teacher + RAMS (Asama 1) — forward(student, coco, ssl); two
        generator sets and Reconstruction-Adaptive Mask Sampling.

RAMS — Reconstruction-Adaptive Mask Sampling:
    Earlier DAMS variants sampled the mask from the two teachers' cosine
    or L2 disagreement. Measured: that signal is STATIC (both teachers are
    frozen; their disagreement does not depend on the student), so the mask
    targets the same positions every epoch and the plateau is not broken —
    a static mask source is just another static target.

    RAMS instead samples the mask from the student's own reconstruction
    error — how badly the student currently regenerates each teacher
    feature. This signal EVOLVES with the student: as the student improves
    in a region, that region's error drops and RAMS shifts attention
    elsewhere. The target moves with the student; it cannot be exhausted.
    A reconstruction-diagnostic measured this signal both structured
    (CV ~0.35) and dynamic (rank-correlation 0.55-0.72 over 5 epochs of
    training), unlike the static disagreement map.

    This is the key distinction from the AMD / DMKD / SAMKD family of
    attention-guided masking: those take the mask cue from the (static)
    teacher attention; RAMS takes it from the (evolving) student
    reconstruction state.

RAMS mechanics (per level, per batch):
    1. Mask is sampled from err_memory[level] — an EMA of the student's
       past per-position reconstruction error. First batch: err_memory is
       empty, the mask is uniform (cold-start).
    2. masked_student = student * mask;  gen_coco/gen_ssl regenerate.
    3. After the loss, err = max(||gen_coco - coco||, ||gen_ssl - ssl||)
       per position — the "weakest link" across the two teachers — is
       computed (no grad), per-image min-max normalized, batch-averaged to
       [H,W], and EMA-folded into err_memory. The EMA both solves the
       chicken-and-egg (the mask uses PAST error, independent of the
       current mask) and smooths a noisy single-batch signal.

err_memory is a buffer (not a parameter — no gradient), but it is module
state: it is saved/restored on checkpoint/resume.

The mask is spatial (B,1,H,W). Teacher features are detached — the
regeneration target; gradient reaches the student backbone through the
masked student feature and the generator blocks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _simple_block(c: int) -> nn.Sequential:
    """MGD's generator: two 3x3 convs with a ReLU between, C -> C."""
    return nn.Sequential(
        nn.Conv2d(c, c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c, kernel_size=3, padding=1),
    )


class MGDDistiller(nn.Module):
    """Masked Generative Distillation, single-teacher or dual-teacher + RAMS.

    Args:
        levels: scale level names, e.g. ("P3", "P4", "P5").
        channels: per-level student channel count. Generators map C -> C.
        lambda_mask: fraction of feature positions to hide (MGD's lambda).
            0.65 is MGD's detection default. Fixed in Asama 0/1.
        dual_teacher: if True, build a second generator set and enable RAMS.
        tau_mask: RAMS softmax temperature over the reconstruction-error
            memory. Lower -> sharper. Fixed in Asama 1.
        ema_momentum: EMA coefficient for err_memory. err_memory =
            m * err_memory + (1 - m) * batch_error. 0.9 default.
    """

    def __init__(
        self,
        levels: Tuple[str, ...],
        channels: Dict[str, int],
        lambda_mask: float = 0.65,
        dual_teacher: bool = False,
        tau_mask: float = 0.5,
        ema_momentum: float = 0.9,
    ):
        super().__init__()
        if not 0.0 < lambda_mask < 1.0:
            raise ValueError(f"lambda_mask must be in (0, 1), got {lambda_mask!r}")
        if tau_mask <= 0.0:
            raise ValueError(f"tau_mask must be positive, got {tau_mask!r}")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {ema_momentum!r}")
        missing = [lv for lv in levels if lv not in channels]
        if missing:
            raise ValueError(f"channels missing for levels {missing}")

        self.levels = tuple(levels)
        self.lambda_mask = float(lambda_mask)
        self.dual_teacher = bool(dual_teacher)
        self.tau_mask = float(tau_mask)
        self.ema_momentum = float(ema_momentum)

        self.generators = nn.ModuleDict(
            {lv: _simple_block(channels[lv]) for lv in self.levels}
        )
        self.generators_ssl: Optional[nn.ModuleDict] = None
        if self.dual_teacher:
            self.generators_ssl = nn.ModuleDict(
                {lv: _simple_block(channels[lv]) for lv in self.levels}
            )
            # RAMS error memory — one [H,W] map per level, lazily sized on
            # the first forward (feature spatial size is not known here).
            # Registered as a buffer: module state, saved on checkpoint,
            # not a gradient parameter. None until the first batch.
            for lv in self.levels:
                self.register_buffer(f"err_memory_{lv}", None, persistent=True)

    # ── mask sampling ────────────────────────────────────────────────────

    def _uniform_mask(self, b: int, h: int, w: int, device, dtype):
        """Per-position Bernoulli, mask=1 (kept) with prob (1-lambda)."""
        return (torch.rand(b, 1, h, w, device=device) > self.lambda_mask).to(dtype)

    def _rams_mask(self, err_map: torch.Tensor, b: int, dtype):
        """RAMS — sample the mask from the reconstruction-error memory.

        Args:
            err_map: [H, W] EMA error memory for this level, in [0, 1].
            b: batch size (the same mask distribution, sampled per image).

        Returns:
            [b, 1, H, W] mask — floor(lambda*H*W) positions hidden (0),
            sampled without replacement from softmax(err_map / tau_mask).
        """
        h, w = err_map.shape
        n = h * w
        k = int(self.lambda_mask * n)
        probs = F.softmax(err_map.reshape(-1) / self.tau_mask, dim=0)  # [HW]
        mask = torch.ones(b, n, device=err_map.device, dtype=dtype)
        if k > 0:
            # Sample k distinct positions per image from the shared dist.
            probs_b = probs.unsqueeze(0).expand(b, -1)                # [b,HW]
            idx = torch.multinomial(probs_b, num_samples=k, replacement=False)
            mask.scatter_(1, idx, 0.0)
        return mask.reshape(b, 1, h, w)

    @staticmethod
    def _recon_error(gen: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-position reconstruction error, per-image min-max normalized.

        Args:
            gen, target: [B, C, H, W].

        Returns:
            [B, H, W] error in [0, 1] per image.
        """
        d = (gen - target).pow(2).mean(dim=1).sqrt()        # [B,H,W]
        b = d.shape[0]
        flat = d.reshape(b, -1)
        d_min = flat.min(dim=1, keepdim=True).values
        d_max = flat.max(dim=1, keepdim=True).values
        flat = (flat - d_min) / (d_max - d_min + 1e-6)
        return flat.reshape(d.shape)

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
                        RAMS mask from the reconstruction-error memory.
        """
        if self.dual_teacher and ssl_feats is None:
            raise ValueError("dual_teacher=True requires ssl_feats")
        if not self.dual_teacher and ssl_feats is not None:
            raise ValueError("single-teacher mode got an unexpected ssl_feats")

        per_level: Dict[str, float] = {}
        level_losses = []
        coco_losses, ssl_losses, err_means = [], [], []

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
                # RAMS — mask from the error memory (or uniform on cold-start).
                mem = getattr(self, f"err_memory_{lv}")
                if mem is None:
                    mask = self._uniform_mask(b, h, w, s.device, s.dtype)
                else:
                    mask = self._rams_mask(mem, b, s.dtype)

                masked = s * mask
                gen_coco = self.generators[lv](masked)
                gen_ssl = self.generators_ssl[lv](masked)
                l_coco = F.mse_loss(gen_coco, t_coco)
                l_ssl = F.mse_loss(gen_ssl, t_ssl)
                loss_lv = l_coco + l_ssl
                coco_losses.append(l_coco)
                ssl_losses.append(l_ssl)

                # Update err_memory — weakest-link error across teachers.
                with torch.no_grad():
                    err_coco = self._recon_error(gen_coco.detach(), t_coco)
                    err_ssl = self._recon_error(gen_ssl.detach(), t_ssl)
                    err = torch.maximum(err_coco, err_ssl)      # [B,H,W]
                    err_batch = err.mean(dim=0)                 # [H,W]
                    err_means.append(float(err_batch.mean()))
                    if mem is None:
                        self.register_buffer(
                            f"err_memory_{lv}", err_batch.clone(),
                            persistent=True,
                        )
                    else:
                        m = self.ema_momentum
                        new_mem = m * mem + (1.0 - m) * err_batch
                        setattr(self, f"err_memory_{lv}", new_mem)
            else:
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
            info["recon_error_mean"] = sum(err_means) / len(err_means)
        return loss, info
