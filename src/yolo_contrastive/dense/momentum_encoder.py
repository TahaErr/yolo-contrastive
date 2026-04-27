"""Momentum encoder — EMA wrapper for MoCo-style contrastive learning.

Faz 1.3 — Foundation for Dense + Multi-scale CL (WORK_PLAN_v3 §5).

Maintains a momentum-averaged copy of the online encoder. The momentum
encoder produces "stable" keys whose representations don't shift rapidly
with optimizer steps, providing temporally consistent negatives for the
queue (Faz 1.2).

Update rule (per call to update()):
    θ_m ← m · θ_m + (1 - m) · θ_online
    b_m ← m · b_m + (1 - m) · b_online   (for float buffers, e.g. BN running stats)
    b_m ← b_online                       (for integer buffers, e.g. num_batches_tracked)

Key design choices:
    - Stateless update: caller passes the online encoder on each update().
      Online is NOT stored as an attribute, so it doesn't double-register
      in state_dict and there's no stale-reference risk.
    - Buffer EMA (MoCo-v3 / BYOL convention) for float buffers; integer
      buffers (num_batches_tracked) are copied verbatim.
    - Always eval() mode — BN uses running stats, never updates them
      during forward. Forward is wrapped in @torch.no_grad.
    - force_fp32 default True: disables AMP autocast inside momentum
      forward to prevent fp16 numerics from leaking into queue/similarity
      downstream. Toggle off only if you understand the implications.
    - Hooks stripped from the deep copy: if the online encoder has hooks
      (e.g. a MultiScaleFeatureTap), the deepcopy would carry them but
      with closures pointing at the *original* tap's state dict — a
      silent footgun. We clear forward hooks on the momentum copy.

Single-GPU only. DDP shuffle-BN deferred — see WORK_PLAN_v3 risk #2.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


def _strip_forward_hooks(module: nn.Module) -> None:
    """Remove forward hooks recursively (defensive — see module docstring)."""
    for m in module.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()


class MomentumEncoder(nn.Module):
    """EMA-averaged copy of an online encoder.

    Args:
        online: The online encoder (will be deep-copied at init).
        m: Momentum coefficient. θ_m ← m·θ_m + (1-m)·θ_online.
           Larger m = slower update = more stable keys.
           Common: 0.999 (MoCo, BYOL), 0.99 (smaller datasets / faster adapt).
        force_fp32: If True (default), forward runs with autocast disabled.

    Usage:
        online = build_yolo()
        momentum = MomentumEncoder(online, m=0.999)
        for batch in dataloader:
            ...
            with torch.no_grad():
                k = momentum(view2)        # stable keys
            momentum.update(online)        # EMA step
    """

    def __init__(
        self,
        online: nn.Module,
        m: float = 0.999,
        force_fp32: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= m <= 1.0:
            raise ValueError(f"m must be in [0, 1], got {m}")

        self.momentum = copy.deepcopy(online)
        _strip_forward_hooks(self.momentum)

        for p in self.momentum.parameters():
            p.requires_grad = False
        self.momentum.eval()

        self.m: float = float(m)
        self.force_fp32: bool = bool(force_fp32)

    # ── EMA update ────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        """One EMA step: θ_m ← m·θ_m + (1-m)·θ_online (and buffers).

        Caller is responsible for passing the same structural online module
        used at init (param iteration order must match).
        """
        m = self.m

        # Parameters
        for p_o, p_m in zip(online.parameters(), self.momentum.parameters()):
            p_m.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

        # Buffers (BN running stats etc.)
        for b_o, b_m in zip(online.buffers(), self.momentum.buffers()):
            if b_m.dtype.is_floating_point:
                b_m.data.mul_(m).add_(b_o.data.to(b_m.dtype), alpha=1.0 - m)
            else:
                # Integer buffers (e.g. num_batches_tracked) — copy verbatim
                b_m.data.copy_(b_o.data)

    # ── forward ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Any:
        """Run momentum encoder. Always no_grad + eval mode."""
        if self.force_fp32:
            device_type = "cuda" if x.is_cuda else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                return self.momentum(x)
        return self.momentum(x)

    # ── train/eval override ──────────────────────────────────────────────

    def train(self, mode: bool = True) -> "MomentumEncoder":
        """Override: momentum encoder always stays in eval mode."""
        super().train(mode)
        self.momentum.eval()
        return self

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.momentum.parameters())
        return (
            f"MomentumEncoder(m={self.m}, force_fp32={self.force_fp32}, "
            f"params={n_params:,})"
        )
