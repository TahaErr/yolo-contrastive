"""Contrastive loss functions for yolo-contrastive.

Fixes applied (audit report §2):
  2.1 [HIGH]   B<2 returns zero-loss with warning (instead of silent degenerate result).
  2.2 [MEDIUM] torch.eye / torch.arange cached per (B, device) to avoid repeated allocation.
  2.3 [MEDIUM] `labels` param raises warning when passed (not yet supported by NTXent).
  2.4 [LOW]    z2=None fallback clearly documented.
  2.5 [LOW]    Large-batch memory warning added (B>=1024).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Type

import torch

from ..exceptions import ContrastiveLossError
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE / SimCLR) contrastive loss.

    Expects two views of the same batch:
        ``z1, z2`` — each with shape ``[B, D]`` (or ``[B, ...]`` which will be flattened).

    Positives are ``(i) <-> (i + B)`` pairs in the concatenated ``[2B, D]`` tensor.

    Behaviour notes:
        - If ``z2 is None``, it defaults to ``z1`` (self-similarity).  This is a
          degenerate case where every sample is its own positive — the loss will be
          near ``log(2B-1)`` and carry no contrastive signal.  Only useful as a
          passthrough / sanity-check mode.
        - If ``B < 2``, meaningful negatives cannot be formed.  The method returns
          ``0.0`` (with ``requires_grad=True``) and emits a warning.

    Args:
        temperature: Softmax temperature (must be > 0).  Default ``0.2``.
        eps: Epsilon for L2 normalisation.
        large_batch_warn: Emit a warning when ``B >= large_batch_warn`` about the
            ``[2B, 2B]`` similarity matrix memory cost.  Set ``0`` to disable.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        eps: float = 1e-8,
        large_batch_warn: int = 1024,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ContrastiveLossError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.large_batch_warn = int(large_batch_warn)

        # Cache for diag mask and target indices — keyed by (B, device)
        self._cache: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    _MAX_CACHE_SIZE = 8

    def _get_cached(
        self, B: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (diag_mask[2B,2B], targets[2B]) — created once per (B, device)."""
        key = (B, device)
        if key not in self._cache:
            if len(self._cache) >= self._MAX_CACHE_SIZE:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            n = 2 * B
            diag = torch.eye(n, device=device, dtype=torch.bool)
            targets = (torch.arange(n, device=device) + B) % n
            self._cache[key] = (diag, targets)
        return self._cache[key]

    def forward(
        self,
        z1: torch.Tensor,
        z2: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- labels guard (not supported yet — SupCon planned) ---
        if labels is not None:
            warnings.warn(
                "NTXentLoss received `labels` but does not use them. "
                "Labels will be supported in a future SupConLoss. "
                "Ignoring for now.",
                UserWarning,
                stacklevel=2,
            )

        if z2 is None:
            z2 = z1

        # Flatten to [B, D]
        z1 = z1.view(z1.size(0), -1)
        z2 = z2.view(z2.size(0), -1)

        if z1.size(0) != z2.size(0):
            raise ContrastiveLossError(
                f"Batch size mismatch: z1={z1.size(0)} vs z2={z2.size(0)}"
            )

        B = z1.size(0)

        # --- B < 2 guard ---
        if B < 2:
            warnings.warn(
                f"NTXentLoss: batch size B={B} < 2 — cannot form meaningful negatives. "
                f"Returning zero loss. Increase batch size for contrastive learning to work.",
                UserWarning,
                stacklevel=2,
            )
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype, requires_grad=True)

        # --- Large-batch memory warning ---
        if self.large_batch_warn > 0 and B >= self.large_batch_warn:
            mem_mb = (2 * B) ** 2 * 4 / (1024 ** 2)  # float32
            warnings.warn(
                f"NTXentLoss: B={B} → similarity matrix is [{2*B}, {2*B}] "
                f"(~{mem_mb:.0f} MB). Consider gradient checkpointing for very large batches.",
                UserWarning,
                stacklevel=2,
            )

        # --- Compute loss in float32 (avoid fp16 overflow in masked_fill) ---
        device_type = "cuda" if z1.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z1f = F.normalize(z1.float(), dim=1, eps=self.eps)
            z2f = F.normalize(z2.float(), dim=1, eps=self.eps)

            z = torch.cat([z1f, z2f], dim=0)  # [2B, D]

            sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

            diag, targets = self._get_cached(B, sim.device)
            sim = sim.masked_fill(diag, -1e9)

            loss = F.cross_entropy(sim, targets)

        return loss


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ntxent": NTXentLoss,
    "infonce": NTXentLoss,
    "simclr": NTXentLoss,
}


def build_contrastive_loss(name: str = "ntxent", **kwargs) -> nn.Module:
    """Build a contrastive loss by name.

    Available names: ``ntxent``, ``infonce``, ``simclr`` (all aliases for NTXentLoss).
    """
    key = str(name).strip().lower()
    if key not in _LOSS_REGISTRY:
        raise ContrastiveLossError(
            f"Unknown contrastive loss '{name}'. "
            f"Available: {sorted(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[key](**kwargs)
