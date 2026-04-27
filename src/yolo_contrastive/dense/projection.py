"""Multi-scale projection head for dense contrastive learning.

Faz 1.6a — Foundation for Dense + Multi-scale CL (WORK_PLAN_v3 §5).

Projects raw FPN feature maps (P3/P4/P5) to a shared embedding dimension D
for contrastive loss. Per-level: each FPN level has its own 2-layer 1×1
conv tower because input channel counts differ (e.g. YOLOv8n: P3=128,
P4=256, P5=512).

Architecture per level:
    Conv2d(in_C → hidden, 1×1)
    BatchNorm2d(hidden)
    ReLU
    Conv2d(hidden → out_dim, 1×1)

Output is NOT L2-normalized — caller is responsible. This keeps the head
debuggable (raw embedding norms can be inspected) and matches dense_loss's
"caller normalizes" convention.

No SyncBN — single-GPU only (see WORK_PLAN_v3 risk #2).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MultiScaleProjectionHead(nn.Module):
    """Per-level 2-layer 1×1 conv projection head.

    Args:
        in_channels: dict {level_name: input channel count}.
                     For YOLOv8n: {"P3": 128, "P4": 256, "P5": 512}.
                     Pass actual channel counts from the backbone — they
                     differ across YOLOv8 model sizes.
        out_dim: embedding dimension D shared across levels.
        hidden_dim: hidden channel count in the tower. Default 2*out_dim.
        use_bn: insert BatchNorm2d after the first conv. Default True.

    Usage:
        head = MultiScaleProjectionHead(
            in_channels={"P3": 128, "P4": 256, "P5": 512},
            out_dim=256,
        )
        # features from MultiScaleFeatureTap
        embeds = head(features)   # {level: [B, D, H, W]}
        # caller normalizes
        embeds_n = {lv: F.normalize(t, dim=1) for lv, t in embeds.items()}
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        out_dim: int = 256,
        hidden_dim: Optional[int] = None,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels dict is empty")
        for lv, c in in_channels.items():
            if not isinstance(c, int) or c <= 0:
                raise ValueError(f"in_channels[{lv!r}]={c} must be positive int")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.in_channels: Dict[str, int] = dict(in_channels)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else 2 * out_dim
        self.use_bn = bool(use_bn)

        # Per-level conv towers — registered as ModuleDict so state_dict + .to() work
        towers: Dict[str, nn.Module] = {}
        for lv, in_c in self.in_channels.items():
            layers = [nn.Conv2d(in_c, self.hidden_dim, kernel_size=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1))
            towers[lv] = nn.Sequential(*layers)
        self.towers = nn.ModuleDict(towers)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project per-level features to embedding space.

        Args:
            features: {level_name: [B, C_in, H, W]} — keys must match in_channels.

        Returns:
            {level_name: [B, out_dim, H, W]} — same spatial size, NOT normalized.
        """
        missing = [lv for lv in self.in_channels if lv not in features]
        if missing:
            raise ValueError(
                f"Missing levels in features: {missing}. "
                f"Got {sorted(features.keys())}, expected {sorted(self.in_channels.keys())}"
            )

        out: Dict[str, torch.Tensor] = {}
        for lv, in_c in self.in_channels.items():
            t = features[lv]
            if t.dim() != 4:
                raise ValueError(
                    f"Feature {lv!r} must be [B, C, H, W], got shape {tuple(t.shape)}"
                )
            if t.shape[1] != in_c:
                raise ValueError(
                    f"Feature {lv!r} channel mismatch: got {t.shape[1]}, "
                    f"expected {in_c}"
                )
            out[lv] = self.towers[lv](t)
        return out

    # ── repr ──────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim}, use_bn={self.use_bn}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Helper: infer in_channels from a YOLO-like model + tap
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer_in_channels(
    model: nn.Module,
    tap: "MultiScaleFeatureTap",  # type: ignore[name-defined]  # forward ref to avoid circular
    imgsz: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, int]:
    """Probe a model with a dummy forward to read FPN channel counts.

    Useful when you don't want to hardcode YOLOv8n/s/m/l/x channel widths.

    Args:
        model: YOLO model (or any nn.Module accepting [B, 3, H, W]).
        tap: an already-set-up MultiScaleFeatureTap on `model`.
        imgsz: dummy input size — small (64) is plenty for shape probing.
        device: where to run the probe. Default: model's device.

    Returns:
        {level_name: channel_count}
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    was_training = model.training
    model.eval()
    try:
        _ = model(dummy)
    finally:
        if was_training:
            model.train()
    feats = tap.get_features()
    return {lv: int(t.shape[1]) for lv, t in feats.items()}
