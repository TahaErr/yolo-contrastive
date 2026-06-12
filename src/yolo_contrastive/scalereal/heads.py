"""GASP-Real student-side heads — all trainable, all DISCARDED at export.

Four small modules (~370K params total at C4=128):

    PatchDescriptor φ : RoIAlign(P4, boxes, 3x3, aligned=True) -> flatten ->
                        MLP C4*9 -> 256 -> 128 (LayerNorm + SiLU).
    ScaleHead g       : 128 -> 64 -> 1 scalar s(x) — the "log apparent-scale
                        potential". Pair prediction is the DIFFERENCE
                        s(φ_B) - s(φ_A): antisymmetric BY CONSTRUCTION and
                        provably per-patch (a pair-MLP could shortcut via
                        pair interactions while leaving per-patch features
                        scale-blind).
    ContentProjector p: 128 -> 128 -> 64 (LN + SiLU, no BatchNorm).
    Predictor q       : 64 -> 64 -> 64 (the SimSiam predictor).

E5 discipline: ``requires_grad`` is explicitly set True on every parameter at
construction (ultralytics models ship some params frozen; fresh heads should
never inherit that ambiguity). C4 is inferred from the tap at attach time,
never hardcoded.

R6: none of these modules ever touches a teacher — they are student heads
registered into the trainer's head-LR optimizer group via
``ScaleRealChannel.attach`` and excluded from the exported detector (R8).

torchvision (roi_align) is imported lazily inside ``forward`` (E2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _enable_grads(module: nn.Module) -> None:
    """E5: explicitly enable requires_grad on every parameter."""
    for p in module.parameters():
        p.requires_grad_(True)


def _mlp(dims, final_norm: bool = True) -> nn.Sequential:
    """LayerNorm+SiLU MLP over ``dims`` (no BatchNorm — tiny RoI batches)."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        last = i == len(dims) - 2
        if not last or final_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PatchDescriptor(nn.Module):
    """RoIAlign-pooled per-patch descriptor from the P4 tap.

    Args:
        in_channels: P4 channel count (inferred from the tap at attach time
            via ``probe_tap_channels`` — 128 for yolov8n, never hardcoded).
        output_size: RoIAlign cell grid (3 -> 3x3).
        hidden: MLP hidden width.
        out_dim: descriptor dimensionality.
    """

    def __init__(
        self,
        in_channels: int,
        output_size: int = 3,
        hidden: int = 256,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.in_channels = int(in_channels)
        self.output_size = int(output_size)
        self.out_dim = int(out_dim)
        self.mlp = _mlp([self.in_channels * self.output_size ** 2, hidden, out_dim])
        _enable_grads(self)

    def forward(
        self,
        p4: torch.Tensor,
        rois: torch.Tensor,
        spatial_scale: float,
    ) -> torch.Tensor:
        """Pool + embed patches.

        Args:
            p4: [B, C4, Hf, Wf] feature map from the shared tap.
            rois: [M, 5] (batch_idx, x1, y1, x2, y2) in INPUT-IMAGE pixels.
            spatial_scale: feature/image scale (1/16 for P4; pass the value
                computed from the actual tap shape, asserted by the channel).

        Returns:
            [M, out_dim] descriptors.
        """
        from torchvision.ops import roi_align  # lazy heavy dep (E2)

        if rois.dim() != 2 or rois.shape[1] != 5:
            raise ValueError(f"rois must be [M, 5] (batch_idx, xyxy px), got {tuple(rois.shape)}")
        pooled = roi_align(
            p4, rois.to(p4.dtype),
            output_size=self.output_size,
            spatial_scale=float(spatial_scale),
            sampling_ratio=2,
            aligned=True,
        )  # [M, C4, k, k]
        return self.mlp(pooled.flatten(1))


class ScaleHead(nn.Module):
    """Per-patch scalar log apparent-scale potential s(x): D -> hidden -> 1."""

    def __init__(self, in_dim: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        _enable_grads(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [M]


class ContentProjector(nn.Module):
    """SimSiam projector p: D -> D -> proj_dim (LN + SiLU, no BN)."""

    def __init__(self, in_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = _mlp([in_dim, in_dim, out_dim], final_norm=False)
        _enable_grads(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """SimSiam predictor q: proj_dim -> proj_dim -> proj_dim."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.net = _mlp([dim, dim, dim], final_norm=False)
        _enable_grads(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
