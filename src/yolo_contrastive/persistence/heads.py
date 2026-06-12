"""Trainable heads for the REVISIT persistence channel.

All three modules are STUDENT-side (R6: zero teacher-side trainables), live
in the trainer's head-LR optimizer group, and are discarded at export (R8:
inference cost stays exactly YOLOv8n).

    CorrespondenceProjector g : Linear(C -> 256) + BN1d + ReLU + Linear(256 -> 128)
    CorrespondencePredictor q : Linear(128 -> 256) + BN1d + ReLU + Linear(256 -> 128)
    PersistenceHead           : Conv3x3(C -> C) + BN2d + SiLU + Conv1x1(C -> 3)

The SimSiam pairing (predictor + stop-grad, NO second EMA encoder) keeps the
trainer's model-EMA reserved for export. ``passthrough=True`` is a test hook:
forward returns its input unchanged so analytic loss-minimum tests can reason
about raw tap features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CorrespondenceProjector",
    "CorrespondencePredictor",
    "PersistenceHead",
    "simsiam_one_way",
]


class CorrespondenceProjector(nn.Module):
    """SimSiam-style projector g over per-point P3 features ([N, C] -> [N, D])."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 passthrough: bool = False) -> None:
        super().__init__()
        self.passthrough = bool(passthrough)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.passthrough:
            return x
        return self.net(x)


class CorrespondencePredictor(nn.Module):
    """SimSiam predictor q ([N, D] -> [N, D]); the stop-grad counterpart."""

    def __init__(self, dim: int = 128, hidden_dim: int = 256,
                 passthrough: bool = False) -> None:
        super().__init__()
        self.passthrough = bool(passthrough)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.passthrough:
            return x
        return self.net(x)


class PersistenceHead(nn.Module):
    """Tiny dense 3-class head on P3 (~40K params at C=64).

    Classes: 0 background / 1 persistent / 2 transient (ignore = 255 handled
    by the loss, not the head).
    """

    def __init__(self, in_ch: int, n_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simsiam_one_way(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """One direction of the SimSiam loss: mean(1 - cos(p, stopgrad(z))).

    The stop-gradient on the target branch is INSIDE this function so it can
    never be forgotten at a call site (R4: positive-only consistency)."""
    return (1.0 - F.cosine_similarity(p, z.detach(), dim=-1)).mean()
