"""Anchored joint training — shared scaffold for teacher channels.

The carrier for TERRA (geoteach/), REVISIT (persistence/) and GASP-Real
(scalereal/): a COCO-replay-anchored joint trainer (R3) into which each method
plugs as an :class:`AuxChannel` — its own heads, loss terms and dataloader —
trained in the same optimizer steps as the replay detection loss (the repo's
only historical win: 0.6719 vs 0.6593).

Public API:
    AuxChannel, probe_tap_channels,
    probe_tap_features                      — channel plugin interface
    AnchoredJointTrainer                    — the joint trainer
    SentinelLog, SentinelThresholds,
    SentinelAbort, effective_rank,
    linear_cka                              — R9 health monitors
    save_checkpoint, load_for_finetune      — R8 whole-detector transplant

Heavy deps (ultralytics) are imported lazily inside trainer/export functions;
importing this package needs torch only.
"""

from .channel import AuxChannel, probe_tap_channels, probe_tap_features
from .export import load_for_finetune, save_checkpoint
from .sentinels import (
    SentinelAbort,
    SentinelLog,
    SentinelThresholds,
    effective_rank,
    linear_cka,
)
from .trainer import AnchoredJointTrainer

__all__ = [
    "AuxChannel",
    "probe_tap_channels",
    "probe_tap_features",
    "AnchoredJointTrainer",
    "SentinelLog",
    "SentinelThresholds",
    "SentinelAbort",
    "effective_rank",
    "linear_cka",
    "save_checkpoint",
    "load_for_finetune",
]
