"""Evaluation utilities — linear probe, leakage check, run matrix."""

from .linear_probe import LinearProbeTrainer, LinearProbeHead
from .run_matrix import RunMatrix, CSV_COLUMNS

__all__ = [
    "LinearProbeTrainer",
    "LinearProbeHead",
    "RunMatrix",
    "CSV_COLUMNS",
]
