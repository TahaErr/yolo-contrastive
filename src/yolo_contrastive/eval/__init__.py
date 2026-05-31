"""Evaluation utilities — linear probe, leakage check, run matrix."""

from .cross_val import (
    aggregate_cv_results,
    build_cv_matrix,
    load_backbones,
    run_cv_eval,
)
from .linear_probe import LinearProbeTrainer, LinearProbeHead
from .run_matrix import RunMatrix, CSV_COLUMNS

__all__ = [
    "LinearProbeTrainer",
    "LinearProbeHead",
    "RunMatrix",
    "CSV_COLUMNS",
    "aggregate_cv_results",
    "build_cv_matrix",
    "load_backbones",
    "run_cv_eval",
]
