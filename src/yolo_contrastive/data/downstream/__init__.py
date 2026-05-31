"""Downstream eval pool assembly from Roboflow sources (WORK_PLAN_v9 §13.4)."""

from .allocate import resolve_target, water_fill_allocation
from .prepare import (
    build_selection_manifest,
    consolidate_to_train,
    prepare_downstream,
    read_selection_manifest,
    verify_single_label,
)
from .sources import load_sources
from .splits import build_cv_splits, build_holdout_split

__all__ = [
    "build_cv_splits",
    "build_holdout_split",
    "build_selection_manifest",
    "consolidate_to_train",
    "load_sources",
    "prepare_downstream",
    "read_selection_manifest",
    "resolve_target",
    "verify_single_label",
    "water_fill_allocation",
]
