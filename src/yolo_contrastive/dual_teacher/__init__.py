"""Dual-teacher distillation framework — DT-SAPS (Faz 5.3, WORK_PLAN_v9 §1.5).

The DT-SAPS framework augments dense SAPS pretraining with distillation from
two teachers: a frozen COCO-pretrained YOLOv8x (supervised knowledge) and an
SSL momentum encoder (self-supervised knowledge). Consensus + disagreement
weighting fuse the two signals.

Modules:
    coco_teacher    — frozen COCO YOLOv8x feature teacher + per-scale adapter
    teacher_cache   — feature cache I/O (planned)
    consensus_loss  — Form B / Form C distillation loss (planned)
    disagreement    — disagreement-based sample weighting (planned)
"""

from .coco_teacher import CocoTeacher

__all__ = ["CocoTeacher"]
