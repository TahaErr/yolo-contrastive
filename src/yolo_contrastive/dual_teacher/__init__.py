"""Dual-teacher distillation framework — DT-SAPS (Faz 5.3, WORK_PLAN_v9 §1.5).

The DT-SAPS framework augments dense SAPS pretraining with distillation from
two teachers: a frozen COCO-pretrained YOLOv8x (supervised knowledge) and an
SSL momentum encoder (self-supervised knowledge). Consensus + disagreement
weighting fuse the two signals.

Modules:
    coco_teacher    — frozen COCO YOLOv8x feature teacher + per-scale adapter
    teacher_cache   — FP16 npz feature cache I/O (§2.4)
    disagreement    — per-position cosine disagreement weighting (§10.29)
    consensus_loss  — Form B / Form C distillation loss (planned)
"""

from .coco_teacher import CocoTeacher
from .teacher_cache import TeacherCache
from .disagreement import DisagreementWeighter, cosine_disagreement

__all__ = ["CocoTeacher", "TeacherCache", "DisagreementWeighter", "cosine_disagreement"]
