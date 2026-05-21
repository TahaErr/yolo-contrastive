"""Dual-teacher distillation framework — DT-SAPS (Faz 5.3, WORK_PLAN_v9 §1.5).

The DT-SAPS framework augments dense SAPS pretraining with distillation from
two teachers: a frozen COCO-pretrained YOLOv8x (supervised knowledge) and a
frozen SSL teacher — the Faz 5.1 pure-SAPS winner (self-supervised knowledge).
Consensus + disagreement weighting fuse the two signals.

Modules:
    coco_teacher          — frozen YOLO feature teacher + per-scale adapter
    teacher_cache         — FP16 npz feature cache I/O (§2.4)
    disagreement          — per-position cosine disagreement weighting (§10.29)
    consensus_loss        — Form B (learned weighted L2) + Form C (CWD dual KL) (§10.28)
    dual_teacher_trainer  — DT-SAPS trainer, composition over DenseSSLPretrainer (§10.30)
"""

from .coco_teacher import CocoTeacher
from .teacher_cache import TeacherCache
from .disagreement import DisagreementWeighter, cosine_disagreement
from .consensus_loss import ConsensusLoss
from .dual_teacher_trainer import DualTeacherTrainer

__all__ = [
    "CocoTeacher",
    "TeacherCache",
    "DisagreementWeighter",
    "cosine_disagreement",
    "ConsensusLoss",
    "DualTeacherTrainer",
]
