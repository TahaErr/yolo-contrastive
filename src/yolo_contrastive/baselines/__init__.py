"""External SSL baselines for fair comparison against DT-SAPS (Faz 5.4).

Each baseline applies a known SSL method's core mechanism to the YOLOv8n
backbone, trained on the same pool with the same protocol — the paper's
Table 4 "DT-SAPS vs SOTA" comparison.

Modules:
    simclr_yolo  — SimCLR (in-batch NT-Xent, global-pooled, no momentum/queue)
    moco_v3      — MoCo-v3 (momentum encoder + predictor, no queue, symmetric InfoNCE)
    comad_yolo   — CoMAD (3 SSL teachers, asymmetric masking, consensus gating)
"""

from .simclr_yolo import SimCLRYOLOTrainer
from .moco_v3 import MoCoV3YOLOTrainer
from .comad_yolo import CoMADYOLOTrainer

__all__ = ["SimCLRYOLOTrainer", "MoCoV3YOLOTrainer", "CoMADYOLOTrainer"]
