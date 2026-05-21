"""External SSL baselines for fair comparison against DT-SAPS (Faz 5.4).

Each baseline applies a known SSL method's core mechanism to the YOLOv8n
backbone, trained on the same pool with the same protocol — the paper's
Table 4 "DT-SAPS vs SOTA" comparison.

Modules:
    simclr_yolo  — SimCLR (in-batch NT-Xent, global-pooled, no momentum/queue)
"""

from .simclr_yolo import SimCLRYOLOTrainer

__all__ = ["SimCLRYOLOTrainer"]
