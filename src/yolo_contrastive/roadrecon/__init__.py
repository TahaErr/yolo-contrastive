"""roadrecon — pure, self-contained reconstruction-based detection pretraining (B2/M2/M3).

The appearance half of the "beat COCO without COCO" program: train our OWN road
reconstruction net from scratch on the unlabeled RGB pool (no COCO, no external
model), then

    * transfer its backbone as a representation init (**M2**), and
    * mine reconstruction-anomaly boxes and detection-pretrain a scratch detector on
      them (**M3**), evaluated whole-detector against the ``coco`` / ``scratch``
      baselines under LOSO.

Modules:
    recon_net.py     — RoadReconNet (scratch encoder + light decoder) + ReconDecoder
    reconstructor.py — RoadReconstructor (standalone denoising/inpainting pretrainer; B2)
    mining.py        — offline anomaly-label factory (recon error → YOLO boxes)
    channel.py       — RoadReconChannel (AuxChannel; content-pressuring aux for M3)
    eval_runner.py   — full_transplant_detection_runner (R8 whole-detector LOSO eval)

Importing this package needs torch + numpy only; ultralytics is imported lazily
inside RoadReconNet, and cv2 lazily inside the mining / loader functions (E2).
"""

from .recon_net import ReconDecoder, RoadReconNet, build_scratch_detector
from .reconstructor import RoadReconstructor, load_reconstructor
from .channel import RoadReconChannel
from .mining import (
    AnomalyMineConfig,
    box_iou_xywh,
    mine_anomaly_labels,
    mine_image_boxes,
    mining_fidelity,
)
from .eval_runner import detection_runners, full_transplant_detection_runner

__all__ = [
    "RoadReconNet",
    "ReconDecoder",
    "build_scratch_detector",
    "RoadReconstructor",
    "load_reconstructor",
    "RoadReconChannel",
    "AnomalyMineConfig",
    "mine_anomaly_labels",
    "mine_image_boxes",
    "mining_fidelity",
    "box_iou_xywh",
    "full_transplant_detection_runner",
    "detection_runners",
]
