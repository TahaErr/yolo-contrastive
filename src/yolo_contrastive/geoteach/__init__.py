"""TERRA — Geometry-as-Teacher pretraining via road-plane residuals (geoteach).

Turns a foundation monocular depth model (Depth-Anything-V2) into a physics
teacher for YOLOv8n: each pool image's road plane is robustly fitted in
uncalibrated inverse-depth space (planarity makes inverse depth affine in
pixel coordinates — no metric calibration needed), and the standardized
plane-residual field — depression / flat / elevation, a taxonomy isomorphic
to pothole / cover / speed bump — supervises the detector as dense ordinal
labels (backbone/neck) and mined polarity boxes (head, via the real TAL
assigner), anchored by the COCO replay loss of
:class:`~yolo_contrastive.anchored.AnchoredJointTrainer`.

Stage 0 (offline label factory):
    depth_cache.py     — Depth-Anything-V2 inference → uint16 PNG cache
    plane_fit.py       — RANSAC + Huber road-plane fit (pure numpy)
    residual_labels.py — z bins, gates, dense label maps, box mining

Stage 1 (anchored joint pretraining):
    heads.py           — DenseOrdinalHead (P3) + fresh 2-class GeoDetectHead
    channel.py         — TerraChannel (AuxChannel) + R5 joint-augmented loader

Kill-gate first (before any training GPU): examples/09_terra_e1_fidelity.py
measures teacher recall / polarity / false-anomaly rates on labeled data.

Importing this package needs torch + numpy only; transformers, cv2 and
ultralytics are imported lazily inside the functions that use them (E2).
"""

from .channel import (
    OrdinalLossConfig,
    TerraAugConfig,
    TerraChannel,
    TerraPoolDataset,
    joint_crop_flip,
    majority_pool_labels,
    ordinal_smoothing_matrix,
    sample_balanced_cells,
    terra_collate,
)
from .depth_cache import DepthCache, run_depth_anything
from .heads import DenseOrdinalHead, GeoDetectHead
from .plane_fit import (
    PlaneFitConfig,
    PlaneFitResult,
    evaluate_surface,
    fit_road_plane,
    standardized_residual,
    trapezoid_mask,
)
from .residual_labels import (
    ANOMALY_CLASSES,
    BOX_DEPRESSION,
    BOX_ELEVATION,
    CLASS_NAMES,
    CLS_D1,
    CLS_D2,
    CLS_E1,
    CLS_E2,
    CLS_F,
    CLS_X,
    GEOBOX_CLASS_NAMES,
    IGNORE_LABEL,
    NUM_CLASSES,
    GeoLabels,
    LabelMapResult,
    MinedBox,
    ResidualLabelConfig,
    bin_residual,
    boxes_to_yolo_lines,
    compute_label_map,
    labels_from_inverse_depth,
    mine_boxes,
    write_yolo_txt,
)

__all__ = [
    # channel / training
    "TerraChannel",
    "TerraPoolDataset",
    "TerraAugConfig",
    "OrdinalLossConfig",
    "terra_collate",
    "joint_crop_flip",
    "majority_pool_labels",
    "ordinal_smoothing_matrix",
    "sample_balanced_cells",
    # heads
    "DenseOrdinalHead",
    "GeoDetectHead",
    # depth cache
    "DepthCache",
    "run_depth_anything",
    # plane fit
    "PlaneFitConfig",
    "PlaneFitResult",
    "fit_road_plane",
    "evaluate_surface",
    "standardized_residual",
    "trapezoid_mask",
    # residual labels
    "ResidualLabelConfig",
    "LabelMapResult",
    "GeoLabels",
    "MinedBox",
    "bin_residual",
    "compute_label_map",
    "mine_boxes",
    "labels_from_inverse_depth",
    "boxes_to_yolo_lines",
    "write_yolo_txt",
    "NUM_CLASSES",
    "IGNORE_LABEL",
    "CLASS_NAMES",
    "GEOBOX_CLASS_NAMES",
    "ANOMALY_CLASSES",
    "CLS_D2",
    "CLS_D1",
    "CLS_F",
    "CLS_E1",
    "CLS_E2",
    "CLS_X",
    "BOX_DEPRESSION",
    "BOX_ELEVATION",
]
