"""Dense + multi-scale contrastive learning (Faz 1+2 of WORK_PLAN_v3)."""

from .multi_scale_tap import (
    MultiScaleFeatureTap,
    YOLOV8_FPN_LAYERS,
    YOLOV8_FPN_STRIDES,
)
from .queue import FeatureQueue, combine_queues
from .momentum_encoder import MomentumEncoder
from .spatial_aug import SpatialTwoViewAugmentation, TwoView
from .dense_loss import dense_ntxent_loss, coords_to_feature_map
from .multi_scale_loss import multi_scale_dense_loss
from .projection import MultiScaleProjectionHead, infer_in_channels

__all__ = [
    "MultiScaleFeatureTap",
    "YOLOV8_FPN_LAYERS",
    "YOLOV8_FPN_STRIDES",
    "FeatureQueue",
    "combine_queues",
    "MomentumEncoder",
    "SpatialTwoViewAugmentation",
    "TwoView",
    "dense_ntxent_loss",
    "coords_to_feature_map",
    "multi_scale_dense_loss",
    "MultiScaleProjectionHead",
    "infer_in_channels",
]
