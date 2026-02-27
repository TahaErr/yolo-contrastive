"""yolo-contrastive: Self-supervised pretraining + contrastive learning for YOLOv8+."""

__version__ = "0.2.0"

from .contrastive import NTXentLoss, build_contrastive_loss
from .feature_tap import FeatureTap
from .pipeline import SSLFinetunePipeline, PipelineConfig, auto_train
from .discovery import discover, DatasetInfo, TrainMode
from .exceptions import (
    YoloContrastiveError, FeatureTapError,
    ContrastiveLossError, ConfigError, PatchError,
)

__all__ = [
    "__version__",
    "NTXentLoss", "build_contrastive_loss", "FeatureTap",
    "SSLFinetunePipeline", "PipelineConfig", "auto_train",
    "discover", "DatasetInfo", "TrainMode",
    "YoloContrastiveError", "FeatureTapError",
    "ContrastiveLossError", "ConfigError", "PatchError",
]
