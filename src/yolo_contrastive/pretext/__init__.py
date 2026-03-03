"""Self-supervised pretext tasks."""

from .base import (
    BasePretextTask,
    register_task,
    get_task,
    list_tasks,
)
from .rotation import RotationTask
from .tasks import (
    SolarizationTask,
    ColorPermutationTask,
    PatchShuffleTask,
    BlurPredictionTask,
)
from .freq_band import FrequencyBandPrediction
from .composite import CompositeTask
from .heads import ProjectionHead, PredictionHead

__all__ = [
    "BasePretextTask", "register_task", "get_task", "list_tasks",
    "RotationTask",
    "SolarizationTask", "ColorPermutationTask",
    "PatchShuffleTask", "BlurPredictionTask",
    "FrequencyBandPrediction",
    "CompositeTask",
    "ProjectionHead", "PredictionHead",
]
