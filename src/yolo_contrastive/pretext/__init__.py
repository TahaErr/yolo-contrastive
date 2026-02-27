"""Self-supervised pretext tasks."""

from .rotation import RotationTask
from .heads import ProjectionHead, PredictionHead

__all__ = ["RotationTask", "ProjectionHead", "PredictionHead"]
