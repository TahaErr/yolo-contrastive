"""Unified exception hierarchy for yolo-contrastive."""


class YoloContrastiveError(Exception):
    """Base exception for all yolo-contrastive errors."""


class FeatureTapError(YoloContrastiveError):
    """Raised when feature extraction / layer selection fails."""


class ContrastiveLossError(YoloContrastiveError):
    """Raised when contrastive loss computation fails."""


class ConfigError(YoloContrastiveError):
    """Raised for invalid configuration values."""


class PatchError(YoloContrastiveError):
    """Raised when model patching fails."""
