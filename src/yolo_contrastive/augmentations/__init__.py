"""Modular augmentation system for contrastive learning."""

from .registry import (
    BaseAugmentation, PerImageAugmentation, AugmentationPipeline,
    register, get_augmentation, list_augmentations,
)
from .presets import build_pipeline, PRESETS

# Explicit imports instead of star imports
from .geometric import (
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90,
    RandomRotation, RandomAffine,
)
from .color import (
    RandomBrightness, RandomContrast, RandomSaturation, RandomHue,
    RandomColorJitter, RandomGrayscale, RandomSolarize, RandomPosterize,
    RandomEqualize,
)
from .erasing import RandomCutout, RandomErasing, GridMask
from .filtering import RandomGaussianBlur, GaussianNoise, RandomSharpen

__all__ = [
    "BaseAugmentation", "PerImageAugmentation", "AugmentationPipeline",
    "register", "get_augmentation", "list_augmentations",
    "build_pipeline", "PRESETS",
    "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation90",
    "RandomRotation", "RandomAffine",
    "RandomBrightness", "RandomContrast", "RandomSaturation", "RandomHue",
    "RandomColorJitter", "RandomGrayscale", "RandomSolarize", "RandomPosterize",
    "RandomEqualize",
    "RandomCutout", "RandomErasing", "GridMask",
    "RandomGaussianBlur", "GaussianNoise", "RandomSharpen",
]
