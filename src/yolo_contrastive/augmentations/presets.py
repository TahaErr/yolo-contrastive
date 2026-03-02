"""Ready-made augmentation pipeline presets."""

from .registry import AugmentationPipeline
from .geometric import RandomHorizontalFlip, RandomRotation
from .color import RandomColorJitter, RandomGrayscale, RandomSolarize
from .filtering import RandomGaussianBlur, GaussianNoise
from .erasing import RandomCutout


def simclr_v1(imgsz: int = 640) -> AugmentationPipeline:
    """SimCLR paper augmentation pipeline."""
    return AugmentationPipeline([
        RandomHorizontalFlip(p=0.5),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        RandomGrayscale(p=0.2),
        RandomGaussianBlur(kernel_size=max(3, imgsz // 64 * 2 + 1), p=0.5),
    ])


def simclr_v2(imgsz: int = 640) -> AugmentationPipeline:
    """SimCLR v2 — with solarize."""
    return AugmentationPipeline([
        RandomHorizontalFlip(p=0.5),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        RandomGrayscale(p=0.2),
        RandomGaussianBlur(kernel_size=max(3, imgsz // 64 * 2 + 1), p=0.5),
        RandomSolarize(threshold=0.5, p=0.2),
    ])


def byol(imgsz: int = 640) -> AugmentationPipeline:
    """BYOL paper augmentation pipeline."""
    return AugmentationPipeline([
        RandomHorizontalFlip(p=0.5),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        RandomGrayscale(p=0.2),
        RandomGaussianBlur(kernel_size=max(3, imgsz // 64 * 2 + 1), p=1.0),
        RandomSolarize(threshold=0.5, p=0.0),  # not used in view1
    ])


def aggressive() -> AugmentationPipeline:
    """Aggressive augmentation — for small datasets."""
    return AugmentationPipeline([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15, p=0.3),
        RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.15, p=0.9),
        RandomGrayscale(p=0.3),
        RandomGaussianBlur(p=0.5),
        RandomSolarize(threshold=0.5, p=0.2),
        RandomCutout(num_holes=1, max_h=48, max_w=48, p=0.3),
        GaussianNoise(std=0.03, p=0.2),
    ])


PRESETS = {
    "simclr_v1": simclr_v1,
    "simclr_v2": simclr_v2,
    "byol": byol,
    "aggressive": aggressive,
}


def build_pipeline(name: str, **kwargs) -> AugmentationPipeline:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS.keys())}")
    return PRESETS[name](**kwargs)
