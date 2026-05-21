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


def aggressive(imgsz: int = 640) -> AugmentationPipeline:
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


def mocov3_v3_v2(imgsz: int = 640) -> AugmentationPipeline:
    """MoCo-v3 photometric augmentation (WORK_PLAN_v9 §5.3 — DT-SAPS
    aug_photometric="mocov3"). Color jitter + grayscale + blur, the
    standard MoCo-v3 photometric set. Geometric augmentation is handled
    separately by dense/spatial_aug.py; this preset is photometric-only,
    applied as a post-step on each view."""
    return AugmentationPipeline([
        RandomColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1, p=0.8),
        RandomGrayscale(p=0.2),
        RandomGaussianBlur(kernel_size=max(3, imgsz // 64 * 2 + 1), p=0.5),
    ])


def dino_v1(imgsz: int = 640) -> AugmentationPipeline:
    """DINO photometric augmentation (WORK_PLAN_v9 §5.3 — DT-SAPS
    aug_photometric="dino"). Color jitter + grayscale + stronger blur +
    solarize, the standard DINO photometric set. Photometric-only, applied
    as a post-step on each view (geometric handled by spatial_aug.py)."""
    return AugmentationPipeline([
        RandomColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1, p=0.8),
        RandomGrayscale(p=0.2),
        RandomGaussianBlur(kernel_size=max(3, imgsz // 64 * 2 + 1), p=0.5),
        RandomSolarize(threshold=0.5, p=0.2),
    ])


PRESETS = {
    "simclr_v1": simclr_v1,
    "simclr_v2": simclr_v2,
    "byol": byol,
    "aggressive": aggressive,
    "mocov3_v3_v2": mocov3_v3_v2,
    "dino_v1": dino_v1,
}


def build_pipeline(name: str, **kwargs) -> AugmentationPipeline:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS.keys())}")
    return PRESETS[name](**kwargs)
