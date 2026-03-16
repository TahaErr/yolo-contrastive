"""Test augmentation system."""

import pytest
from yolo_contrastive.augmentations import (
    AugmentationPipeline, build_pipeline, list_augmentations,
)


@pytest.mark.parametrize("name", ["simclr_v1", "simclr_v2", "byol", "aggressive"])
def test_preset(name, batch_img):
    pipe = build_pipeline(name)
    out = pipe(batch_img)
    assert out.shape == batch_img.shape
    assert out.min() >= -0.01 and out.max() <= 1.01


def test_unknown_preset():
    with pytest.raises(KeyError):
        build_pipeline("nonexistent")


def test_registry_not_empty():
    assert len(list_augmentations()) > 0


def test_empty_pipeline(batch_img):
    pipe = AugmentationPipeline([])
    assert (pipe(batch_img) == batch_img).all()
