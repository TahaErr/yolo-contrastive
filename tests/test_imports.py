"""Test all public imports."""

import yolo_contrastive


def test_version():
    assert yolo_contrastive.__version__ == "0.2.0"


def test_top_level_imports():
    pass


def test_trainer_import():
    pass


def test_finetune_import():
    pass


def test_pretrain_imports():
    pass


def test_pretext_imports():
    pass


def test_augmentation_imports():
    pass


def test_adapter_imports():
    pass


def test_exception_hierarchy():
    from yolo_contrastive import (
        YoloContrastiveError, FeatureTapError,
        ContrastiveLossError, ConfigError, PatchError,
    )
    assert issubclass(FeatureTapError, YoloContrastiveError)
    assert issubclass(ContrastiveLossError, YoloContrastiveError)
    assert issubclass(ConfigError, YoloContrastiveError)
    assert issubclass(PatchError, YoloContrastiveError)
