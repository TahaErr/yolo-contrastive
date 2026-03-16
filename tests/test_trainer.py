"""Test trainer class hierarchy."""

from ultralytics.models.yolo.detect.train import DetectionTrainer
from yolo_contrastive.trainer import ContrastiveDetectionTrainer
from yolo_contrastive.finetune import FinetuneDetectionTrainer


def test_contrastive_inherits():
    assert issubclass(ContrastiveDetectionTrainer, DetectionTrainer)


def test_finetune_inherits():
    assert issubclass(FinetuneDetectionTrainer, DetectionTrainer)


def test_has_methods():
    assert hasattr(ContrastiveDetectionTrainer, "make_view2")
    assert hasattr(ContrastiveDetectionTrainer, "_compute_pretext")
    assert hasattr(ContrastiveDetectionTrainer, "_compute_cl")
    assert hasattr(ContrastiveDetectionTrainer, "cleanup")
