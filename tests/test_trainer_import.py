"""Trainer import test — skips gracefully if ultralytics is not installed."""

import pytest


def test_trainer_import():
    pytest.importorskip("ultralytics", reason="ultralytics not installed")
    from yolo_contrastive.trainer import ContrastiveDetectionTrainer
    assert ContrastiveDetectionTrainer is not None
