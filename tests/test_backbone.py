"""Test backbone utils."""

import os
import tempfile
import shutil
import pytest
from yolo_contrastive.pretrain import save_backbone, load_backbone, freeze_backbone, unfreeze_all


@pytest.fixture
def yolo_pair():
    from ultralytics import YOLO
    m1 = YOLO("yolov8n.pt").model
    m2 = YOLO("yolov8n.pt").model
    return m1, m2


def test_save_load(yolo_pair):
    m1, m2 = yolo_pair
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "bb.pt")
        result = save_backbone(m1, path, epoch=10, extra={"test": True})
        assert result == path and os.path.exists(path)

        n = load_backbone(m2, path, backbone_only=True, verbose=False)
        assert n > 0
    finally:
        shutil.rmtree(tmp)


def test_freeze_unfreeze():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt").model
    total = sum(1 for _ in model.parameters())

    freeze_backbone(model, num_layers=10, verbose=False)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable < total

    unfreeze_all(model, verbose=False)
    trainable_after = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable_after == total
