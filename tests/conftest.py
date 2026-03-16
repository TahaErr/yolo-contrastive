"""Shared pytest fixtures."""

import os
import sys
import tempfile
import shutil

import pytest
import torch
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_img(device):
    """4x3x64x64 random image batch."""
    return torch.rand(4, 3, 64, 64, device=device)


@pytest.fixture
def feat(device):
    """4x256 random feature batch with grad."""
    return torch.randn(4, 256, device=device, requires_grad=True)


@pytest.fixture
def dummy_images():
    """Temp dir with 8 dummy JPEG images."""
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        for i in range(8):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(tmp, f"img_{i:03d}.jpg"))
    except ImportError:
        pass
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def yolo_model(device):
    """YOLOv8n model on device."""
    from ultralytics import YOLO
    return YOLO("yolov8n.pt").model.to(device)
