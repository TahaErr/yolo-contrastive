"""Test FeatureTap."""

import pytest
import torch
import torch.nn as nn
from yolo_contrastive import FeatureTap, FeatureTapError


class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.pool(x).flatten(1)


def test_layer_selection(device):
    model = SimpleBackbone().to(device)
    tap = FeatureTap(model, min_channels=128, head_class_names=())
    tap.setup(device=device, imgsz=64)
    assert tap.layer_name is not None
    tap.close()


def test_embedding(device):
    model = SimpleBackbone().to(device)
    tap = FeatureTap(model, min_channels=128, head_class_names=())
    tap.setup(device=device, imgsz=64)
    _ = model(torch.randn(4, 3, 64, 64, device=device))
    emb = tap.get_embedding()
    assert emb is not None
    assert emb.dim() == 2 and emb.shape[0] == 4
    assert emb.shape[1] >= 128
    tap.close()


def test_context_manager(device):
    model = SimpleBackbone().to(device)
    with FeatureTap(model, min_channels=128, head_class_names=()) as tap:
        tap.setup(device=device, imgsz=64)
        _ = model(torch.randn(2, 3, 64, 64, device=device))
        assert tap.get_embedding() is not None


def test_min_channels_too_high(device):
    model = SimpleBackbone().to(device)
    with pytest.raises(FeatureTapError):
        FeatureTap(model, min_channels=1024, head_class_names=()).setup(
            device=device, imgsz=64)
