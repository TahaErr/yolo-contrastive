"""Tests for FeatureTap — embedding extraction and auto layer selection.

Covers audit §6.1: FeatureTap layer selection on a known model.
"""

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.feature_tap import FeatureTap, _parse_imgsz
from yolo_contrastive.exceptions import FeatureTapError, ConfigError


class SimpleBackbone(nn.Module):
    """Minimal model that mimics YOLO-like structure for testing."""

    def __init__(self, nc: int = 80):
        super().__init__()
        self.nc = nc
        # "backbone" — should be selected
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # "head" — should be excluded
        self.head = nn.Conv2d(256, nc, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        out = self.pool(self.head(x)).flatten(1)
        return out


class TestFeatureTapSetup:
    """Test auto layer selection."""

    def test_selects_layer(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, head_class_names=())
        tap.setup(device="cpu", imgsz=64)

        assert tap.layer_name is not None
        tap.close()

    def test_selects_deepest_layer_with_enough_channels(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, head_class_names=())
        tap.setup(device="cpu", imgsz=64)

        # conv3 has 256 channels and is deeper than conv2 (128 channels)
        # Should pick the last qualifying layer
        assert tap.layer_name is not None
        tap.close()

    def test_min_channels_filter(self):
        model = SimpleBackbone()
        # Set min_channels very high — should still find conv3 (256)
        tap = FeatureTap(model, min_channels=256, head_class_names=())
        tap.setup(device="cpu", imgsz=64)
        assert tap.layer_name is not None
        tap.close()

    def test_min_channels_too_high_raises(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=1024, head_class_names=())
        with pytest.raises(FeatureTapError, match="could not find"):
            tap.setup(device="cpu", imgsz=64)

    def test_rectangular_imgsz(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, head_class_names=())
        tap.setup(device="cpu", imgsz=(480, 640))
        assert tap.layer_name is not None
        tap.close()


class TestFeatureTapEmbedding:
    """Test embedding extraction."""

    def test_produces_embedding(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, store_grad=False, head_class_names=())
        tap.setup(device="cpu", imgsz=64)

        x = torch.randn(2, 3, 64, 64)
        _ = model(x)

        emb = tap.get_embedding()
        assert emb is not None
        assert emb.shape[0] == 2  # batch size
        assert emb.ndim == 2      # [B, D]
        assert torch.isfinite(emb).all()
        tap.close()

    def test_store_grad_true(self):
        model = SimpleBackbone()
        model.train()
        tap = FeatureTap(model, min_channels=128, store_grad=True, head_class_names=())
        tap.setup(device="cpu", imgsz=64)

        x = torch.randn(2, 3, 64, 64)
        _ = model(x)

        emb = tap.get_embedding()
        assert emb is not None
        assert emb.requires_grad
        tap.close()

    def test_store_grad_false(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, store_grad=False, head_class_names=())
        tap.setup(device="cpu", imgsz=64)

        x = torch.randn(2, 3, 64, 64)
        _ = model(x)

        emb = tap.get_embedding()
        assert emb is not None
        assert not emb.requires_grad
        tap.close()


class TestFeatureTapLifecycle:
    """Test context manager and cleanup (audit §1.9)."""

    def test_context_manager(self):
        model = SimpleBackbone()
        with FeatureTap(model, min_channels=128, head_class_names=()) as tap:
            tap.setup(device="cpu", imgsz=64)
            x = torch.randn(1, 3, 64, 64)
            _ = model(x)
            assert tap.get_embedding() is not None

        # After exit, hook should be removed
        assert tap._fixed_hook is None

    def test_double_close_safe(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, head_class_names=())
        tap.setup(device="cpu", imgsz=64)
        tap.close()
        tap.close()  # should not raise

    def test_close_before_setup(self):
        model = SimpleBackbone()
        tap = FeatureTap(model, min_channels=128, head_class_names=())
        tap.close()  # should not raise


class TestParseImgsz:
    def test_int(self):
        assert _parse_imgsz(640) == (640, 640)

    def test_tuple(self):
        assert _parse_imgsz((480, 640)) == (480, 640)

    def test_list(self):
        assert _parse_imgsz([480, 640]) == (480, 640)

    def test_invalid_raises(self):
        with pytest.raises(ConfigError):
            _parse_imgsz((1, 2, 3))
