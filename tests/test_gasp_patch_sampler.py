"""Tests for MultiScalePatchSampler — GASP §2.2 + §4 Karar 3.

Bir görüntüden çoklu ölçekte konumsuz yamalar; kasıtlı ölçek-farklılık.
Eşleşme adaylarını öngörülebilir biçimde üretir (per-image gruplama,
deterministik scale × patches_per_scale).
"""

from __future__ import annotations

import math

import pytest
import torch

from yolo_contrastive.gasp import MultiScalePatchSampler


class TestMultiScalePatchSampler:
    def test_attrs(self):
        s = MultiScalePatchSampler(scales=(0.2, 0.5), patches_per_scale=4,
                                    patch_size=32)
        assert s.scales == (0.2, 0.5)
        assert s.patches_per_image == 8

    def test_output_shapes(self):
        s = MultiScalePatchSampler(scales=(0.2, 0.5), patches_per_scale=4,
                                    patch_size=32)
        imgs = torch.randn(2, 3, 320, 320)
        patches, log_scales, image_ids = s(imgs)
        N = s.patches_per_image
        assert patches.shape == (2 * N, 3, 32, 32)
        assert log_scales.shape == (2 * N, 1)
        assert image_ids.shape == (2 * N,)

    def test_image_ids_grouped(self):
        """Per-image gruplama: N tane 0, sonra N tane 1, ..."""
        s = MultiScalePatchSampler(scales=(0.2, 0.5), patches_per_scale=4)
        imgs = torch.randn(2, 3, 320, 320)
        _, _, image_ids = s(imgs)
        N = s.patches_per_image
        expected = torch.cat([torch.zeros(N, dtype=torch.long),
                              torch.ones(N, dtype=torch.long)])
        assert torch.equal(image_ids, expected)

    def test_intentional_scale_diversity(self):
        """Karar 3'ün özü: her görüntüden tam M küçük + M büyük yama."""
        s = MultiScalePatchSampler(scales=(0.2, 0.5), patches_per_scale=4)
        imgs = torch.randn(2, 3, 320, 320)
        _, log_scales, _ = s(imgs)
        # Her görüntüde 4 küçük + 4 büyük; toplam 2 farklı ölçek
        assert len(log_scales.unique()) == 2
        # ilk 4: küçük, sonraki 4: büyük (görüntü 0)
        for i in range(4):
            assert log_scales[i].item() == pytest.approx(math.log(0.2))
        for i in range(4, 8):
            assert log_scales[i].item() == pytest.approx(math.log(0.5))

    def test_deterministic_with_seed(self):
        s1 = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=2,
                                     patch_size=16, seed=42)
        s2 = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=2,
                                     patch_size=16, seed=42)
        imgs = torch.randn(1, 3, 100, 100)
        p1, _, _ = s1(imgs)
        p2, _, _ = s2(imgs)
        assert torch.allclose(p1, p2)

    def test_random_without_seed(self):
        s1 = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=10,
                                     patch_size=16)
        s2 = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=10,
                                     patch_size=16)
        imgs = torch.randn(1, 3, 100, 100)
        p1, _, _ = s1(imgs)
        p2, _, _ = s2(imgs)
        assert not torch.allclose(p1, p2)

    def test_return_positions(self):
        """Smoke-test kapısı: konumlar opsiyonel olarak dönebilir."""
        s = MultiScalePatchSampler(scales=(0.5,), patches_per_scale=4)
        imgs = torch.randn(1, 3, 200, 200)
        patches, log_s, ids, positions = s(imgs, return_positions=True)
        assert positions.shape == (4, 2)
        assert ((positions >= 0) & (positions <= 200)).all().item()

    def test_resize_to_patch_size(self):
        """Yama-alanı kırpılır (scale × img_size), sonra patch_size'a resize."""
        s = MultiScalePatchSampler(scales=(0.5,), patches_per_scale=1,
                                    patch_size=128)
        imgs = torch.randn(1, 3, 64, 64)
        patches, _, _ = s(imgs)
        # Kırpma 32×32 (0.5 × 64), resize 128×128
        assert patches.shape == (1, 3, 128, 128)

    def test_gradient_flows_to_input(self):
        s = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=2)
        imgs = torch.randn(1, 3, 100, 100, requires_grad=True)
        patches, _, _ = s(imgs)
        patches.sum().backward()
        assert imgs.grad is not None
        assert imgs.grad.abs().sum().item() > 0

    def test_rejects_empty_scales(self):
        with pytest.raises(ValueError):
            MultiScalePatchSampler(scales=())

    def test_rejects_invalid_scale(self):
        with pytest.raises(ValueError):
            MultiScalePatchSampler(scales=(1.5,))
        with pytest.raises(ValueError):
            MultiScalePatchSampler(scales=(0.0,))
        with pytest.raises(ValueError):
            MultiScalePatchSampler(scales=(-0.1,))

    def test_rejects_invalid_patches_per_scale(self):
        with pytest.raises(ValueError):
            MultiScalePatchSampler(patches_per_scale=0)
        with pytest.raises(ValueError):
            MultiScalePatchSampler(patches_per_scale=-1)

    def test_rejects_invalid_patch_size(self):
        with pytest.raises(ValueError):
            MultiScalePatchSampler(patch_size=0)

    def test_rejects_wrong_input_shape(self):
        s = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=2)
        with pytest.raises(ValueError):
            s(torch.randn(2, 4, 100, 100))  # 4 kanal
        with pytest.raises(ValueError):
            s(torch.randn(3, 100, 100))     # 3D

    def test_device_consistency(self):
        s = MultiScalePatchSampler(scales=(0.3,), patches_per_scale=2)
        imgs = torch.randn(2, 3, 100, 100)
        patches, log_s, ids = s(imgs)
        assert patches.device == imgs.device
        assert log_s.device == imgs.device
        assert ids.device == imgs.device
