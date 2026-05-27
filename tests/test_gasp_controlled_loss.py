"""Tests for controlled_loss — GASP §2.3 L_kontrollü."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_contrastive.gasp import (
    ScaleEquivariantTransform,
    MultiScalePatchSampler,
    controlled_loss,
)


def _mock_encoder(D: int = 64) -> nn.Module:
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, D, 3, padding=1)
            self.fc = nn.Linear(D, D)

        def forward(self, x):
            x = self.conv(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    return Enc()


class TestControlledLoss:
    def test_returns_dict(self):
        out = controlled_loss(
            torch.randn(4, 3, 64, 64),
            _mock_encoder(D=64),
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
            scale_a=0.5, scale_b=1.0, target_patch_size=32,
        )
        assert set(out.keys()) == {"loss", "mse", "log_ratio"}
        assert out["loss"].dim() == 0
        assert out["loss"].item() > 0

    def test_log_ratio_correct(self):
        out = controlled_loss(
            torch.randn(4, 3, 64, 64),
            _mock_encoder(D=64),
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
            scale_a=0.5, scale_b=1.0, target_patch_size=32,
        )
        assert out["log_ratio"] == pytest.approx(math.log(2.0), abs=1e-4)

    def test_gradient_flows_to_encoder_and_T(self):
        encoder = _mock_encoder(D=64)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        out = controlled_loss(
            torch.randn(4, 3, 64, 64), encoder, T,
            scale_a=0.5, scale_b=1.0, target_patch_size=32,
        )
        out["loss"].backward()
        g_enc = sum(
            p.grad.abs().sum().item()
            for p in encoder.parameters() if p.grad is not None
        )
        g_T = sum(
            p.grad.abs().sum().item()
            for p in T.parameters() if p.grad is not None
        )
        assert g_enc > 0
        assert g_T > 0

    def test_different_ratios_give_different_losses(self):
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        with torch.no_grad():
            T.gen[-1].weight.normal_(0, 0.3)
            T.gen[-1].bias.normal_(0, 0.3)
        encoder = _mock_encoder(D=64)
        torch.manual_seed(0)
        patches = torch.randn(8, 3, 64, 64)
        o_ab = controlled_loss(patches, encoder, T,
                                scale_a=0.5, scale_b=1.0,
                                target_patch_size=32, augment=False)
        o_ba = controlled_loss(patches, encoder, T,
                                scale_a=1.0, scale_b=0.5,
                                target_patch_size=32, augment=False)
        assert not torch.allclose(o_ab["loss"], o_ba["loss"])

    def test_augment_can_be_disabled(self):
        out = controlled_loss(
            torch.randn(4, 3, 64, 64),
            _mock_encoder(D=64),
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
            scale_a=0.5, scale_b=1.0, target_patch_size=32,
            augment=False,
        )
        assert out["loss"].dim() == 0

    def test_rejects_invalid_patches(self):
        with pytest.raises(ValueError):
            controlled_loss(
                torch.randn(3, 64, 64),
                _mock_encoder(), ScaleEquivariantTransform(feat_dim=64),
                scale_a=0.5, scale_b=1.0, target_patch_size=32,
            )

    def test_rejects_invalid_scales(self):
        with pytest.raises(ValueError):
            controlled_loss(
                torch.randn(4, 3, 64, 64), _mock_encoder(),
                ScaleEquivariantTransform(feat_dim=64),
                scale_a=0, scale_b=1.0, target_patch_size=32,
            )

    def test_rejects_invalid_target_size(self):
        with pytest.raises(ValueError):
            controlled_loss(
                torch.randn(4, 3, 64, 64), _mock_encoder(),
                ScaleEquivariantTransform(feat_dim=64),
                scale_a=0.5, scale_b=1.0, target_patch_size=0,
            )

    def test_integration_with_sampler(self):
        """Uçtan uca: sampler → controlled_loss."""
        sampler = MultiScalePatchSampler(
            scales=(0.5,), patches_per_scale=4, patch_size=32,
        )
        imgs = torch.randn(2, 3, 320, 320)
        patches, _, _ = sampler(imgs)
        out = controlled_loss(
            patches, _mock_encoder(D=64),
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=16),
            scale_a=0.5, scale_b=1.0, target_patch_size=32,
        )
        assert out["loss"].dim() == 0
