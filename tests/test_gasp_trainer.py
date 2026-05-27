"""Tests for GASPTrainer — GASP'ın ana eğitim sınıfı (§2-§4 birleşimi)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from yolo_contrastive.gasp import GASPTrainer


def _mock_yolo_like(out_channels: int = 256) -> nn.Module:
    """[B,3,H,W] → [B, C, H/32, W/32] mock encoder (YOLO P5 mimari)."""
    class MockYOLO(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            )

        def forward(self, x):
            return self.layers(x)

    return MockYOLO()


def _make_trainer(**kw):
    defaults = dict(
        model=_mock_yolo_like(256), feat_dim=256,
        scales=(0.3, 0.6), patches_per_scale=4, patch_size=64,
        target_patch_size=64, alpha=1.0, momentum=0.99,
        similarity_threshold=0.5, T_hidden_dim=16,
        imgsz=128, device="cpu",
    )
    defaults.update(kw)
    return GASPTrainer(**defaults)


class TestGASPTrainer:
    def test_construction(self):
        t = _make_trainer()
        assert t.alpha == 1.0
        assert hasattr(t, "ema_model")
        assert all(not p.requires_grad for p in t.ema_model.parameters())
        t.cleanup()

    def test_encode_online_and_ema(self):
        t = _make_trainer()
        patches = torch.randn(8, 3, 64, 64)
        f_online = t._encode(patches, use_ema=False)
        f_ema = t._encode(patches, use_ema=True)
        assert f_online.shape == (8, 256)
        assert f_ema.shape == (8, 256)
        assert not f_ema.requires_grad
        t.cleanup()

    def test_step_returns_loss_and_components(self):
        t = _make_trainer()
        out = t._step(torch.randn(2, 3, 128, 128))
        assert out["loss"].dim() == 0
        assert out["loss"].requires_grad
        assert "L_ctrl" in out and "L_nat" in out and "n_pairs" in out
        t.cleanup()

    def test_gradient_flow(self):
        t = _make_trainer()
        out = t._step(torch.randn(2, 3, 128, 128))
        out["loss"].backward()
        g_model = sum(
            p.grad.abs().sum().item()
            for p in t.model.parameters() if p.grad is not None
        )
        g_T = sum(
            p.grad.abs().sum().item()
            for p in t.transform.parameters() if p.grad is not None
        )
        assert g_model > 0
        assert g_T > 0
        # EMA grad'lar None ya da 0
        g_ema = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0)
            for p in t.ema_model.parameters()
        )
        assert g_ema == 0
        t.cleanup()

    def test_ema_update_changes_params(self):
        t = _make_trainer()
        ema_before = next(t.ema_model.parameters()).clone()
        with torch.no_grad():
            for p in t.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        t._ema_update()
        ema_after = next(t.ema_model.parameters())
        assert not torch.allclose(ema_before, ema_after)
        t.cleanup()

    def test_rejects_invalid_momentum(self):
        with pytest.raises(ValueError):
            _make_trainer(momentum=1.5)
        with pytest.raises(ValueError):
            _make_trainer(momentum=-0.1)

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError):
            _make_trainer(alpha=-0.1)

    def test_mini_train_loop(self):
        """Uçtan uca: küçük sentetik dataset üzerinde 2 epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
                Image.fromarray(arr).save(f"{tmpdir}/img_{i:03d}.jpg")
            t = _make_trainer(similarity_threshold=-1.0, T_hidden_dim=8,
                              patches_per_scale=2, patch_size=32,
                              target_patch_size=32, imgsz=64)
            out_path = os.path.join(tmpdir, "gasp_test.pt")
            t.train(
                images_dir=tmpdir, epochs=2, batch_size=4, lr=1e-3,
                weight_decay=0.0, warmup_epochs=1, num_workers=0,
                output=out_path, save_every=2, print_every=10,
            )
            assert os.path.exists(out_path.replace(".pt", "_ep2.pt"))
            assert len(t.loss_history) == 2
            assert "T_identity_dist" in t.loss_history[0]
            t.cleanup()
