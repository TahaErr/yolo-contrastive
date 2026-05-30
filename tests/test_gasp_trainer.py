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


def _mock_yolo_like(channels=(32, 64, 128)) -> nn.Module:
    """23-layer Sequential mimicking YOLOv8 FPN — P3/P4/P5 at layers 15/18/21.

    test_moco_v3.py'deki mock paterniyle uyumlu — MultiScaleFeatureTap'in
    "P5" hook'unu doğru katmana takabilmesi için.
    """
    p3, p4, p5 = channels
    layers = []
    layers.append(nn.Conv2d(3, p3, 3, stride=2, padding=1))    # 0 /2
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))         # 1-5
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))   # 6 /4
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))         # 7-11
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))   # 12 /8
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 13
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 14
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 15 P3
    layers.append(nn.Conv2d(p3, p4, 3, stride=2, padding=1))   # 16 /16
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))             # 17
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))             # 18 P4
    layers.append(nn.Conv2d(p4, p5, 3, stride=2, padding=1))   # 19 /32
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))             # 20
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))             # 21 P5
    return nn.Sequential(*layers)


def _make_trainer(**kw):
    defaults = dict(
        model=_mock_yolo_like(), feat_dim=128,  # P5 = channels[2] = 128 in mock
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
        # Mock encoder P5 = channels[2] = 128 (yeni mock _mock_yolo_like()).
        # Önceki test 256 bekliyordu — eski (out_channels=256) mock için doğruydu.
        assert f_online.shape == (8, 128)
        assert f_ema.shape == (8, 128)
        assert not f_ema.requires_grad
        t.cleanup()

    def test_step_returns_loss_and_components(self):
        t = _make_trainer()
        out = t._step(torch.randn(2, 3, 128, 128))
        assert out["loss"].dim() == 0
        assert out["loss"].requires_grad
        # VICReg eklendi: L_var, L_cov da geri döner
        for key in ["L_ctrl", "L_nat", "L_var", "L_cov", "n_pairs"]:
            assert key in out, f"_step output eksik anahtar: {key}"
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


class TestGASPProjectorV7:
    """v7: projeksiyon başlığı — kayıplar z'de, backbone f projektörden önce."""

    def test_projection_head_shape(self):
        from yolo_contrastive.gasp.transform import ProjectionHead
        proj = ProjectionHead(in_dim=128, hidden_dim=64, out_dim=96)
        out = proj(torch.randn(8, 128))
        assert out.shape == (8, 96)

    def test_default_no_projector_is_v6(self):
        """use_projector=False (default) → projektör yok, T backbone dim'inde."""
        t = _make_trainer()
        assert t.use_projector is False
        assert t.projector is None
        # _project kimlik olmalı
        f = torch.randn(4, 128)
        assert torch.equal(t._project(f), f)

    def test_projector_construction_and_dims(self):
        t = _make_trainer(use_projector=True, proj_hidden_dim=64, proj_dim=96)
        assert t.use_projector is True
        assert t.projector is not None
        # backbone f hâlâ feat_dim (128); projeksiyon z proj_dim (96)
        patches = torch.randn(8, 3, 64, 64)
        f = t._encode(patches, use_ema=False)
        assert f.shape[1] == 128, "backbone feat_dim değişmemeli (downstream/eff_rank)"
        z = t._project(f)
        assert z.shape[1] == 96, "projeksiyon proj_dim'e eşlemeli"

    def test_step_runs_with_projector(self):
        t = _make_trainer(use_projector=True, proj_hidden_dim=64, proj_dim=96,
                          lambda_var=1.0, lambda_cov=1.0, lambda_iso=0.5)
        out = t._step(torch.randn(2, 3, 128, 128))
        for key in ("loss", "L_ctrl", "L_nat", "L_var", "L_cov", "L_iso"):
            assert key in out
        assert torch.isfinite(out["loss"]), "v7 step kaybı sonlu olmalı"
        assert out["loss"].requires_grad, "gradient akışı (projektör + backbone)"

    def test_projector_grads_flow_to_backbone(self):
        """z üzerindeki kayıp backbone'a gradient akıtmalı (decouple ≠ kopuk)."""
        t = _make_trainer(use_projector=True, proj_hidden_dim=64, proj_dim=96)
        out = t._step(torch.randn(2, 3, 128, 128))
        out["loss"].backward()
        bb_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in t.model.parameters())
        pj_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in t.projector.parameters())
        assert bb_grad, "backbone gradient almalı (z=proj(f), f backbone'dan)"
        assert pj_grad, "projektör gradient almalı"
