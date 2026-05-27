"""Tests for natural_loss — GASP §2.3 L_doğal."""

from __future__ import annotations

import pytest
import torch

from yolo_contrastive.gasp import (
    ScaleEquivariantTransform,
    NaturalPairMatcher,
    natural_loss,
)


def _matched_setup():
    """4 yama: img 0 (s=0, s=1), img 1 (s=0, s=1). Features ikişerli aynı."""
    online = torch.randn(4, 64, requires_grad=True)
    ema = torch.tensor([
        [1.0, 0.0] + [0.0] * 62,
        [1.0, 0.0] + [0.0] * 62,
        [0.0, 1.0] + [0.0] * 62,
        [0.0, 1.0] + [0.0] * 62,
    ])
    log_scales = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    image_ids = torch.tensor([0, 0, 1, 1])
    return online, ema, log_scales, image_ids


class TestNaturalLoss:
    def test_no_match_returns_zero(self):
        m = NaturalPairMatcher(similarity_threshold=0.7)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        online = torch.randn(2, 64, requires_grad=True)
        ema = online.detach().clone()
        # Farklı görüntüler → eşleşme yok
        out = natural_loss(online, ema,
                            torch.tensor([[0.0], [1.0]]),
                            torch.tensor([0, 1]), m, T)
        assert out["loss"].item() == 0.0
        assert out["n_pairs"] == 0

    def test_matched_pairs_produce_loss(self):
        online, ema, log_scales, image_ids = _matched_setup()
        m = NaturalPairMatcher(similarity_threshold=0.7)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        out = natural_loss(online, ema, log_scales, image_ids, m, T)
        assert out["n_pairs"] == 2
        assert out["loss"].dim() == 0

    def test_gradient_flows_to_online_and_T(self):
        online, ema, log_scales, image_ids = _matched_setup()
        m = NaturalPairMatcher(similarity_threshold=0.7)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        out = natural_loss(online, ema, log_scales, image_ids, m, T)
        out["loss"].backward()
        assert online.grad is not None
        assert online.grad.abs().sum().item() > 0
        g_T = sum(
            p.grad.abs().sum().item()
            for p in T.parameters() if p.grad is not None
        )
        assert g_T > 0

    def test_ema_features_get_no_grad(self):
        """ema_features eşleştirme için; gradyan o yola akmamalı."""
        online, ema, log_scales, image_ids = _matched_setup()
        m = NaturalPairMatcher(similarity_threshold=0.7)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        out = natural_loss(online, ema, log_scales, image_ids, m, T)
        out["loss"].backward()
        # ema requires_grad=False (default) → grad attr None
        assert ema.grad is None

    def test_zero_loss_when_T_identity_and_features_equal(self):
        """T=identity + aynı features → loss ≈ 0 (simetrik tutarlılık doğru)."""
        online = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                              requires_grad=True)
        ema = online.detach().clone()
        m = NaturalPairMatcher(similarity_threshold=-1.0)
        T = ScaleEquivariantTransform(feat_dim=3, hidden_dim=4)
        with torch.no_grad():
            for p in T.gen.parameters():
                p.zero_()
        out = natural_loss(online, ema,
                            torch.tensor([[0.0], [1.0]]),
                            torch.tensor([0, 0]), m, T)
        assert out["loss"].item() < 1e-6

    def test_zero_loss_does_not_break_total_gradient(self):
        """Mod A: L_nat=0 olduğunda L_total = L_ctrl + 0 hala gradyan akıtmalı."""
        m = NaturalPairMatcher(similarity_threshold=0.7)
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        online = torch.randn(2, 64, requires_grad=True)
        # Farklı görüntüler → eşleşme yok → L_nat = 0
        out = natural_loss(online, online.detach().clone(),
                            torch.tensor([[0.0], [1.0]]),
                            torch.tensor([0, 1]), m, T)
        L_ctrl_dummy = (online * online).sum()
        (L_ctrl_dummy + out["loss"]).backward()
        assert online.grad is not None
        assert online.grad.abs().sum().item() > 0

    def test_rejects_1d_features(self):
        m = NaturalPairMatcher()
        T = ScaleEquivariantTransform(feat_dim=64)
        with pytest.raises(ValueError):
            natural_loss(
                torch.randn(4), torch.randn(4, 64),
                torch.zeros(4, 1), torch.zeros(4, dtype=torch.long), m, T,
            )

    def test_rejects_shape_mismatch(self):
        m = NaturalPairMatcher()
        T = ScaleEquivariantTransform(feat_dim=64)
        with pytest.raises(ValueError):
            natural_loss(
                torch.randn(4, 64), torch.randn(4, 32),
                torch.zeros(4, 1), torch.zeros(4, dtype=torch.long), m, T,
            )
