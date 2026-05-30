"""Tests for feature_regularization_loss — VICReg-style kollaps engeli.

GASP'ın bilgi-koruma şartı: feature uzayı bilgi taşımıyorsa
eşdeğişirlik trivially sağlanır (her şey ~sabit). Bu kayıp variance
(boyut başına std ≥ γ) ve covariance (boyut-bağımsızlık) zorunluluğunu
ekler.
"""

from __future__ import annotations

import pytest
import torch

from yolo_contrastive.gasp import feature_regularization_loss


class TestFeatureRegularizationLoss:
    def test_returns_dict_with_three_terms(self):
        features = torch.randn(32, 64)
        out = feature_regularization_loss(features)
        assert set(out.keys()) == {"variance", "covariance", "isotropy"}
        assert out["variance"].dim() == 0
        assert out["covariance"].dim() == 0
        assert out["isotropy"].dim() == 0

    def test_collapsed_features_high_variance_loss(self):
        """Tüm görüntüler aynı feature → std=0 → variance loss = γ."""
        # 32 örnek, hepsi aynı vektör
        collapsed = torch.zeros(32, 64) + torch.randn(1, 64)
        out = feature_regularization_loss(collapsed, variance_target=1.0)
        # std ~0, γ=1 → hinge ~1
        assert out["variance"].item() > 0.9

    def test_diverse_features_low_variance_loss(self):
        """Geniş varyanslı feature'lar → variance loss ≈ 0."""
        features = torch.randn(64, 32) * 2.0   # std ~2
        out = feature_regularization_loss(features, variance_target=1.0)
        # std > γ → hinge = 0
        assert out["variance"].item() < 0.01

    def test_highly_correlated_features_high_cov_loss(self):
        """Tüm boyutlar tek bir kaynaktan kopyalanmış → kovaryans yüksek."""
        N = 64
        z = torch.randn(N, 1)
        features = z.expand(N, 32)   # her boyut z'nin kopyası
        out = feature_regularization_loss(features.clone() + 1e-3 * torch.randn(N, 32))
        assert out["covariance"].item() > 0.1

    def test_independent_features_low_cov_loss(self):
        """Tamamen bağımsız boyutlar → kovaryans ≈ 0."""
        features = torch.randn(128, 32)
        out = feature_regularization_loss(features)
        assert out["covariance"].item() < 1.0   # gevşek üst sınır, ama mantıklı

    def test_gradient_flows(self):
        features = torch.randn(32, 64, requires_grad=True)
        out = feature_regularization_loss(features)
        total = out["variance"] + out["covariance"]
        total.backward()
        assert features.grad is not None
        assert features.grad.abs().sum().item() > 0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            feature_regularization_loss(torch.randn(32))   # 1D
        with pytest.raises(ValueError):
            feature_regularization_loss(torch.randn(2, 3, 4))   # 3D

    def test_single_sample_returns_zero(self):
        """N=1 — varyans tanımsız, sıfır kayıp dön."""
        out = feature_regularization_loss(torch.randn(1, 64))
        assert out["variance"].item() == 0.0
        assert out["covariance"].item() == 0.0
        assert out["isotropy"].item() == 0.0

    def test_isotropy_distinguishes_rank_collapse(self):
        """Isotropy terimi, kovaryansın KÖR olduğu varyans-eşitsizliği
        collapse'ını yakalamalı: varyans birkaç boyuta yığılı (düşük
        eff_rank) ama boyutlar dik → cov≈eşit ama isotropy ÇOK yüksek."""
        torch.manual_seed(0)
        N, D = 256, 256
        # sağlıklı: eşit-varyanslı, korelasyonsuz
        healthy = torch.randn(N, D)
        # çökmüş: varyans 10 boyuta yığılı, geri kalan ~0, ama dik
        collapsed = torch.zeros(N, D)
        collapsed[:, :10] = torch.randn(N, 10) * 3.0
        collapsed[:, 10:] = torch.randn(N, D - 10) * 0.02
        # AYNI norm'a ölçekle — kovaryans-körlüğü ancak eşit norm'da net
        healthy = healthy * (3.63 / healthy.norm(dim=1).mean())
        collapsed = collapsed * (3.63 / collapsed.norm(dim=1).mean())
        h = feature_regularization_loss(healthy)
        c = feature_regularization_loss(collapsed)
        # kovaryans ikisini ayırt EDEMEZ (kanıtlanmış körlük, eşit norm'da)
        assert abs(h["covariance"].item() - c["covariance"].item()) < 0.01
        # isotropy çökmüşe çok daha yüksek loss atmalı (ayırt eder)
        assert c["isotropy"].item() > 10.0 * (h["isotropy"].item() + 1e-6)

    def test_isotropy_gradient_flows(self):
        features = torch.randn(64, 32, requires_grad=True)
        out = feature_regularization_loss(features)
        out["isotropy"].backward()
        assert features.grad is not None
        assert features.grad.abs().sum().item() > 0
