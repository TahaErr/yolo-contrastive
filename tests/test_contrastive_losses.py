"""Tests for contrastive loss functions.

Covers audit §6.3 edge cases:
  - B=1 (degenerate — should warn + return 0)
  - B=2 (minimum valid)
  - Various temperatures (very small, very large)
  - Identical embeddings (z1 == z2)
  - All-zero embeddings
  - Gradient flow
"""

import warnings

import pytest
import torch

from yolo_contrastive.contrastive import NTXentLoss, build_contrastive_loss
from yolo_contrastive.exceptions import ContrastiveLossError


class TestNTXentLossBasic:
    """Core forward + backward checks."""

    def test_scalar_and_finite(self):
        loss_fn = build_contrastive_loss("ntxent", temperature=0.2)
        z1 = torch.randn(8, 256, requires_grad=True)
        z2 = torch.randn(8, 256)
        loss = loss_fn(z1, z2)

        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        loss.backward()
        assert z1.grad is not None
        assert torch.isfinite(z1.grad).all().item()

    @pytest.mark.parametrize("B", [2, 4, 16])
    @pytest.mark.parametrize("D", [32, 128, 512])
    def test_various_shapes(self, B, D):
        loss_fn = NTXentLoss(temperature=0.2)
        z1 = torch.randn(B, D, requires_grad=True)
        z2 = torch.randn(B, D)
        loss = loss_fn(z1, z2)

        assert loss.ndim == 0
        assert torch.isfinite(loss).item()
        assert loss.item() > 0  # should not be exactly zero for random inputs

    def test_aliases(self):
        """ntxent, infonce, simclr should all build NTXentLoss."""
        for name in ("ntxent", "infonce", "simclr"):
            fn = build_contrastive_loss(name)
            assert isinstance(fn, NTXentLoss)

    def test_unknown_loss_raises(self):
        with pytest.raises(ContrastiveLossError, match="Unknown contrastive loss"):
            build_contrastive_loss("triplet")


class TestNTXentEdgeCases:
    """Edge cases from audit §2.1, §6.3."""

    def test_batch_size_1_returns_zero_with_warning(self):
        """B=1: no meaningful negatives -> zero loss + UserWarning."""
        loss_fn = NTXentLoss(temperature=0.2)
        z1 = torch.randn(1, 128, requires_grad=True)
        z2 = torch.randn(1, 128)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss = loss_fn(z1, z2)

        assert loss.item() == 0.0
        assert loss.requires_grad  # must still be differentiable
        assert any("B=1" in str(warning.message) for warning in w)

    def test_batch_size_2_works(self):
        """B=2: minimum valid batch."""
        loss_fn = NTXentLoss(temperature=0.2)
        z1 = torch.randn(2, 64, requires_grad=True)
        z2 = torch.randn(2, 64)
        loss = loss_fn(z1, z2)

        assert loss.ndim == 0
        assert torch.isfinite(loss).item()
        assert loss.item() > 0

    def test_identical_embeddings(self):
        """z1 == z2: loss should be valid (positives are perfect matches)."""
        loss_fn = NTXentLoss(temperature=0.2)
        z1 = torch.randn(4, 128, requires_grad=True)
        z2 = z1.detach().clone()  # identical
        loss = loss_fn(z1, z2)

        assert torch.isfinite(loss).item()

    def test_all_zero_embeddings(self):
        """All-zero inputs: L2 norm -> 0/0, but eps should prevent NaN."""
        loss_fn = NTXentLoss(temperature=0.2, eps=1e-8)
        z1 = torch.zeros(4, 128, requires_grad=True)
        z2 = torch.zeros(4, 128)
        loss = loss_fn(z1, z2)

        # Should not be NaN (eps protects normalize)
        assert not torch.isnan(loss).item()

    def test_z2_none_fallback(self):
        """z2=None should fall back to z1 and still produce valid output."""
        loss_fn = NTXentLoss(temperature=0.2)
        z1 = torch.randn(4, 128)
        loss = loss_fn(z1, z2=None)

        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

    def test_batch_size_mismatch_raises(self):
        loss_fn = NTXentLoss()
        z1 = torch.randn(4, 128)
        z2 = torch.randn(8, 128)
        with pytest.raises(ContrastiveLossError, match="Batch size mismatch"):
            loss_fn(z1, z2)


class TestNTXentTemperature:
    """Temperature edge cases (audit §6.3)."""

    def test_very_small_temperature(self):
        """Very small temp: logits are large but should not overflow with float32 cast."""
        loss_fn = NTXentLoss(temperature=0.001)
        z1 = torch.randn(4, 64, requires_grad=True)
        z2 = torch.randn(4, 64)
        loss = loss_fn(z1, z2)

        assert torch.isfinite(loss).item()

    def test_very_large_temperature(self):
        """Very large temp: logits -> 0, loss -> log(2B-1)."""
        loss_fn = NTXentLoss(temperature=100.0)
        z1 = torch.randn(4, 64)
        z2 = torch.randn(4, 64)
        loss = loss_fn(z1, z2)

        import math
        expected_approx = math.log(2 * 4 - 1)  # log(7) ~ 1.946
        assert torch.isfinite(loss).item()
        assert abs(loss.item() - expected_approx) < 0.5  # rough check

    def test_zero_temperature_raises(self):
        with pytest.raises(ContrastiveLossError, match="temperature must be > 0"):
            NTXentLoss(temperature=0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ContrastiveLossError, match="temperature must be > 0"):
            NTXentLoss(temperature=-0.1)


class TestNTXentLabelsWarning:
    """Audit §2.3 — labels param should warn when used."""

    def test_labels_emits_warning(self):
        loss_fn = NTXentLoss()
        z1 = torch.randn(4, 64)
        z2 = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 0, 1])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss = loss_fn(z1, z2, labels=labels)

        assert torch.isfinite(loss).item()
        assert any("labels" in str(warning.message).lower() for warning in w)

    def test_no_labels_no_warning(self):
        loss_fn = NTXentLoss()
        z1 = torch.randn(4, 64)
        z2 = torch.randn(4, 64)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = loss_fn(z1, z2)

        label_warnings = [x for x in w if "labels" in str(x.message).lower()]
        assert len(label_warnings) == 0
