"""Tests for ScaleEquivariantTransform — GASP §2.1.

T(f, r) = (1 + γ(log r)) ⊙ f + β(log r)  — affine/FiLM, channel-wise.
Constrained capacity: small hidden_dim, channel-wise γ/β (no cross-channel
mixing). Bias-only zero-init at the final layer (gradient flows through
all parameters; ``yaklaşık identity'' init was an artifact of the previous
weight-zeroing and is not a design requirement).
"""

from __future__ import annotations

import pytest
import torch

from yolo_contrastive.gasp.transform import ScaleEquivariantTransform


class TestScaleEquivariantTransform:
    def test_attrs_and_param_count(self):
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        assert T.feat_dim == 128
        assert T.hidden_dim == 32
        # constrained — keep T small (kaçış kapısı yok)
        assert sum(p.numel() for p in T.parameters()) < 10_000

    def test_output_shape(self):
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        f = torch.randn(4, 128)
        log_r = torch.randn(4, 1)
        assert T(f, log_r).shape == f.shape

    def test_initial_output_scale_is_sane(self):
        """T(f, r) starts in O(|f|) — not exploded, not degenerate.

        Old version asserted ``T ≈ f exactly'' but that was an artifact
        of zero-init weight (which boğdu gradient flow). With bias-only
        zero-init, T(f, r) has finite small perturbation at init.
        """
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        f = torch.randn(8, 64)
        log_r = torch.randn(8, 1)
        out = T(f, log_r)
        ratio = (out.norm() / f.norm()).item()
        assert 0.1 < ratio < 10.0, f"|T(f,r)|/|f| = {ratio:.2f} out of sane range"

    def test_initial_identity_distance_finite(self):
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        log_r = torch.randn(8, 1)
        d = T.identity_distance(log_r).item()
        assert 0 < d < 5.0, f"identity_distance {d} out of sane range"

    def test_gradient_flows_to_input(self):
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        f = torch.randn(4, 128, requires_grad=True)
        T(f, torch.randn(4, 1)).sum().backward()
        assert f.grad is not None
        assert f.grad.abs().sum().item() > 0

    def test_gradient_flows_to_first_layer(self):
        """Regression: weight-zero-init at gen[-1] boğmuştu gen[0] gradient'ini.

        Bias-only zero-init ile bütün katmanlara gradient akıyor.
        """
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        f = torch.randn(4, 128, requires_grad=True)
        T(f, torch.randn(4, 1)).sum().backward()
        g_first = T.gen[0].weight.grad
        assert g_first is not None
        assert g_first.abs().sum().item() > 0

    def test_gradient_flows_to_last_layer(self):
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        f = torch.randn(4, 128, requires_grad=True)
        T(f, torch.randn(4, 1)).sum().backward()
        assert T.gen[-1].weight.grad.abs().sum().item() > 0
        assert T.gen[-1].bias.grad.abs().sum().item() > 0

    def test_deterministic_in_eval_mode(self):
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        T.eval()
        f = torch.randn(3, 64)
        log_r = torch.randn(3, 1)
        assert torch.allclose(T(f, log_r), T(f, log_r))

    def test_different_ratios_give_different_outputs(self):
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        # simulate trained: nonzero output layer
        with torch.no_grad():
            T.gen[-1].weight.normal_(0, 0.1)
            T.gen[-1].bias.normal_(0, 0.1)
        f = torch.randn(2, 64)
        out_small = T(f, torch.full((2, 1), -1.0))
        out_large = T(f, torch.full((2, 1), +1.0))
        assert not torch.allclose(out_small, out_large)

    def test_rejects_batch_mismatch(self):
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        with pytest.raises(ValueError):
            T(torch.randn(4, 128), torch.randn(3, 1))

    def test_rejects_wrong_log_ratio_shape(self):
        T = ScaleEquivariantTransform(feat_dim=128, hidden_dim=32)
        with pytest.raises(ValueError):
            T(torch.randn(4, 128), torch.randn(4))  # [4] instead of [4,1]

    def test_identity_distance_grows_when_trained(self):
        """Smoke-test diagnostic: distance from identity grows as T learns.

        If at end of training identity_distance is ~unchanged from init,
        T has collapsed to identity — equivariance not learned.
        """
        T = ScaleEquivariantTransform(feat_dim=64, hidden_dim=16)
        log_r = torch.randn(8, 1)
        d_init = T.identity_distance(log_r).item()
        with torch.no_grad():
            T.gen[-1].weight.normal_(0, 0.5)
            T.gen[-1].bias.normal_(0, 0.5)
        d_after = T.identity_distance(log_r).item()
        assert d_after > d_init * 2

    def test_rejects_invalid_feat_dim(self):
        with pytest.raises(ValueError):
            ScaleEquivariantTransform(feat_dim=0)
        with pytest.raises(ValueError):
            ScaleEquivariantTransform(feat_dim=-5)

    def test_rejects_invalid_hidden_dim(self):
        with pytest.raises(ValueError):
            ScaleEquivariantTransform(feat_dim=64, hidden_dim=0)
