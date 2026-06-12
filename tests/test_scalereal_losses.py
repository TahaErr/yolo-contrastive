"""Loss-landscape tests for GASP-Real — analytic ground truth, CPU-only.

The analytic case: features whose scalar potential is exactly s = -log Z make
the pair prediction s_B - s_A equal log(Z_A / Z_B) — the true label — so the
loss is ~0 at truth and strictly increases with decoy offset |delta|.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.scalereal.losses import (
    connected_zero,
    content_consistency_loss,
    pair_scale_pred,
    scale_pair_loss,
    spearman_corr,
)


def _analytic_pairs():
    """All ordered pairs from depths [2, 5, 10, 20] with s = -log Z."""
    zs = [2.0, 5.0, 10.0, 20.0]
    s_a, s_b, log_r = [], [], []
    for za in zs:
        for zb in zs:
            if za == zb:
                continue
            s_a.append(-math.log(za))
            s_b.append(-math.log(zb))
            log_r.append(math.log(za / zb))
    return (torch.tensor(s_a), torch.tensor(s_b), torch.tensor(log_r))


class TestScalePairLoss:
    def test_zero_at_truth(self):
        s_a, s_b, log_r = _analytic_pairs()
        out = scale_pair_loss(s_a, s_b, log_r)
        assert float(out["loss"]) == pytest.approx(0.0, abs=1e-10)
        assert out["n_pairs"] == len(log_r)
        assert out["sign_acc"] == pytest.approx(1.0)

    def test_loss_increases_with_decoy_offset(self):
        """Loss at the true log_r is lower than at every decoy, and strictly
        monotone in |delta| (the loss-at-truth-lower-than-decoys requirement)."""
        s_a, s_b, log_r = _analytic_pairs()
        truth = float(scale_pair_loss(s_a, s_b, log_r)["loss"])
        prev = truth
        for delta in (0.25, 0.5, 1.0):
            for sign in (1.0, -1.0):
                decoy = float(scale_pair_loss(s_a, s_b, log_r + sign * delta)["loss"])
                assert decoy > truth
            up = float(scale_pair_loss(s_a, s_b, log_r + delta)["loss"])
            assert up > prev
            prev = up

    def test_exact_antisymmetry(self):
        torch.manual_seed(0)
        s_a, s_b = torch.randn(32), torch.randn(32)
        log_r = torch.randn(32)
        # prediction antisymmetric BY CONSTRUCTION: s_AB == -s_BA exactly
        assert torch.equal(pair_scale_pred(s_a, s_b), -pair_scale_pred(s_b, s_a))
        # loss invariant under (A, B, log_r) -> (B, A, -log_r), bit-exact
        l_fwd = scale_pair_loss(s_a, s_b, log_r)["loss"]
        l_bwd = scale_pair_loss(s_b, s_a, -log_r)["loss"]
        assert torch.equal(l_fwd, l_bwd)

    def test_diagnostics(self):
        s_a = torch.tensor([0.0, 0.0])
        s_b = torch.tensor([1.0, -1.0])
        log_r = torch.tensor([1.0, 1.0])  # second pair has the wrong sign
        out = scale_pair_loss(s_a, s_b, log_r)
        assert out["sign_acc"] == pytest.approx(0.5)
        assert out["pred_std"] == pytest.approx(1.0)

    def test_zero_pairs_graph_connected(self):
        s_a = torch.zeros(0, requires_grad=True)
        s_b = torch.zeros(0, requires_grad=True)
        out = scale_pair_loss(s_a, s_b, torch.zeros(0))
        assert out["n_pairs"] == 0
        assert out["loss"].grad_fn is not None
        out["loss"].backward()  # must not raise

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            scale_pair_loss(torch.zeros(3), torch.zeros(3), torch.zeros(4))

    def test_bad_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            scale_pair_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), beta=0.0)

    def test_gradient_flows_to_stub_backbone(self):
        """End-to-end grad flow: feature -> linear scale head -> pair loss."""
        torch.manual_seed(1)
        feats = torch.randn(8, 4, requires_grad=True)  # stub backbone output
        head = nn.Linear(4, 1)
        s = head(feats).squeeze(-1)
        out = scale_pair_loss(s[:4], s[4:], torch.randn(4))
        out["loss"].backward()
        assert feats.grad is not None
        assert float(feats.grad.abs().sum()) > 0
        assert head.weight.grad is not None


class TestContentConsistencyLoss:
    def test_stop_grad_side_receives_no_grad(self):
        torch.manual_seed(2)
        z_a = torch.randn(6, 8, requires_grad=True)
        z_b = torch.randn(6, 8, requires_grad=True)
        q_a = torch.randn(6, 8, requires_grad=True)
        q_b = torch.randn(6, 8, requires_grad=True)
        out = content_consistency_loss(q_a, q_b, z_a, z_b)
        out["loss"].backward()
        # sg() side: z receives no grad; predictor side: q does
        assert z_a.grad is None
        assert z_b.grad is None
        assert q_a.grad is not None
        assert q_b.grad is not None

    def test_identical_inputs_zero_loss(self):
        torch.manual_seed(3)
        z = torch.randn(5, 16)
        out = content_consistency_loss(z, z, z, z)
        assert float(out["loss"]) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_inputs_max_loss(self):
        z = torch.randn(4, 16)
        out = content_consistency_loss(-z, -z, z, z)
        assert float(out["loss"]) == pytest.approx(2.0, abs=1e-5)

    def test_zero_pairs_graph_connected(self):
        q = torch.zeros(0, 8, requires_grad=True)
        z = torch.zeros(0, 8)
        out = content_consistency_loss(q, q, z, z)
        assert out["n_pairs"] == 0
        assert out["loss"].grad_fn is not None
        out["loss"].backward()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            content_consistency_loss(
                torch.zeros(3, 8), torch.zeros(3, 8),
                torch.zeros(3, 8), torch.zeros(3, 4),
            )


class TestHelpers:
    def test_connected_zero_with_grad_refs(self):
        p = torch.randn(3, requires_grad=True)
        z = connected_zero([p])
        assert float(z.detach()) == 0.0
        assert z.grad_fn is not None
        z.backward()
        assert torch.allclose(p.grad, torch.zeros(3))

    def test_connected_zero_without_grad_refs(self):
        z = connected_zero([torch.randn(3)])
        assert float(z) == 0.0  # plain zero fallback, no crash

    def test_spearman_monotone(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert spearman_corr(x, x.exp()) == pytest.approx(1.0)
        assert spearman_corr(x, -x) == pytest.approx(-1.0)
        assert spearman_corr(torch.zeros(1), torch.zeros(1)) == 0.0
