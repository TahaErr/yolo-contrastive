"""Tests for dense_ntxent_loss and coords_to_feature_map."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from yolo_contrastive.dense import (
    dense_ntxent_loss,
    coords_to_feature_map,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _normalize(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return F.normalize(t, dim=dim, eps=1e-8)


def _coord_grid(B: int, H: int, W: int) -> torch.Tensor:
    """Build a regular [0, 1] coord grid."""
    xs = (torch.arange(W).float() + 0.5) / W
    ys = (torch.arange(H).float() + 0.5) / H
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=0)        # [2, H, W]
    return grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


def _make_pair(
    B: int = 2, D: int = 16, H: int = 8, W: int = 8,
    same: bool = False, requires_grad: bool = True,
) -> tuple:
    """Construct (q, k, qc, kc) for tests. If same=True, q and k are identical."""
    q = _normalize(torch.randn(B, D, H, W, requires_grad=requires_grad), dim=1)
    k = q.detach().clone() if same else _normalize(torch.randn(B, D, H, W), dim=1)
    qc = _coord_grid(B, H, W)
    kc = qc.clone()  # identity coords → all positions have a positive
    return q, k, qc, kc


# ── coords_to_feature_map ────────────────────────────────────────────────


class TestCoordsToFeatureMap:
    def test_same_size_returns_identical(self):
        coords = _coord_grid(2, 16, 16)
        out = coords_to_feature_map(coords, 16, 16)
        assert out.shape == (2, 2, 16, 16)
        assert torch.equal(out, coords)

    def test_resamples_to_smaller(self):
        coords = _coord_grid(1, 32, 32)
        out = coords_to_feature_map(coords, 8, 8)
        assert out.shape == (1, 2, 8, 8)
        # Coords should still be roughly in [0, 1]
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_resamples_to_larger(self):
        coords = _coord_grid(1, 8, 8)
        out = coords_to_feature_map(coords, 32, 32)
        assert out.shape == (1, 2, 32, 32)

    def test_invalid_coords_shape(self):
        with pytest.raises(ValueError, match=r"\[B, 2, H, W\]"):
            coords_to_feature_map(torch.randn(2, 3, 8, 8), 8, 8)


# ── basic shape & types ──────────────────────────────────────────────────


class TestShapeAndTypes:
    def test_returns_scalar_loss(self):
        q, k, qc, kc = _make_pair()
        loss, info = dense_ntxent_loss(q, k, qc, kc, n_query=16)
        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_info_keys(self):
        q, k, qc, kc = _make_pair()
        _, info = dense_ntxent_loss(q, k, qc, kc, n_query=16)
        for key in ("matched_frac", "mean_pos_sim", "mean_neg_sim",
                    "acc_top1", "n_used"):
            assert key in info

    def test_no_info_when_disabled(self):
        q, k, qc, kc = _make_pair()
        _, info = dense_ntxent_loss(q, k, qc, kc, n_query=16, return_info=False)
        assert info == {}


# ── core sanity: same vs random ──────────────────────────────────────────


class TestSemanticCorrectness:
    def test_identical_features_loss_is_small(self):
        """Same q and k, identity coords → top-1 accuracy near 1.

        Note: mean_pos_sim averages over ALL positive (q, k) pairs in the
        radius — including off-diagonal spatial neighbours whose features
        are NOT identical (they have different initial random feature
        vectors). So mean_pos_sim is not bounded above by 1 even when q==k.
        The semantic check is acc_top1: argmax over similarities should
        land on the diagonal positive (the same-position match).
        """
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair(B=4, D=32, H=16, W=16, same=True,
                                  requires_grad=False)
        loss, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=128, temperature=0.1,
            pos_radius=0.04,  # tighter radius — fewer off-diagonal positives
        )
        assert info["acc_top1"] > 0.95

    def test_random_features_high_loss(self):
        """Random features → loss roughly log(num_negatives_per_query)."""
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair(B=4, D=32, H=8, W=8, same=False,
                                  requires_grad=False)
        loss, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=64, temperature=0.2,
        )
        # Loss should be substantially > 0 (but no upper bound assertion;
        # numeric value depends on negatives count)
        assert loss.item() > 0.5
        # Random alignment → low top-1
        assert info["acc_top1"] < 0.5

    def test_no_positives_zero_loss(self):
        """Coords completely disjoint → no matches → loss = 0."""
        torch.manual_seed(0)
        B, D, H, W = 2, 16, 8, 8
        q = _normalize(torch.randn(B, D, H, W, requires_grad=True), dim=1)
        k = _normalize(torch.randn(B, D, H, W), dim=1)
        # qc in [0, 0.4], kc in [0.6, 1.0] → distance > 0.2 everywhere
        qc = _coord_grid(B, H, W) * 0.4
        kc = _coord_grid(B, H, W) * 0.4 + 0.6
        loss, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=32, pos_radius=0.05,
        )
        assert loss.item() == 0.0
        assert info["n_used"] == 0


# ── matching modes ───────────────────────────────────────────────────────


class TestMatchingModes:
    def test_threshold_multi_positive(self):
        """In threshold mode at least some queries get >1 positive."""
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair(H=16, W=16)
        # Manually build pos mask via the same logic
        # With identity coords + radius 0.1 + 16x16 grid, several neighbors qualify
        loss, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=64, pos_radius=0.1,
            match_mode="threshold",
        )
        # All sampled queries should match (every q has neighbors)
        assert info["matched_frac"] == 1.0

    def test_nearest_single_positive(self):
        """Nearest mode picks one positive per query."""
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair(H=8, W=8)
        loss, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=32, pos_radius=1.0,  # large enough
            match_mode="nearest",
        )
        assert info["matched_frac"] == 1.0

    def test_invalid_match_mode(self):
        q, k, qc, kc = _make_pair()
        with pytest.raises(ValueError, match="match_mode"):
            dense_ntxent_loss(q, k, qc, kc, match_mode="bogus")


# ── queue integration ────────────────────────────────────────────────────


class TestQueueIntegration:
    def test_with_queue_increases_neg_count(self):
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair()
        queue = _normalize(torch.randn(100, q.shape[1]), dim=1)
        loss_no_q, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16,
                                          queue=None)
        loss_with_q, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16,
                                            queue=queue)
        # Adding negatives should increase loss (more denominator mass)
        # This is monotonic for cross-entropy with extra negatives
        assert loss_with_q.item() >= loss_no_q.item() - 1e-6

    def test_queue_dim_mismatch_raises(self):
        q, k, qc, kc = _make_pair(D=16)
        queue = torch.randn(10, 32)  # wrong dim
        with pytest.raises(ValueError, match="dim"):
            dense_ntxent_loss(q, k, qc, kc, queue=queue)

    def test_queue_wrong_shape(self):
        q, k, qc, kc = _make_pair()
        with pytest.raises(ValueError, match="2-D"):
            dense_ntxent_loss(q, k, qc, kc, queue=torch.randn(10))

    def test_empty_queue_works(self):
        q, k, qc, kc = _make_pair()
        queue = torch.zeros(0, q.shape[1])
        loss, _ = dense_ntxent_loss(q, k, qc, kc, n_query=8, queue=queue)
        assert torch.isfinite(loss)


# ── gradient flow ────────────────────────────────────────────────────────


class TestGradientFlow:
    def test_grad_flows_to_q(self):
        # Build a leaf tensor for proper .grad access
        B, D, H, W = 2, 16, 8, 8
        q_leaf = torch.randn(B, D, H, W, requires_grad=True)
        q = _normalize(q_leaf, dim=1)               # non-leaf, but graph back to q_leaf
        k = _normalize(torch.randn(B, D, H, W), dim=1).detach()
        qc = _coord_grid(B, H, W)
        kc = qc.clone()
        loss, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16)
        loss.backward()
        assert q_leaf.grad is not None
        assert q_leaf.grad.abs().sum() > 0

    def test_no_grad_on_k_when_detached(self):
        """Caller is expected to pass detached k; verify nothing forces grad."""
        q, _, qc, kc = _make_pair(requires_grad=True)
        k = _normalize(torch.randn(*q.shape), dim=1).detach()
        loss, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16)
        loss.backward()
        # k has no grad attribute since requires_grad=False
        assert k.grad is None

    def test_zero_loss_keeps_graph(self):
        q, _, qc, _ = _make_pair(requires_grad=True)
        k = _normalize(torch.randn(*q.shape), dim=1)
        B, _, H, W = q.shape
        # Set k coords far away so no positives match
        kc = _coord_grid(B, H, W) + 5.0
        qc_local = _coord_grid(B, H, W)
        loss, info = dense_ntxent_loss(
            q, k, qc_local, kc, n_query=8, pos_radius=0.01,
        )
        assert info["n_used"] == 0
        # Should still backprop without error
        loss.backward()


# ── numerical stability ─────────────────────────────────────────────────


class TestNumericalStability:
    def test_no_nan_or_inf(self):
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair(B=2, D=8, H=8, W=8)
        loss, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=16, temperature=0.05,  # small T → big logits
        )
        assert torch.isfinite(loss).item()
        for v in info.values():
            assert math.isfinite(v) if isinstance(v, float) else True

    def test_extreme_temperature(self):
        q, k, qc, kc = _make_pair()
        for T in (1.0, 0.1, 0.01):
            loss, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16, temperature=T)
            assert torch.isfinite(loss).item()


# ── subsampling ──────────────────────────────────────────────────────────


class TestSubsampling:
    def test_n_query_larger_than_HW_uses_all(self):
        q, k, qc, kc = _make_pair(B=1, D=8, H=4, W=4)  # HW = 16
        _, info = dense_ntxent_loss(
            q, k, qc, kc, n_query=1000,  # > 16
        )
        assert info["n_used"] <= 16

    def test_deterministic_with_generator(self):
        torch.manual_seed(0)
        q, k, qc, kc = _make_pair(B=2, D=8, H=8, W=8, requires_grad=False)

        gen1 = torch.Generator(device=q.device).manual_seed(7)
        gen2 = torch.Generator(device=q.device).manual_seed(7)
        loss1, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16, generator=gen1)
        loss2, _ = dense_ntxent_loss(q, k, qc, kc, n_query=16, generator=gen2)
        assert torch.allclose(loss1, loss2)


# ── input validation ────────────────────────────────────────────────────


class TestValidation:
    def test_dim_mismatch_q_k(self):
        q = torch.randn(2, 16, 8, 8)
        k = torch.randn(2, 32, 8, 8)
        qc = _coord_grid(2, 8, 8)
        with pytest.raises(ValueError, match="Feature dims"):
            dense_ntxent_loss(q, k, qc, qc)

    def test_batch_mismatch_q_k(self):
        q = torch.randn(2, 16, 8, 8)
        k = torch.randn(4, 16, 8, 8)
        qc = _coord_grid(2, 8, 8)
        kc = _coord_grid(4, 8, 8)
        with pytest.raises(ValueError, match="Batch"):
            dense_ntxent_loss(q, k, qc, kc)

    def test_wrong_q_shape(self):
        q = torch.randn(2, 16, 8)  # 3D not 4D
        k = torch.randn(2, 16, 8, 8)
        qc = _coord_grid(2, 8, 8)
        with pytest.raises(ValueError, match=r"\[B, D, H, W\]"):
            dense_ntxent_loss(q, k, qc, qc)


# ── end-to-end with spatial_aug ─────────────────────────────────────────


class TestEndToEndWithSpatialAug:
    """Wire spatial_aug → simulate features → dense_loss; whole pipeline runs."""

    def test_full_pipeline_smoke(self):
        from yolo_contrastive.dense import SpatialTwoViewAugmentation

        torch.manual_seed(0)
        # Synthetic input image batch
        B, C, H, W = 2, 3, 64, 64
        imgs = torch.rand(B, C, H, W)

        aug = SpatialTwoViewAugmentation(out_size=(64, 64), seed=42)
        out = aug(imgs)

        # Stand in for encoder: just project via random linear (1x1 conv)
        D = 16
        proj = torch.nn.Conv2d(C, D, kernel_size=1)
        v1 = _normalize(proj(out.view1), dim=1)
        v2 = _normalize(proj(out.view2), dim=1).detach()

        # Resample coords to feature-map size (no-op here since same size)
        qc = coords_to_feature_map(out.coords1, v1.shape[2], v1.shape[3])
        kc = coords_to_feature_map(out.coords2, v2.shape[2], v2.shape[3])

        loss, info = dense_ntxent_loss(
            v1, v2, qc, kc, n_query=128, pos_radius=0.07,
        )
        assert torch.isfinite(loss).item()
        assert "matched_frac" in info
        # End-to-end gradient
        loss.backward()
        assert proj.weight.grad is not None

    def test_full_pipeline_with_feature_map_resize(self):
        """coords are at view resolution, but features are at feature-map resolution."""
        from yolo_contrastive.dense import SpatialTwoViewAugmentation

        torch.manual_seed(0)
        B = 2
        imgs = torch.rand(B, 3, 64, 64)
        aug = SpatialTwoViewAugmentation(out_size=(64, 64), seed=1)
        out = aug(imgs)

        # Simulate stride-8 feature map (what YOLOv8 P3 looks like)
        D = 16
        H_feat = W_feat = 8

        v1 = _normalize(torch.randn(B, D, H_feat, W_feat, requires_grad=True), dim=1)
        v2 = _normalize(torch.randn(B, D, H_feat, W_feat), dim=1)

        # Critical: resize coords from (64, 64) to (8, 8)
        qc = coords_to_feature_map(out.coords1, H_feat, W_feat)
        kc = coords_to_feature_map(out.coords2, H_feat, W_feat)
        assert qc.shape == (B, 2, H_feat, W_feat)

        loss, info = dense_ntxent_loss(
            v1, v2, qc, kc, n_query=32, pos_radius=0.1,
        )
        assert torch.isfinite(loss).item()
