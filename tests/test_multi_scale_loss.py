"""Tests for multi_scale_dense_loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from yolo_contrastive.dense import (
    multi_scale_dense_loss,
    dense_ntxent_loss,
    coords_to_feature_map,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _normalize(t, dim=1):
    return F.normalize(t, dim=dim, eps=1e-8)


def _coord_grid(B: int, H: int, W: int) -> torch.Tensor:
    xs = (torch.arange(W).float() + 0.5) / W
    ys = (torch.arange(H).float() + 0.5) / H
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=0)
    return grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


def _make_multiscale_features(
    B: int = 2, D: int = 16,
    sizes={"P3": 16, "P4": 8, "P5": 4},
    requires_grad: bool = False,
):
    """Build a dict of multi-scale features at YOLOv8-like FPN sizes."""
    q = {
        lv: _normalize(torch.randn(B, D, s, s, requires_grad=requires_grad), dim=1)
        for lv, s in sizes.items()
    }
    k = {
        lv: _normalize(torch.randn(B, D, s, s), dim=1)
        for lv, s in sizes.items()
    }
    return q, k


# ── basic shape / types ──────────────────────────────────────────────────


class TestBasic:
    def test_returns_scalar(self):
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        loss, info = multi_scale_dense_loss(q, k, qc, kc, n_query=16)
        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_info_has_per_level_and_total(self):
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        _, info = multi_scale_dense_loss(q, k, qc, kc, n_query=16)
        for lv in ("P3", "P4", "P5"):
            assert lv in info
            assert "loss" in info[lv]
            assert "weight" in info[lv]
        assert "total" in info
        for k_ in ("loss", "active_levels", "n_used_total"):
            assert k_ in info["total"]


# ── equivalence with single-level dense_loss ─────────────────────────────


class TestSingleLevelEquivalence:
    def test_p3_only_matches_dense_ntxent(self):
        """weights={'P3': 1.0} should give same loss as direct dense_ntxent_loss."""
        torch.manual_seed(0)
        B, D, S = 2, 16, 16
        q_feat = _normalize(torch.randn(B, D, S, S), dim=1)
        k_feat = _normalize(torch.randn(B, D, S, S), dim=1)
        qc = _coord_grid(B, 64, 64)
        kc = _coord_grid(B, 64, 64)

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)

        # Multi-scale: only P3 active
        q_dict = {"P3": q_feat, "P4": _normalize(torch.randn(B, D, 8, 8), dim=1),
                  "P5": _normalize(torch.randn(B, D, 4, 4), dim=1)}
        k_dict = {"P3": k_feat, "P4": _normalize(torch.randn(B, D, 8, 8), dim=1),
                  "P5": _normalize(torch.randn(B, D, 4, 4), dim=1)}
        loss_ms, _ = multi_scale_dense_loss(
            q_dict, k_dict, qc, kc,
            weights={"P3": 1.0, "P4": 0.0, "P5": 0.0},
            n_query=32, generator=gen1,
        )

        # Direct: with the same coords resampled
        qc_p3 = coords_to_feature_map(qc, S, S)
        kc_p3 = coords_to_feature_map(kc, S, S)
        loss_direct, _ = dense_ntxent_loss(
            q_feat, k_feat, qc_p3, kc_p3, n_query=32, generator=gen2,
        )

        # weights normalization makes P3 weight = 1.0, so they should be equal
        assert torch.allclose(loss_ms, loss_direct, atol=1e-5)


# ── weight semantics ─────────────────────────────────────────────────────


class TestWeights:
    def test_weight_normalization(self):
        """Unnormalized weights should be auto-normalized to sum=1."""
        torch.manual_seed(0)
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)

        loss_norm, _ = multi_scale_dense_loss(
            q, k, qc, kc, weights={"P3": 1/3, "P4": 1/3, "P5": 1/3},
            n_query=16, generator=gen1,
        )
        loss_unnorm, _ = multi_scale_dense_loss(
            q, k, qc, kc, weights={"P3": 5.0, "P4": 5.0, "P5": 5.0},
            n_query=16, generator=gen2,
        )
        assert torch.allclose(loss_norm, loss_unnorm, atol=1e-5)

    def test_default_weights_equal(self):
        """No weights kwarg → equal 1/L per level."""
        torch.manual_seed(0)
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)

        loss_default, info_default = multi_scale_dense_loss(
            q, k, qc, kc, n_query=16, generator=gen1,
        )
        loss_explicit, _ = multi_scale_dense_loss(
            q, k, qc, kc, weights={"P3": 1/3, "P4": 1/3, "P5": 1/3},
            n_query=16, generator=gen2,
        )
        assert torch.allclose(loss_default, loss_explicit, atol=1e-5)

    def test_weight_zero_skips_level(self):
        torch.manual_seed(0)
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        _, info = multi_scale_dense_loss(
            q, k, qc, kc, weights={"P3": 1.0, "P4": 0.0, "P5": 0.0},
            n_query=16,
        )
        assert info["P4"].get("skipped") is True
        assert info["P5"].get("skipped") is True
        assert info["total"]["active_levels"] == 1

    def test_unknown_level_in_weights_raises(self):
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="unknown levels"):
            multi_scale_dense_loss(
                q, k, qc, qc, weights={"P3": 0.5, "P9": 0.5}, n_query=16,
            )

    def test_negative_total_weight_raises(self):
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="non-positive"):
            multi_scale_dense_loss(
                q, k, qc, qc,
                weights={"P3": 0.0, "P4": 0.0, "P5": 0.0},
                n_query=16,
            )


# ── queue forwarding ────────────────────────────────────────────────────


class TestQueueForwarding:
    def test_per_level_queues_used(self):
        """Loss with queues > loss without (more negatives → larger denom)."""
        torch.manual_seed(0)
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)

        D = q["P3"].shape[1]
        queues = {
            "P3": _normalize(torch.randn(50, D), dim=1),
            "P4": _normalize(torch.randn(50, D), dim=1),
            "P5": _normalize(torch.randn(50, D), dim=1),
        }

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)
        l_no_q, _ = multi_scale_dense_loss(q, k, qc, kc, n_query=16, generator=gen1)
        l_q, _ = multi_scale_dense_loss(q, k, qc, kc, queues=queues, n_query=16,
                                          generator=gen2)
        assert l_q.item() >= l_no_q.item() - 1e-5

    def test_partial_queues(self):
        """Some levels with queue, some without (None) — should run."""
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        D = q["P3"].shape[1]
        queues = {"P3": _normalize(torch.randn(20, D), dim=1)}
        loss, _ = multi_scale_dense_loss(q, k, qc, kc, queues=queues, n_query=16)
        assert torch.isfinite(loss).item()


# ── shape errors ─────────────────────────────────────────────────────────


class TestShapeErrors:
    def test_key_mismatch(self):
        q = {"P3": _normalize(torch.randn(2, 16, 8, 8), dim=1)}
        k = {"P4": _normalize(torch.randn(2, 16, 8, 8), dim=1)}
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="keys"):
            multi_scale_dense_loss(q, k, qc, qc, n_query=16)

    def test_q_k_spatial_mismatch(self):
        q = {"P3": _normalize(torch.randn(2, 16, 8, 8), dim=1)}
        k = {"P3": _normalize(torch.randn(2, 16, 16, 16), dim=1)}
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="spatial size"):
            multi_scale_dense_loss(q, k, qc, qc, n_query=16)

    def test_invalid_qcoords(self):
        q, k = _make_multiscale_features()
        with pytest.raises(ValueError, match="q_coords"):
            multi_scale_dense_loss(q, k, torch.randn(2, 3, 64, 64),
                                   _coord_grid(2, 64, 64), n_query=16)

    def test_empty_features_raises(self):
        with pytest.raises(ValueError, match="empty"):
            multi_scale_dense_loss({}, {}, _coord_grid(2, 8, 8),
                                   _coord_grid(2, 8, 8))


# ── coord resampling integration ────────────────────────────────────────


class TestCoordResampling:
    def test_coords_resampled_per_level(self):
        """Different feature-map sizes per level → coords resized internally."""
        # YOLOv8-like sizes for 64×64 input: P3=8, P4=4, P5=2
        torch.manual_seed(0)
        q = {
            "P3": _normalize(torch.randn(2, 16, 8, 8), dim=1),
            "P4": _normalize(torch.randn(2, 16, 4, 4), dim=1),
            "P5": _normalize(torch.randn(2, 16, 2, 2), dim=1),
        }
        k = {
            "P3": _normalize(torch.randn(2, 16, 8, 8), dim=1),
            "P4": _normalize(torch.randn(2, 16, 4, 4), dim=1),
            "P5": _normalize(torch.randn(2, 16, 2, 2), dim=1),
        }
        qc = _coord_grid(2, 64, 64)  # view-resolution
        kc = _coord_grid(2, 64, 64)

        loss, info = multi_scale_dense_loss(q, k, qc, kc, n_query=8)
        assert torch.isfinite(loss).item()
        assert info["total"]["active_levels"] == 3


# ── gradient ─────────────────────────────────────────────────────────────


class TestGradient:
    def test_grad_flows_to_all_levels(self):
        torch.manual_seed(0)
        B, D = 2, 16
        leafs = {
            "P3": torch.randn(B, D, 8, 8, requires_grad=True),
            "P4": torch.randn(B, D, 4, 4, requires_grad=True),
            "P5": torch.randn(B, D, 2, 2, requires_grad=True),
        }
        q = {lv: _normalize(t, dim=1) for lv, t in leafs.items()}
        k = {lv: _normalize(torch.randn(*t.shape), dim=1).detach()
             for lv, t in leafs.items()}
        qc = _coord_grid(B, 64, 64)
        kc = _coord_grid(B, 64, 64)

        loss, _ = multi_scale_dense_loss(q, k, qc, kc, n_query=16)
        loss.backward()

        for lv, leaf in leafs.items():
            assert leaf.grad is not None, f"No grad for level {lv}"
            assert leaf.grad.abs().sum() > 0, f"Zero grad for level {lv}"

    def test_zero_weight_no_grad_at_that_level(self):
        """A skipped level should NOT receive gradient."""
        torch.manual_seed(0)
        B, D = 2, 16
        leafs = {
            "P3": torch.randn(B, D, 8, 8, requires_grad=True),
            "P4": torch.randn(B, D, 4, 4, requires_grad=True),
            "P5": torch.randn(B, D, 2, 2, requires_grad=True),
        }
        q = {lv: _normalize(t, dim=1) for lv, t in leafs.items()}
        k = {lv: _normalize(torch.randn(*t.shape), dim=1).detach()
             for lv, t in leafs.items()}
        qc = _coord_grid(B, 64, 64)
        kc = _coord_grid(B, 64, 64)

        loss, _ = multi_scale_dense_loss(
            q, k, qc, kc,
            weights={"P3": 1.0, "P4": 0.0, "P5": 0.0},
            n_query=16,
        )
        loss.backward()
        assert leafs["P3"].grad is not None
        # P4 and P5 should have None or all-zero grads
        for lv in ("P4", "P5"):
            g = leafs[lv].grad
            assert g is None or g.abs().sum().item() == 0.0, \
                f"Skipped level {lv} got non-zero grad"


# ── total info aggregation ───────────────────────────────────────────────


class TestTotalInfo:
    def test_total_loss_matches_returned(self):
        torch.manual_seed(0)
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        loss, info = multi_scale_dense_loss(q, k, qc, kc, n_query=16)
        assert abs(info["total"]["loss"] - loss.item()) < 1e-6

    def test_n_used_total_sums_correctly(self):
        torch.manual_seed(0)
        q, k = _make_multiscale_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        _, info = multi_scale_dense_loss(q, k, qc, kc, n_query=16)
        n_sum = sum(info[lv]["n_used"] for lv in ("P3", "P4", "P5")
                    if not info[lv].get("skipped"))
        assert info["total"]["n_used_total"] == n_sum
