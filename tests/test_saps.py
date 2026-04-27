"""Tests for saps_within_loss (Faz 2.1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from yolo_contrastive.dense import (
    saps_within_loss,
    multi_scale_dense_loss,
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


def _make_features(B=2, D=16, sizes={"P3": 16, "P4": 8, "P5": 4},
                   requires_grad=False):
    q = {lv: _normalize(torch.randn(B, D, s, s, requires_grad=requires_grad), dim=1)
         for lv, s in sizes.items()}
    k = {lv: _normalize(torch.randn(B, D, s, s), dim=1)
         for lv, s in sizes.items()}
    return q, k


# ── basic shape & types ──────────────────────────────────────────────────


class TestBasic:
    def test_returns_scalar(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        loss, info = saps_within_loss(q, k, qc, kc, n_query=16)
        assert loss.dim() == 0
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss).item()

    def test_info_includes_cross_scale_negs(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        _, info = saps_within_loss(q, k, qc, kc, n_query=16)
        for lv in ("P3", "P4", "P5"):
            assert "cross_scale_negs" in info[lv]
            # P3 query gets negatives from P4 + P5 in-image
            # Sizes: P4=8x8=64, P5=4x4=16 → P3 cross_scale_negs = 80
        # Verify P3 specifically
        assert info["P3"]["cross_scale_negs"] == 8 * 8 + 4 * 4
        assert info["P4"]["cross_scale_negs"] == 16 * 16 + 4 * 4
        assert info["P5"]["cross_scale_negs"] == 16 * 16 + 8 * 8

    def test_total_aggregate(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        loss, info = saps_within_loss(q, k, qc, kc, n_query=16)
        assert "total" in info
        assert abs(info["total"]["loss"] - loss.item()) < 1e-6
        assert info["total"]["active_levels"] == 3


# ── single-level reduction (sanity) ──────────────────────────────────────


class TestSingleLevelReduction:
    def test_single_level_no_cross_scale_negs(self):
        """With only one FPN level, there are no cross-scale negatives.
        SAPS-within should produce zero cross_scale_negs.
        """
        q = {"P3": _normalize(torch.randn(2, 16, 16, 16), dim=1)}
        k = {"P3": _normalize(torch.randn(2, 16, 16, 16), dim=1)}
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)
        _, info = saps_within_loss(q, k, qc, kc, n_query=16)
        assert info["P3"]["cross_scale_negs"] == 0


# ── core SAPS hypothesis: cross-scale increases denominator ─────────────


class TestSAPSEffect:
    def test_saps_loss_geq_multiscale_loss(self):
        """SAPS-within adds extra negatives → loss >= multi_scale baseline.

        Math: same numerator (positives), larger denominator (extra cross-
        scale negatives) → -log(num/denom) is monotonically larger.
        """
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)

        loss_ms, _ = multi_scale_dense_loss(
            q, k, qc, kc, n_query=32, generator=gen1,
        )
        loss_saps, _ = saps_within_loss(
            q, k, qc, kc, n_query=32, generator=gen2,
        )
        # SAPS loss should be >= multi-scale (extra denominator entries).
        # Allow tiny float tolerance — both compute with same q/k/coord/seed.
        assert loss_saps.item() >= loss_ms.item() - 1e-5, (
            f"SAPS={loss_saps.item():.4f} expected >= MS={loss_ms.item():.4f}"
        )

    def test_strict_negatives_reduces_loss(self):
        """`strict_negatives=True` excludes spatially-overlapping cross-scale
        candidates → fewer denominator terms → loss <= saf SAPS.
        """
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)

        gen1 = torch.Generator().manual_seed(11)
        gen2 = torch.Generator().manual_seed(11)
        loss_naive, _ = saps_within_loss(
            q, k, qc, kc, n_query=32,
            strict_negatives=False, generator=gen1,
        )
        loss_strict, _ = saps_within_loss(
            q, k, qc, kc, n_query=32,
            strict_negatives=True, generator=gen2,
        )
        assert loss_strict.item() <= loss_naive.item() + 1e-5


# ── matching modes ──────────────────────────────────────────────────────


class TestMatchingModes:
    def test_threshold_mode_runs(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        loss, info = saps_within_loss(
            q, k, qc, kc, n_query=16, match_mode="threshold",
        )
        assert torch.isfinite(loss).item()
        assert info["P3"]["matched_frac"] > 0

    def test_nearest_mode_runs(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        loss, info = saps_within_loss(
            q, k, qc, kc, n_query=16,
            pos_radius=1.0,  # large, ensure nearest is within
            match_mode="nearest",
        )
        assert torch.isfinite(loss).item()

    def test_invalid_mode_raises(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="match_mode"):
            saps_within_loss(q, k, qc, qc, match_mode="bogus")


# ── queue forwarding ────────────────────────────────────────────────────


class TestQueueForwarding:
    def test_queue_increases_loss(self):
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        D = q["P3"].shape[1]
        queues = {lv: _normalize(torch.randn(50, D), dim=1) for lv in q}

        gen1 = torch.Generator().manual_seed(13)
        gen2 = torch.Generator().manual_seed(13)
        l_no_q, _ = saps_within_loss(q, k, qc, kc, n_query=16, generator=gen1)
        l_q, _ = saps_within_loss(q, k, qc, kc, queues=queues, n_query=16,
                                    generator=gen2)
        assert l_q.item() >= l_no_q.item() - 1e-5


# ── weight semantics ────────────────────────────────────────────────────


class TestWeights:
    def test_weight_zero_skips(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        _, info = saps_within_loss(
            q, k, qc, kc, weights={"P3": 1.0, "P4": 0.0, "P5": 0.0},
            n_query=16,
        )
        assert info["P4"].get("skipped") is True
        assert info["P5"].get("skipped") is True
        assert info["total"]["active_levels"] == 1

    def test_weight_unnormalized_normalized(self):
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        gen1 = torch.Generator().manual_seed(17)
        gen2 = torch.Generator().manual_seed(17)
        l1, _ = saps_within_loss(q, k, qc, kc,
                                  weights={"P3": 1, "P4": 1, "P5": 1},
                                  n_query=16, generator=gen1)
        l2, _ = saps_within_loss(q, k, qc, kc,
                                  weights={"P3": 5, "P4": 5, "P5": 5},
                                  n_query=16, generator=gen2)
        assert torch.allclose(l1, l2, atol=1e-5)

    def test_unknown_level_raises(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="unknown levels"):
            saps_within_loss(q, k, qc, qc, weights={"P3": 0.5, "P9": 0.5})


# ── shape errors ────────────────────────────────────────────────────────


class TestShapeErrors:
    def test_key_mismatch(self):
        q = {"P3": _normalize(torch.randn(2, 16, 8, 8), dim=1)}
        k = {"P4": _normalize(torch.randn(2, 16, 8, 8), dim=1)}
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="keys"):
            saps_within_loss(q, k, qc, qc)

    def test_invalid_qcoords(self):
        q, k = _make_features()
        with pytest.raises(ValueError, match="q_coords"):
            saps_within_loss(q, k, torch.randn(2, 3, 64, 64),
                              _coord_grid(2, 64, 64))

    def test_empty_features(self):
        with pytest.raises(ValueError, match="empty"):
            saps_within_loss({}, {}, _coord_grid(2, 8, 8),
                              _coord_grid(2, 8, 8))


# ── gradient flow ───────────────────────────────────────────────────────


class TestGradient:
    def test_grad_to_all_levels(self):
        torch.manual_seed(0)
        leafs = {
            "P3": torch.randn(2, 16, 8, 8, requires_grad=True),
            "P4": torch.randn(2, 16, 4, 4, requires_grad=True),
            "P5": torch.randn(2, 16, 2, 2, requires_grad=True),
        }
        q = {lv: _normalize(t, dim=1) for lv, t in leafs.items()}
        k = {lv: _normalize(torch.randn(*t.shape), dim=1).detach()
             for lv, t in leafs.items()}
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        loss, _ = saps_within_loss(q, k, qc, kc, n_query=16)
        loss.backward()
        for lv, leaf in leafs.items():
            assert leaf.grad is not None
            assert leaf.grad.abs().sum() > 0

    def test_no_grad_in_keys(self):
        """Caller passes detached k → no grad path through k_features."""
        q, k = _make_features(requires_grad=False)
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        # q_features without requires_grad → no backward path at all
        # Add a leaf to force the loss to have gradient origin
        q_leaf = torch.randn(2, 16, 16, 16, requires_grad=True)
        q["P3"] = _normalize(q_leaf, dim=1)
        loss, _ = saps_within_loss(q, k, qc, kc, n_query=16)
        loss.backward()
        # k tensors don't have requires_grad, so .grad is None
        for v in k.values():
            assert v.grad is None


# ── numerical stability ─────────────────────────────────────────────────


class TestStability:
    def test_no_nan_with_strict_neg_extreme(self):
        """strict_negatives masks many entries to -inf; ensure no NaN."""
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        kc = _coord_grid(2, 64, 64)  # identity coords → many overlaps
        loss, _ = saps_within_loss(
            q, k, qc, kc, n_query=16,
            strict_negatives=True,
            pos_radius=0.5,  # huge — masks lots of cross-scale entries
        )
        assert torch.isfinite(loss).item()

    def test_no_nan_with_low_temperature(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        loss, _ = saps_within_loss(q, k, qc, kc, n_query=16, temperature=0.05)
        assert torch.isfinite(loss).item()


# ── deterministic subsample ─────────────────────────────────────────────


class TestDeterminism:
    def test_same_generator_same_loss(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        l1, _ = saps_within_loss(q, k, qc, kc, n_query=16, generator=gen1)
        l2, _ = saps_within_loss(q, k, qc, kc, n_query=16, generator=gen2)
        assert torch.allclose(l1, l2)


# ═════════════════════════════════════════════════════════════════════════
# Cross-image SAPS tests (Faz 2.2)
# ═════════════════════════════════════════════════════════════════════════


from yolo_contrastive.dense import saps_cross_loss, FeatureQueue, combine_queues


LEVEL_TO_ID = {"P3": 0, "P4": 1, "P5": 2}


def _make_tagged_queue(D=16, K=32, fill=20):
    """Build {P3, P4, P5} queues with tags, partially filled with random keys."""
    queues = {lv: FeatureQueue(dim=D, K=K, with_tags=True)
              for lv in ("P3", "P4", "P5")}
    for lv, q in queues.items():
        keys = _normalize(torch.randn(fill, D), dim=1)
        # Tag each entry with its level id
        tags = torch.full((fill,), LEVEL_TO_ID[lv], dtype=torch.long)
        q.enqueue(keys, tags)
    keys, tags = combine_queues(queues, level_to_id=LEVEL_TO_ID)
    return keys, tags


# ── basic ───────────────────────────────────────────────────────────────


class TestCrossBasic:
    def test_returns_scalar(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        loss, info = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=1.0,
        )
        assert loss.dim() == 0 and torch.isfinite(loss).item()

    def test_info_includes_queue_neg_count(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue(fill=20)  # 60 total entries
        _, info = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=1.0,
        )
        for lv in ("P3", "P4", "P5"):
            assert info[lv]["queue_neg_count"] == 60
        assert info["total"]["t_scale"] == 1.0


# ── core SAPS-cross hypothesis: t_scale tunes scale awareness ───────────


class TestCrossSAPSEffect:
    def test_t_scale_huge_approximates_uniform(self):
        """Very large t_scale → exp(-Δ/t) → ~1 for all → behaves like MoCo
        with full queue. We just check finite loss & no crash."""
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        loss, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=1e6,
        )
        assert torch.isfinite(loss).item()

    def test_t_scale_small_isolates_levels(self):
        """Very small t_scale → only same-level negatives matter. Loss should
        still be finite and reasonable."""
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        loss, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=0.01,
        )
        assert torch.isfinite(loss).item()

    def test_t_scale_changes_loss(self):
        """Different t_scale → different loss (otherwise the parameter is
        meaningless)."""
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)
        l_small, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=0.1,
            generator=gen1,
        )
        l_large, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=10.0,
            generator=gen2,
        )
        assert not torch.allclose(l_small, l_large, atol=1e-3), (
            "t_scale had no effect on loss"
        )

    def test_small_t_scale_loss_lower_or_equal(self):
        """Mathematically: smaller t_scale → smaller weights for cross-level
        queue entries → smaller denominator → -log(num/denom) is LARGER or
        equal? Wait no: smaller denom → ratio larger → -log smaller. So
        smaller t_scale should give SMALLER (or equal) loss."""
        torch.manual_seed(0)
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()

        gen1 = torch.Generator().manual_seed(13)
        gen2 = torch.Generator().manual_seed(13)
        l_small, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=0.01,
            generator=gen1,
        )
        l_large, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=10.0,
            generator=gen2,
        )
        # Small t_scale isolates same-level → fewer effective negatives
        # → smaller denominator → smaller loss.
        assert l_small.item() <= l_large.item() + 1e-5, (
            f"small t_scale ({l_small.item():.4f}) "
            f"unexpectedly > large ({l_large.item():.4f})"
        )


# ── empty queue handling ────────────────────────────────────────────────


class TestCrossEmptyQueue:
    def test_empty_queue_finite(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        D = q["P3"].shape[1]
        empty_keys = torch.zeros(0, D)
        empty_tags = torch.zeros(0, dtype=torch.long)
        loss, info = saps_cross_loss(
            q, k, qc, kc, queue_keys=empty_keys, queue_tags=empty_tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=1.0,
        )
        assert torch.isfinite(loss).item()
        for lv in ("P3", "P4", "P5"):
            assert info[lv]["queue_neg_count"] == 0


# ── matching modes ──────────────────────────────────────────────────────


class TestCrossMatching:
    def test_threshold(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        loss, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16,
            match_mode="threshold", t_scale=1.0,
        )
        assert torch.isfinite(loss).item()

    def test_nearest(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        loss, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, pos_radius=1.0,
            match_mode="nearest", t_scale=1.0,
        )
        assert torch.isfinite(loss).item()


# ── shape / config errors ───────────────────────────────────────────────


class TestCrossErrors:
    def test_invalid_t_scale(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        with pytest.raises(ValueError, match="t_scale"):
            saps_cross_loss(q, k, qc, qc, queue_keys=keys, queue_tags=tags,
                             level_to_id=LEVEL_TO_ID, t_scale=0.0)
        with pytest.raises(ValueError, match="t_scale"):
            saps_cross_loss(q, k, qc, qc, queue_keys=keys, queue_tags=tags,
                             level_to_id=LEVEL_TO_ID, t_scale=-1.0)

    def test_queue_dim_mismatch(self):
        q, k = _make_features(D=16)
        qc = _coord_grid(2, 64, 64)
        bad_keys = torch.randn(10, 32)  # D=32 not 16
        bad_tags = torch.zeros(10, dtype=torch.long)
        with pytest.raises(ValueError, match="dim"):
            saps_cross_loss(q, k, qc, qc, queue_keys=bad_keys, queue_tags=bad_tags,
                             level_to_id=LEVEL_TO_ID)

    def test_tags_length_mismatch(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        keys = torch.randn(10, q["P3"].shape[1])
        tags = torch.zeros(5, dtype=torch.long)  # wrong length
        with pytest.raises(ValueError, match="queue_tags"):
            saps_cross_loss(q, k, qc, qc, queue_keys=keys, queue_tags=tags,
                             level_to_id=LEVEL_TO_ID)

    def test_missing_level_in_id_map(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        with pytest.raises(ValueError, match="level_to_id"):
            saps_cross_loss(q, k, qc, qc, queue_keys=keys, queue_tags=tags,
                             level_to_id={"P3": 0})  # missing P4, P5

    def test_bad_queue_shape(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64)
        with pytest.raises(ValueError, match="queue_keys"):
            saps_cross_loss(q, k, qc, qc,
                             queue_keys=torch.randn(10),  # 1D
                             queue_tags=torch.zeros(10, dtype=torch.long),
                             level_to_id=LEVEL_TO_ID)


# ── gradient flow ───────────────────────────────────────────────────────


class TestCrossGradient:
    def test_grad_to_all_levels(self):
        torch.manual_seed(0)
        leafs = {
            "P3": torch.randn(2, 16, 8, 8, requires_grad=True),
            "P4": torch.randn(2, 16, 4, 4, requires_grad=True),
            "P5": torch.randn(2, 16, 2, 2, requires_grad=True),
        }
        q = {lv: _normalize(t, dim=1) for lv, t in leafs.items()}
        k = {lv: _normalize(torch.randn(*t.shape), dim=1).detach()
             for lv, t in leafs.items()}
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue(D=16)
        loss, _ = saps_cross_loss(
            q, k, qc, kc, queue_keys=keys, queue_tags=tags,
            level_to_id=LEVEL_TO_ID, n_query=16, t_scale=1.0,
        )
        loss.backward()
        for lv, leaf in leafs.items():
            assert leaf.grad is not None
            assert leaf.grad.abs().sum() > 0


# ── determinism ─────────────────────────────────────────────────────────


class TestCrossDeterminism:
    def test_same_generator_same_loss(self):
        q, k = _make_features()
        qc = _coord_grid(2, 64, 64); kc = _coord_grid(2, 64, 64)
        keys, tags = _make_tagged_queue()
        gen1 = torch.Generator().manual_seed(99)
        gen2 = torch.Generator().manual_seed(99)
        l1, _ = saps_cross_loss(q, k, qc, kc,
                                  queue_keys=keys, queue_tags=tags,
                                  level_to_id=LEVEL_TO_ID, n_query=16,
                                  t_scale=1.0, generator=gen1)
        l2, _ = saps_cross_loss(q, k, qc, kc,
                                  queue_keys=keys, queue_tags=tags,
                                  level_to_id=LEVEL_TO_ID, n_query=16,
                                  t_scale=1.0, generator=gen2)
        assert torch.allclose(l1, l2)
