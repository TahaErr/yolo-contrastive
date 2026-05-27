"""Tests for NaturalPairMatcher — GASP §4 Karar 1+2."""

from __future__ import annotations

import math

import pytest
import torch

from yolo_contrastive.gasp import NaturalPairMatcher


class TestNaturalPairMatcher:
    def test_attrs(self):
        m = NaturalPairMatcher(similarity_threshold=0.7)
        assert m.similarity_threshold == 0.7

    def test_perfect_match_same_image_diff_scale(self):
        m = NaturalPairMatcher(similarity_threshold=0.7)
        features = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        log_scales = torch.tensor([[0.0], [1.0]])
        image_ids = torch.tensor([0, 0])
        out = m.match(features, log_scales, image_ids)
        assert out is not None
        idx_a, idx_b, log_r = out
        assert idx_a.tolist() == [0]
        assert idx_b.tolist() == [1]
        assert log_r[0].item() == pytest.approx(1.0)

    def test_same_scale_rejected(self):
        m = NaturalPairMatcher(similarity_threshold=0.7)
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        out = m.match(features, torch.tensor([[0.5], [0.5]]),
                      torch.tensor([0, 0]))
        assert out is None

    def test_different_image_rejected(self):
        m = NaturalPairMatcher(similarity_threshold=0.7)
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        out = m.match(features, torch.tensor([[0.0], [1.0]]),
                      torch.tensor([0, 1]))
        assert out is None

    def test_below_threshold_rejected(self):
        m = NaturalPairMatcher(similarity_threshold=0.7)
        # cosine = 0
        features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = m.match(features, torch.tensor([[0.0], [1.0]]),
                      torch.tensor([0, 0]))
        assert out is None

    def test_low_threshold_accepts_low_similarity(self):
        m = NaturalPairMatcher(similarity_threshold=-1.0)
        features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = m.match(features, torch.tensor([[0.0], [1.0]]),
                      torch.tensor([0, 0]))
        assert out is not None

    def test_mutual_nearest_neighbor(self):
        """A-B karşılıklı, B-C tek-yönlü → yalnız A-B eşleşir."""
        features = torch.tensor([
            [1.0, 0.0, 0.0],   # A
            [0.9, 0.1, 0.0],   # B (A'ya yakın)
            [0.5, 0.5, 0.0],   # C (B'ye yakın ama B'nin en yakını A)
        ])
        m = NaturalPairMatcher(similarity_threshold=0.5)
        out = m.match(features, torch.tensor([[0.0], [1.0], [2.0]]),
                      torch.tensor([0, 0, 0]))
        assert out is not None
        idx_a, idx_b, _ = out
        assert set(zip(idx_a.tolist(), idx_b.tolist())) == {(0, 1)}

    def test_multiple_images_independent_matches(self):
        features = torch.tensor([
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ])
        m = NaturalPairMatcher(similarity_threshold=0.7)
        out = m.match(features,
                      torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
                      torch.tensor([0, 0, 1, 1]))
        assert out is not None
        idx_a, idx_b, _ = out
        assert set(zip(idx_a.tolist(), idx_b.tolist())) == {(0, 1), (2, 3)}

    def test_single_patch_returns_none(self):
        m = NaturalPairMatcher()
        out = m.match(torch.randn(1, 64), torch.tensor([[0.0]]),
                      torch.tensor([0]))
        assert out is None

    def test_log_ratio_sign(self):
        """log_r = log_scale_b - log_scale_a, i < j."""
        m = NaturalPairMatcher(similarity_threshold=-1.0)
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        log_scales = torch.tensor([[math.log(0.2)], [math.log(0.5)]])
        out = m.match(features, log_scales, torch.tensor([0, 0]))
        assert out is not None
        _, _, log_r = out
        expected = math.log(0.5) - math.log(0.2)
        assert log_r[0].item() == pytest.approx(expected, abs=1e-5)

    def test_rejects_invalid_features_shape(self):
        m = NaturalPairMatcher()
        with pytest.raises(ValueError):
            m.match(torch.randn(4), torch.zeros(4, 1),
                    torch.zeros(4, dtype=torch.long))

    def test_rejects_inconsistent_shapes(self):
        m = NaturalPairMatcher()
        with pytest.raises(ValueError):
            m.match(torch.randn(4, 64), torch.zeros(3, 1),
                    torch.zeros(4, dtype=torch.long))
        with pytest.raises(ValueError):
            m.match(torch.randn(4, 64), torch.zeros(4, 1),
                    torch.zeros(5, dtype=torch.long))

    def test_rejects_invalid_threshold(self):
        with pytest.raises(ValueError):
            NaturalPairMatcher(similarity_threshold=1.5)
        with pytest.raises(ValueError):
            NaturalPairMatcher(similarity_threshold=-2.0)
