"""Tests for SpatialTwoViewAugmentation."""

from __future__ import annotations

import pytest
import torch

from yolo_contrastive.dense import SpatialTwoViewAugmentation, TwoView


# ── helpers ──────────────────────────────────────────────────────────────


def _make_images(B: int = 2, C: int = 3, H: int = 64, W: int = 64) -> torch.Tensor:
    return torch.rand(B, C, H, W)


def _identity_aug(out_size=(32, 32)) -> SpatialTwoViewAugmentation:
    """No randomness: full-image crop, no flip."""
    return SpatialTwoViewAugmentation(
        out_size=out_size,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0),
        hflip_prob=0.0,
    )


# ── output shape & types ─────────────────────────────────────────────────


class TestOutputShape:
    def test_returns_named_tuple(self):
        aug = SpatialTwoViewAugmentation(out_size=(32, 32))
        out = aug(_make_images())
        assert isinstance(out, TwoView)

    def test_view_shapes(self):
        aug = SpatialTwoViewAugmentation(out_size=(48, 32))
        out = aug(_make_images(B=4))
        assert out.view1.shape == (4, 3, 48, 32)
        assert out.view2.shape == (4, 3, 48, 32)

    def test_coord_shapes(self):
        aug = SpatialTwoViewAugmentation(out_size=(48, 32))
        out = aug(_make_images(B=4))
        assert out.coords1.shape == (4, 2, 48, 32)
        assert out.coords2.shape == (4, 2, 48, 32)

    def test_coords_are_float32(self):
        aug = SpatialTwoViewAugmentation(out_size=(32, 32))
        imgs = _make_images().half() if False else _make_images()
        out = aug(imgs)
        assert out.coords1.dtype == torch.float32
        assert out.coords2.dtype == torch.float32

    def test_view_dtype_preserved(self):
        aug = SpatialTwoViewAugmentation(out_size=(32, 32))
        imgs = _make_images()  # float32
        out = aug(imgs)
        assert out.view1.dtype == imgs.dtype


# ── coord ranges ─────────────────────────────────────────────────────────


class TestCoordRanges:
    def test_coords_in_unit_square_for_identity(self):
        aug = _identity_aug(out_size=(16, 16))
        out = aug(_make_images())
        # All coords for full-image identity should be within [0, 1]
        assert (out.coords1 >= 0.0).all()
        assert (out.coords1 <= 1.0).all()
        assert (out.coords2 >= 0.0).all()
        assert (out.coords2 <= 1.0).all()

    def test_coords_can_be_outside_for_random_crop(self):
        """Note: with center+halfext sampling we ensure inside, so should still be [0,1]."""
        aug = SpatialTwoViewAugmentation(
            out_size=(32, 32), scale=(0.2, 0.5), hflip_prob=0.0,
        )
        out = aug(_make_images())
        # Crops are sampled to fit in image; coords should still be ⊂ [0,1]
        assert (out.coords1 >= -1e-5).all()
        assert (out.coords1 <= 1.0 + 1e-5).all()


# ── identity behaviour ───────────────────────────────────────────────────


class TestIdentity:
    def test_identity_coords_match_grid(self):
        """Full-image crop with no flip: coords should be a regular grid in [0, 1]."""
        H_out = W_out = 8
        aug = _identity_aug(out_size=(H_out, W_out))
        out = aug(_make_images(H=64, W=64))

        # Build expected coord grid (align_corners=False semantics)
        xs = (torch.arange(W_out).float() + 0.5) / W_out
        ys = (torch.arange(H_out).float() + 0.5) / H_out
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        expected_x = gx
        expected_y = gy

        for b in range(out.coords1.shape[0]):
            assert torch.allclose(out.coords1[b, 0], expected_x, atol=1e-5)
            assert torch.allclose(out.coords1[b, 1], expected_y, atol=1e-5)
            assert torch.allclose(out.coords2[b, 0], expected_x, atol=1e-5)
            assert torch.allclose(out.coords2[b, 1], expected_y, atol=1e-5)

    def test_identity_view_matches_input(self):
        """Identity transform should reconstruct input (up to bilinear resampling)."""
        aug = _identity_aug(out_size=(64, 64))
        # Constant image — bilinear resample is exact
        imgs = torch.full((1, 3, 64, 64), 0.5)
        out = aug(imgs)
        assert torch.allclose(out.view1, imgs, atol=1e-5)


# ── flip behaviour ───────────────────────────────────────────────────────


class TestFlip:
    def test_always_flip_inverts_x_coord(self):
        """hflip_prob=1.0: x coords should run from ~1 to ~0 (left to right)."""
        aug = SpatialTwoViewAugmentation(
            out_size=(8, 8),
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            hflip_prob=1.0,
        )
        out = aug(_make_images())
        # x at column 0 should be larger than x at column W-1
        x0 = out.coords1[:, 0, :, 0]
        x_last = out.coords1[:, 0, :, -1]
        assert (x0 > x_last).all()

    def test_no_flip_normal_x(self):
        aug = _identity_aug(out_size=(8, 8))
        out = aug(_make_images())
        x0 = out.coords1[:, 0, :, 0]
        x_last = out.coords1[:, 0, :, -1]
        assert (x0 < x_last).all()

    def test_flip_doesnt_change_y(self):
        aug_flip = SpatialTwoViewAugmentation(
            out_size=(8, 8), scale=(1.0, 1.0), ratio=(1.0, 1.0), hflip_prob=1.0,
        )
        aug_noflip = _identity_aug(out_size=(8, 8))
        out_f = aug_flip(_make_images())
        out_nf = aug_noflip(_make_images())
        # y coords (channel 1) shouldn't depend on flip
        assert torch.allclose(out_f.coords1[:, 1], out_nf.coords1[:, 1], atol=1e-5)


# ── crop behaviour ───────────────────────────────────────────────────────


class TestCrop:
    def test_small_crop_coords_span_subregion(self):
        """A scale=(0.1, 0.1) crop covers ~10% of image area; coords range should be small."""
        aug = SpatialTwoViewAugmentation(
            out_size=(32, 32),
            scale=(0.1, 0.1),
            ratio=(1.0, 1.0),
            hflip_prob=0.0,
        )
        out = aug(_make_images())
        for b in range(out.coords1.shape[0]):
            x_range = out.coords1[b, 0].max() - out.coords1[b, 0].min()
            y_range = out.coords1[b, 1].max() - out.coords1[b, 1].min()
            # sqrt(0.1) ≈ 0.316 — both axes
            assert x_range < 0.5
            assert y_range < 0.5
            assert x_range > 0.1
            assert y_range > 0.1

    def test_two_views_have_different_crops(self):
        """Sample many: views should very rarely produce identical coord maps."""
        aug = SpatialTwoViewAugmentation(
            out_size=(8, 8), scale=(0.2, 0.5), hflip_prob=0.5,
        )
        out = aug(_make_images(B=8))
        diffs = (out.coords1 - out.coords2).abs().mean()
        assert diffs > 0.01  # very loose, but identical coords would give 0

    def test_min_size_clamping_no_nan(self):
        """Extreme tiny scale should not produce NaN coords."""
        aug = SpatialTwoViewAugmentation(
            out_size=(16, 16),
            scale=(0.01, 0.02),
            ratio=(0.5, 2.0),
            hflip_prob=0.0,
        )
        out = aug(_make_images(H=128, W=128))
        assert not torch.isnan(out.coords1).any()
        assert not torch.isnan(out.coords2).any()
        assert not torch.isnan(out.view1).any()


# ── determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    def test_seed_reproduces(self):
        imgs = _make_images()
        aug1 = SpatialTwoViewAugmentation(out_size=(16, 16), seed=42)
        aug2 = SpatialTwoViewAugmentation(out_size=(16, 16), seed=42)
        out1 = aug1(imgs)
        out2 = aug2(imgs)
        assert torch.equal(out1.coords1, out2.coords1)
        assert torch.equal(out1.coords2, out2.coords2)
        assert torch.allclose(out1.view1, out2.view1)

    def test_different_seeds_differ(self):
        imgs = _make_images()
        out1 = SpatialTwoViewAugmentation(out_size=(16, 16), seed=42)(imgs)
        out2 = SpatialTwoViewAugmentation(out_size=(16, 16), seed=43)(imgs)
        assert not torch.equal(out1.coords1, out2.coords1)

    def test_consecutive_calls_differ(self):
        """Without seed, consecutive calls should differ (random sampling)."""
        torch.manual_seed(0)
        aug = SpatialTwoViewAugmentation(out_size=(16, 16))
        imgs = _make_images()
        out_a = aug(imgs)
        out_b = aug(imgs)
        assert not torch.equal(out_a.coords1, out_b.coords1)


# ── batched independence ─────────────────────────────────────────────────


class TestBatchIndependence:
    def test_each_sample_gets_different_crop(self):
        """In a single batch, samples should get different random crops."""
        aug = SpatialTwoViewAugmentation(
            out_size=(8, 8), scale=(0.2, 0.5), hflip_prob=0.5,
        )
        # Use different image content per sample to avoid coincidence
        imgs = torch.rand(8, 3, 64, 64)
        out = aug(imgs)
        # coords1 across samples should not all be identical
        ref = out.coords1[0]
        all_same = all(torch.equal(ref, out.coords1[i]) for i in range(1, 8))
        assert not all_same


# ── input validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_out_size(self):
        with pytest.raises(ValueError, match="out_size"):
            SpatialTwoViewAugmentation(out_size=(0, 32))
        with pytest.raises(ValueError, match="out_size"):
            SpatialTwoViewAugmentation(out_size=(32, -1))

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale"):
            SpatialTwoViewAugmentation(out_size=(32, 32), scale=(0.5, 0.3))
        with pytest.raises(ValueError, match="scale"):
            SpatialTwoViewAugmentation(out_size=(32, 32), scale=(0.0, 1.0))
        with pytest.raises(ValueError, match="scale"):
            SpatialTwoViewAugmentation(out_size=(32, 32), scale=(0.5, 1.5))

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="ratio"):
            SpatialTwoViewAugmentation(out_size=(32, 32), ratio=(2.0, 1.0))
        with pytest.raises(ValueError, match="ratio"):
            SpatialTwoViewAugmentation(out_size=(32, 32), ratio=(0.0, 2.0))

    def test_invalid_hflip_prob(self):
        with pytest.raises(ValueError, match="hflip_prob"):
            SpatialTwoViewAugmentation(out_size=(32, 32), hflip_prob=-0.1)
        with pytest.raises(ValueError, match="hflip_prob"):
            SpatialTwoViewAugmentation(out_size=(32, 32), hflip_prob=1.1)

    def test_invalid_input_shape(self):
        aug = SpatialTwoViewAugmentation(out_size=(32, 32))
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            aug(torch.rand(3, 64, 64))  # missing batch dim


# ── repr ─────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr(self):
        r = repr(SpatialTwoViewAugmentation(out_size=(32, 48)))
        assert "SpatialTwoViewAugmentation" in r
        assert "32, 48" in r
        assert "scale=" in r
        assert "hflip_prob=" in r


# ── coord-image consistency (sanity for dense_loss) ──────────────────────


class TestCoordImageConsistency:
    """Coords should match what grid_sample actually pulled.

    If we sample the input using the *coord map* (treated as a sampling grid),
    we should recover the same view tensor that aug already produced.
    """

    def test_coords_describe_actual_sampling(self):
        H, W = 48, 48
        # Use a positionally informative input: gradient image
        x = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        y = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        imgs = torch.cat([x, y, torch.zeros_like(x)], dim=1)  # [1, 3, H, W]

        aug = SpatialTwoViewAugmentation(out_size=(32, 32), seed=7)
        out = aug(imgs)

        # If coords1[b, 0, i, j] = orig x, coords1[b, 1, i, j] = orig y,
        # then view1[b, 0, i, j] (R-channel) ≈ orig_x at that position
        # and  view1[b, 1, i, j] (G-channel) ≈ orig_y at that position.
        view1_x = out.view1[0, 0]   # [H_out, W_out]
        view1_y = out.view1[0, 1]
        coord_x = out.coords1[0, 0]
        coord_y = out.coords1[0, 1]
        # Tolerance: bilinear edge effects
        assert torch.allclose(view1_x, coord_x, atol=0.05)
        assert torch.allclose(view1_y, coord_y, atol=0.05)
