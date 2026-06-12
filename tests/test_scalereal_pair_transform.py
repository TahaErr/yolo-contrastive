"""Metamorphic R5 tests — boxes track content through random affines.

Strategy: paint patches with unique flat colors, warp the image with a
sampled theta, transform the boxes with the SAME theta, and assert each
surviving transformed box still contains its color. The log_r payload must be
bit-identical for survivors (labels are aug-invariant; validity is not).
"""

from __future__ import annotations

import pytest
import torch

from yolo_contrastive.scalereal.pair_transform import (
    assert_aspect_ok,
    boxes_to_padded,
    filter_transformed_pairs,
    identity_theta,
    letterbox_to_square,
    sample_rrc_theta,
    theta_anisotropy,
    transform_boxes_theta,
    warp_images,
)

OUT = 128

COLORS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]


def _paint_scene(boxes_norm, size=128):
    """Gray canvas with one unique flat color per box."""
    img = torch.full((3, size, size), 0.5)
    for k, b in enumerate(boxes_norm):
        x1, y1 = int(b[0] * size), int(b[1] * size)
        x2, y2 = int(b[2] * size), int(b[3] * size)
        for c in range(3):
            img[c, y1:y2, x1:x2] = COLORS[k % len(COLORS)][c]
    return img


def _center_color(view, box_norm):
    """Pixel color at the center of a view-normalized box."""
    s = view.shape[-1]
    cx = int((box_norm[0] + box_norm[2]) / 2 * s)
    cy = int((box_norm[1] + box_norm[3]) / 2 * s)
    cx = min(max(cx, 0), s - 1)
    cy = min(max(cy, 0), s - 1)
    return view[:, cy, cx]


class TestContentTracking:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_boxes_track_content_through_random_affine(self, seed):
        boxes = torch.tensor([
            [0.10, 0.10, 0.40, 0.40],
            [0.55, 0.15, 0.90, 0.50],
            [0.15, 0.55, 0.45, 0.90],
            [0.55, 0.55, 0.90, 0.90],
        ])
        img = _paint_scene(boxes, OUT)
        gen = torch.Generator().manual_seed(seed)
        theta = sample_rrc_theta(1, scale=(0.5, 1.0), ratio=(1 / 1.15, 1.15),
                                 hflip_prob=0.5, generator=gen)
        view = warp_images(img.unsqueeze(0), theta, OUT)[0]
        tb = transform_boxes_theta(boxes, theta[0])
        kept = filter_transformed_pairs(
            tb[:2], tb[2:], torch.tensor([0.7, -1.1]), OUT,
            max_clip_frac=0.2, min_patch_px=8,
        )
        # every surviving box must still contain its color at its center
        idx_a = torch.nonzero(kept["keep"]).flatten().tolist()
        for row, orig in enumerate(idx_a):
            color_a = torch.tensor(COLORS[orig])
            color_b = torch.tensor(COLORS[orig + 2])
            got_a = _center_color(view, kept["boxes_a"][row])
            got_b = _center_color(view, kept["boxes_b"][row])
            assert torch.allclose(got_a, color_a, atol=0.05), (seed, orig, got_a)
            assert torch.allclose(got_b, color_b, atol=0.05), (seed, orig, got_b)

    def test_flip_handled(self):
        boxes = torch.tensor([[0.05, 0.40, 0.25, 0.60]])  # left side
        img = _paint_scene(boxes, OUT)
        theta = identity_theta(1)
        theta[0, 0, 0] = -1.0  # pure hflip
        view = warp_images(img.unsqueeze(0), theta, OUT)[0]
        tb = transform_boxes_theta(boxes, theta[0])
        # box moved to the right side, still ordered x1 < x2
        assert tb[0, 0] == pytest.approx(0.75, abs=1e-5)
        assert tb[0, 2] == pytest.approx(0.95, abs=1e-5)
        assert tb[0, 0] < tb[0, 2]
        assert torch.allclose(_center_color(view, tb[0]),
                              torch.tensor(COLORS[0]), atol=0.05)

    def test_identity_theta_is_noop(self):
        boxes = torch.rand(5, 2)
        boxes = torch.cat([boxes * 0.5, boxes * 0.5 + 0.4], dim=1)
        tb = transform_boxes_theta(boxes, identity_theta(1)[0])
        assert torch.allclose(tb, boxes, atol=1e-6)


class TestLabelInvariance:
    def test_log_r_payload_bit_identical_for_survivors(self):
        torch.manual_seed(7)
        boxes_a = torch.tensor([[0.2, 0.2, 0.45, 0.45], [0.1, 0.6, 0.3, 0.8]])
        boxes_b = torch.tensor([[0.6, 0.6, 0.85, 0.85], [0.6, 0.1, 0.8, 0.3]])
        log_r = torch.tensor([0.91234, -1.456])
        gen = torch.Generator().manual_seed(11)
        theta = sample_rrc_theta(1, scale=(0.8, 1.0), generator=gen)
        kept = filter_transformed_pairs(
            transform_boxes_theta(boxes_a, theta[0]),
            transform_boxes_theta(boxes_b, theta[0]),
            log_r, OUT, min_patch_px=4,
        )
        assert torch.equal(kept["log_r"], log_r[kept["keep"]])


class TestValidityGating:
    def test_clipped_pairs_dropped(self):
        # crop = right half of the image -> a box on the left is fully outside
        theta = identity_theta(1)
        theta[0, 0, 0] = 0.5   # half width
        theta[0, 0, 2] = 0.5   # centered at x = 0.75
        boxes_a = torch.tensor([
            [0.10, 0.40, 0.30, 0.60],   # fully outside the crop -> dropped
            [0.60, 0.40, 0.80, 0.60],   # fully inside -> kept
        ])
        boxes_b = torch.tensor([
            [0.60, 0.10, 0.80, 0.30],
            [0.55, 0.65, 0.75, 0.85],
        ])
        log_r = torch.tensor([1.0, -1.0])
        kept = filter_transformed_pairs(
            transform_boxes_theta(boxes_a, theta[0]),
            transform_boxes_theta(boxes_b, theta[0]),
            log_r, OUT, max_clip_frac=0.2, min_patch_px=4,
        )
        assert kept["keep"].tolist() == [False, True]
        assert kept["log_r"].tolist() == [-1.0]

    def test_partial_clip_over_threshold_dropped(self):
        # crop right half; box straddles the boundary with ~50% visible
        theta = identity_theta(1)
        theta[0, 0, 0] = 0.5
        theta[0, 0, 2] = 0.5
        boxes = torch.tensor([[0.40, 0.40, 0.60, 0.60]])  # 50% inside crop
        tb = transform_boxes_theta(boxes, theta[0])
        kept = filter_transformed_pairs(tb, tb.clone(), torch.tensor([1.0]), OUT,
                                        max_clip_frac=0.2, min_patch_px=4)
        assert not bool(kept["keep"][0])

    def test_too_small_boxes_dropped(self):
        tiny = torch.tensor([[0.45, 0.45, 0.55, 0.55]])   # 12.8 px at 128
        big = torch.tensor([[0.10, 0.10, 0.50, 0.50]])
        kept = filter_transformed_pairs(tiny, big, torch.tensor([1.0]), OUT,
                                        min_patch_px=24.0)
        assert not bool(kept["keep"][0])
        kept2 = filter_transformed_pairs(big, big.clone(), torch.tensor([1.0]), OUT,
                                         min_patch_px=24.0)
        assert bool(kept2["keep"][0])

    def test_partner_dropped_jointly(self):
        """A pair whose B-box dies must not leave a partnerless A-box."""
        good = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        dead = torch.tensor([[1.2, 1.2, 1.5, 1.5]])  # fully outside
        kept = filter_transformed_pairs(good, dead, torch.tensor([0.5]), OUT)
        assert kept["boxes_a"].shape[0] == 0
        assert kept["boxes_b"].shape[0] == 0

    def test_empty_input(self):
        kept = filter_transformed_pairs(torch.zeros(0, 4), torch.zeros(0, 4),
                                        torch.zeros(0), OUT)
        assert kept["log_r"].numel() == 0


class TestAspectGuard:
    def test_anisotropy_value(self):
        theta = identity_theta(1)
        theta[0, 0, 0] = 0.5
        theta[0, 1, 1] = 1.0
        assert float(theta_anisotropy(theta)[0]) == pytest.approx(2.0)

    def test_assert_raises_on_anisotropic_theta(self):
        theta = identity_theta(1)
        theta[0, 1, 1] = 2.0
        with pytest.raises(ValueError, match="aspect distortion"):
            assert_aspect_ok(theta, max_ratio=1.2)

    def test_assert_passes_within_bound_and_under_flip(self):
        theta = identity_theta(1)
        theta[0, 0, 0] = -1.1   # flip + mild aspect
        assert_aspect_ok(theta, max_ratio=1.2)

    def test_sampler_respects_bounds(self):
        gen = torch.Generator().manual_seed(3)
        theta = sample_rrc_theta(64, scale=(0.5, 1.0), ratio=(1 / 1.15, 1.15),
                                 generator=gen)
        assert float(theta_anisotropy(theta).max()) <= 1.2 + 1e-4

    def test_sampler_respects_content_box(self):
        gen = torch.Generator().manual_seed(4)
        cb = (0.25, 0.0, 0.75, 1.0)
        theta = sample_rrc_theta(64, scale=(0.5, 1.0), content_box=cb, generator=gen)
        cx = (theta[:, 0, 2] + 1.0) / 2.0
        cy = (theta[:, 1, 2] + 1.0) / 2.0
        hw = theta[:, 0, 0].abs() / 2.0
        hh = theta[:, 1, 1].abs() / 2.0
        assert float((cx - hw).min()) >= cb[0] - 1e-5
        assert float((cx + hw).max()) <= cb[2] + 1e-5
        assert float((cy - hh).min()) >= cb[1] - 1e-5
        assert float((cy + hh).max()) <= cb[3] + 1e-5


class TestLetterbox:
    def test_pad_geometry_and_box_mapping(self):
        img = torch.rand(3, 64, 128)  # wide image
        padded, content = letterbox_to_square(img)
        assert padded.shape == (3, 128, 128)
        assert content == pytest.approx((0.0, 0.25, 1.0, 0.75))
        # a full-image box maps exactly onto the content region
        mapped = boxes_to_padded(torch.tensor([[0.0, 0.0, 1.0, 1.0]]).numpy(), content)
        assert mapped[0] == pytest.approx([0.0, 0.25, 1.0, 0.75], abs=1e-6)
        # center pixel preserved (pure padding, no resampling): pad_y = 32
        assert torch.equal(padded[:, 32 + 48, 64], img[:, 48, 64])

    def test_square_image_is_unchanged(self):
        img = torch.rand(3, 64, 64)
        padded, content = letterbox_to_square(img)
        assert torch.equal(padded, img)
        assert content == pytest.approx((0.0, 0.0, 1.0, 1.0))
