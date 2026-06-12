"""Analytic end-to-end label tests on synthetic pinhole scenes (stub embedder).

THE analytic case: textured squares of known physical size at known metric
depths, exact inverse-depth map -> every mined pair's log_r must equal
log(Z_A / Z_B) within 1e-2. Gate edge cases (ratio band, overlap, texture)
and a ROW-DECORRELATED layout (labels derive from depth, not image row) are
covered alongside.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from yolo_contrastive.scalereal.config import ScaleRealConfig
from yolo_contrastive.scalereal.mine_pairs import (
    MiningStats,
    boxes_disjoint,
    grid_candidate_boxes,
    mine_image_pairs,
    texture_std,
)
from yolo_contrastive.scalereal.synthetic import (
    SyntheticSquare,
    render_pinhole_scene,
    row_decorrelated_scene,
    two_class_scene,
)

METRIC_SIDECAR = {
    "units": "1/m",
    "encoding": "inverse_depth",
    "variant": "dav2_metric_outdoor_small",
}
AFFINE_SIDECAR = {
    "units": "affine",
    "encoding": "inverse_depth",
    "variant": "dav2_rel_base",
}


def _mine(scene, cfg=None, stats=None):
    cfg = cfg or ScaleRealConfig()
    return mine_image_pairs(
        scene.image, scene.inv_depth, METRIC_SIDECAR,
        scene.make_stub_embedder(), cfg, stats=stats,
    )


def _true_log_r(scene, row):
    """Ground-truth log(Z_A / Z_B) from the squares the boxes fall in."""
    box_a = np.array([row["box_a_x1"], row["box_a_y1"], row["box_a_x2"], row["box_a_y2"]])
    box_b = np.array([row["box_b_x1"], row["box_b_y1"], row["box_b_x2"], row["box_b_y2"]])
    ia = scene.square_index_for_box(box_a)
    ib = scene.square_index_for_box(box_b)
    assert ia is not None and ib is not None, "mined box not inside any square"
    return math.log(scene.squares[ia].z_m / scene.squares[ib].z_m), ia, ib


class TestAnalyticLabels:
    def test_mined_log_r_matches_pinhole_geometry(self):
        """REQUIRED analytic case: |log_r - log(Z_A/Z_B)| <= 1e-2 per pair."""
        scene = two_class_scene()
        rows = _mine(scene)
        assert len(rows) >= 2, "miner found no pairs on the canonical scene"
        for row in rows:
            truth, ia, ib = _true_log_r(scene, row)
            assert row["log_r"] == pytest.approx(truth, abs=1e-2)
            # depths recorded faithfully too
            assert row["z_a"] == pytest.approx(scene.squares[ia].z_m, rel=1e-2)
            assert row["z_b"] == pytest.approx(scene.squares[ib].z_m, rel=1e-2)
            # pairs are same-content (the stub embedder keys on class)
            assert scene.squares[ia].class_id == scene.squares[ib].class_id

    def test_row_decorrelated_labels_derive_from_depth(self):
        """Same-row squares at different depths: correct nonzero labels even
        though the row coordinate carries no depth signal."""
        scene = row_decorrelated_scene(depths=(5.0, 15.0), row=0.5)
        rows = _mine(scene)
        assert len(rows) >= 2
        for row in rows:
            truth, _, _ = _true_log_r(scene, row)
            assert abs(truth) == pytest.approx(math.log(3.0), abs=1e-9)
            assert row["log_r"] == pytest.approx(truth, abs=1e-2)
            # the boxes really are on the same row (centers within one side)
            cy_a = (row["box_a_y1"] + row["box_a_y2"]) / 2
            cy_b = (row["box_b_y1"] + row["box_b_y2"]) / 2
            assert abs(cy_a - cy_b) < 0.20

    def test_pair_fields_complete(self):
        rows = _mine(two_class_scene())
        keys = {
            "box_a_x1", "box_a_y1", "box_a_x2", "box_a_y2",
            "box_b_x1", "box_b_y1", "box_b_x2", "box_b_y2",
            "log_r", "z_a", "z_b", "sim", "texture_a", "texture_b",
            "depth_iqr_a", "depth_iqr_b", "on_road_a", "on_road_b",
            "miner_version",
        }
        assert keys <= set(rows[0])
        assert math.isnan(rows[0]["on_road_a"])  # graceful null without TERRA


class TestGates:
    def test_ratio_band_rejects_out_of_band(self):
        """8x depth ratio (> 6x) and 1x (same depth) both yield zero pairs."""
        for za, zb in ((5.0, 40.0), (8.0, 8.0001)):
            f, target_px = 320.0, 96.0
            scene = render_pinhole_scene(
                [
                    SyntheticSquare(za, target_px * za / f, 0.30, 0.30, class_id=0),
                    SyntheticSquare(zb, target_px * zb / f, 0.72, 0.72, class_id=0),
                ],
                h=320, w=320, focal_px=f,
            )
            stats = MiningStats(ScaleRealConfig())
            rows = _mine(scene, stats=stats)
            assert rows == []
            assert stats.counters["pairs_failed_band"] > 0

    def test_textureless_patches_rejected(self):
        """Flat squares (no texture) survive depth gates but fail texture."""
        scene = render_pinhole_scene(
            [
                SyntheticSquare(5.0, 96 * 5.0 / 320, 0.30, 0.30, class_id=0),
                SyntheticSquare(15.0, 96 * 15.0 / 320, 0.72, 0.72, class_id=0),
            ],
            h=320, w=320, focal_px=320.0,
            texture_contrast=0.0,  # kills in-square contrast
        )
        stats = MiningStats(ScaleRealConfig())
        rows = _mine(scene, stats=stats)
        assert rows == []
        assert stats.counters["patches_failed_texture"] > 0

    def test_miner_refuses_relative_depth_cache(self):
        """The affine-ambiguity guard fires inside the miner too."""
        scene = two_class_scene()
        with pytest.raises(ValueError, match="metric|affine|invalid"):
            mine_image_pairs(
                scene.image, scene.inv_depth, AFFINE_SIDECAR,
                scene.make_stub_embedder(), ScaleRealConfig(),
            )

    def test_per_image_budget_and_stratification(self):
        cfg = ScaleRealConfig()
        rows = _mine(two_class_scene(), cfg=cfg)
        assert len(rows) <= cfg.max_pairs_per_image
        # bin occupancy never exceeds the per-bin cap
        edges = cfg.ratio_bin_edges
        counts = [0] * (len(edges) - 1)
        for r in rows:
            a = abs(r["log_r"])
            for k in range(len(edges) - 1):
                if edges[k] <= a < edges[k + 1] or (k == len(edges) - 2 and a == edges[-1]):
                    counts[k] += 1
        assert max(counts) <= cfg.max_pairs_per_bin

    def test_z_validity_band(self):
        """Far-field content (Z > 60 m) contributes no patches."""
        f = 320.0
        scene = render_pinhole_scene(
            [
                SyntheticSquare(70.0, 96 * 70.0 / f, 0.30, 0.30, class_id=0),
                SyntheticSquare(20.0, 96 * 20.0 / f, 0.72, 0.72, class_id=0),
            ],
            h=320, w=320, focal_px=f, background_z_m=75.0,
        )
        stats = MiningStats(ScaleRealConfig())
        rows = _mine(scene, stats=stats)
        assert rows == []
        assert stats.counters["patches_failed_depth_validity"] > 0


class TestPrimitives:
    def test_boxes_disjoint_analytic(self):
        a = np.array([0.10, 0.10, 0.30, 0.30])
        far = np.array([0.60, 0.60, 0.80, 0.80])
        near = np.array([0.32, 0.10, 0.52, 0.30])  # 0.02 gap < 1.25x expansion
        assert boxes_disjoint(a, far, expand=1.25)
        assert not boxes_disjoint(a, near, expand=1.25)
        assert boxes_disjoint(a, near, expand=1.0)  # touching-but-disjoint raw

    def test_grid_candidates_geometry(self):
        cfg = ScaleRealConfig()
        h, w = 240, 320
        boxes = grid_candidate_boxes(h, w, cfg)
        assert len(boxes) > 0
        # normalized, inside the image, inside the horizontal margin
        assert (boxes >= -1e-9).all() and (boxes <= 1 + 1e-9).all()
        assert (boxes[:, 0] >= cfg.central_margin - 1e-9).all()
        assert (boxes[:, 2] <= 1 - cfg.central_margin + 1e-9).all()
        # square in PIXELS at one of the configured side fractions
        side_px_x = (boxes[:, 2] - boxes[:, 0]) * w
        side_px_y = (boxes[:, 3] - boxes[:, 1]) * h
        assert np.allclose(side_px_x, side_px_y, atol=1e-6)
        expected = {round(fr * min(h, w), 4) for fr in cfg.grid_fractions}
        got = {round(float(s), 4) for s in np.unique(side_px_x.round(4))}
        assert got <= expected

    def test_texture_std_flat_vs_textured(self):
        flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
        assert texture_std(flat, np.array([0.0, 0.0, 1.0, 1.0])) < 1e-6
        noisy = np.random.default_rng(0).uniform(0, 1, (64, 64, 3)).astype(np.float32)
        assert texture_std(noisy, np.array([0.0, 0.0, 1.0, 1.0])) > 0.04
