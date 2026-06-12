"""Tests for the TERRA geometry-teacher module (geoteach/).

CPU-only, offline, no model downloads: synthetic inverse-depth images with an
analytically known plane + injected pits/mounds drive the plane fit, residual
binning, gating and box mining; the channel smoke test builds the detector
from ``yolov8n.yaml`` (ships inside ultralytics) through the real
AnchoredJointTrainer; the depth-cache build is exercised with a stub pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from yolo_contrastive.anchored import AnchoredJointTrainer, SentinelThresholds
from yolo_contrastive.geoteach import (
    ANOMALY_CLASSES,
    CLS_D1,
    CLS_D2,
    CLS_E1,
    CLS_E2,
    CLS_F,
    CLS_X,
    IGNORE_LABEL,
    NUM_CLASSES,
    BOX_DEPRESSION,
    BOX_ELEVATION,
    DenseOrdinalHead,
    DepthCache,
    OrdinalLossConfig,
    PlaneFitConfig,
    ResidualLabelConfig,
    TerraChannel,
    TerraPoolDataset,
    bin_residual,
    boxes_to_yolo_lines,
    compute_label_map,
    evaluate_surface,
    fit_road_plane,
    joint_crop_flip,
    labels_from_inverse_depth,
    majority_pool_labels,
    mine_boxes,
    ordinal_smoothing_matrix,
    run_depth_anything,
    sample_balanced_cells,
    standardized_residual,
    terra_collate,
    trapezoid_mask,
)

# Tiny 64px probes cannot reach production sentinel bars (see test_anchored.py).
RELAXED = SentinelThresholds(
    eff_rank_warn=0.0, eff_rank_abort=-1.0,
    cls_drift_warn=1e9, cls_drift_abort=1e9,
    cka_warn=-1.0, head_norm_growth_warn=1e9,
)

H, W = 160, 240
SIGMA = 0.002
TRUE_PARAMS = np.array([0.05, 0.6, 0.05, 0.0, 0.0])  # [u, v, 1, v^2, uv]

# Injected anomaly rectangles (rows, cols) — inside the bottom trapezoid and
# below the far-field cut. Depression = inverse depth SMALLER (farther).
PIT = (slice(120, 144), slice(100, 136))     # 24 x 36 px
MOUND = (slice(120, 140), slice(160, 192))   # 20 x 32 px


def synth_scene(pit_amp=0.0, mound_amp=0.0, noise=SIGMA, seed=0):
    """Synthetic inverse-depth plane with optional pit/mound, plus ground truth."""
    rng = np.random.default_rng(seed)
    d = evaluate_surface(TRUE_PARAMS, (H, W))
    d = d + rng.normal(0.0, noise, (H, W))
    if pit_amp:
        d[PIT] -= pit_amp
    if mound_amp:
        d[MOUND] += mound_amp
    return d


def fitted(d, **cfg_kw):
    fit = fit_road_plane(d, PlaneFitConfig(seed=0, **cfg_kw))
    z = standardized_residual(d, fit)
    d_surf = evaluate_surface(fit.params, d.shape)
    return fit, z, d_surf


# ── plane fit ─────────────────────────────────────────────────────────────────


class TestPlaneFit:
    def test_recovers_known_plane(self):
        d = synth_scene()
        fit, _, d_surf = fitted(d)
        assert fit.trusted
        true_surf = evaluate_surface(TRUE_PARAMS, (H, W))
        assert np.max(np.abs(d_surf - true_surf)) < 3 * SIGMA
        # sigma_MAD recovers the injected noise scale (one truncation pass)
        assert 0.5 * SIGMA < fit.sigma_mad < 1.5 * SIGMA
        # tau = 1.5 sigma keeps ~87% of Gaussian residuals
        assert fit.inlier_ratio > 0.75

    def test_robust_to_anomalies(self):
        """Pit + mound (outliers) must not drag the plane."""
        d = synth_scene(pit_amp=8 * SIGMA, mound_amp=8 * SIGMA)
        _, _, d_surf = fitted(d)
        true_surf = evaluate_surface(TRUE_PARAMS, (H, W))
        flat = np.ones((H, W), dtype=bool)
        flat[PIT] = flat[MOUND] = False
        assert np.max(np.abs(d_surf - true_surf)[flat]) < 3 * SIGMA

    def test_affine_invariance(self):
        """s*d + t (the relative-depth ambiguity) preserves the fit: z unchanged."""
        d = synth_scene(pit_amp=8 * SIGMA)
        _, z1, _ = fitted(d)
        _, z2, _ = fitted(3.7 * d + 11.0)
        assert np.allclose(z1, z2, atol=0.2)

    def test_z_sign_and_magnitude(self):
        d = synth_scene(pit_amp=8 * SIGMA, mound_amp=8 * SIGMA)
        _, z, _ = fitted(d)
        assert np.median(z[PIT]) == pytest.approx(-8.0, abs=2.0)      # depression: z < 0
        assert np.median(z[MOUND]) == pytest.approx(8.0, abs=2.0)     # elevation:  z > 0
        flat = np.abs(z[100:118, 30:90])
        assert np.median(flat) < 1.5

    def test_quadratic_term_recovered(self):
        params = np.array([0.0, 0.4, 0.1, 0.3, 0.0])  # crowned road (v^2 term)
        rng = np.random.default_rng(1)
        d = evaluate_surface(params, (H, W)) + rng.normal(0, SIGMA, (H, W))
        fit, z, _ = fitted(d)
        assert fit.trusted
        assert np.median(np.abs(z[PIT])) < 2.0  # quadratic absorbed, not an anomaly

    def test_untrusted_on_structureless_input(self):
        """Garbage input (noise spanning the disparity range): the absolute
        tau cap keeps the threshold honest and the inlier-ratio gate fires."""
        rng = np.random.default_rng(2)
        d = rng.uniform(0.0, 1.0, (H, W))  # no plane at all
        fit = fit_road_plane(d, PlaneFitConfig(seed=0))
        assert not fit.trusted
        assert fit.inlier_ratio < 0.40

    def test_failure_on_too_few_points(self):
        fit = fit_road_plane(np.full((8, 8), np.nan), PlaneFitConfig(seed=0))
        assert not fit.trusted
        assert fit.reason == "too_few_seed_points"

    def test_trapezoid_mask_geometry(self):
        m = trapezoid_mask((H, W))
        rows = np.nonzero(m.any(axis=1))[0]
        assert rows[0] == pytest.approx(0.55 * H, abs=2)
        assert rows[-1] == pytest.approx(0.97 * H, abs=2)
        assert m[rows[-1]].sum() > m[rows[0]].sum()  # widens toward the bottom
        assert not m[: rows[0]].any()


# ── residual binning / label maps ─────────────────────────────────────────────


class TestBins:
    def test_bin_residual_exact(self):
        cfg = ResidualLabelConfig()
        z = np.array([[-10.0, -4.0, -2.25, 0.0, 2.25, 4.0, 10.0, 2.0, -2.5]])
        labels = bin_residual(z, cfg)
        expected = [CLS_D2, CLS_D1, IGNORE_LABEL, CLS_F, IGNORE_LABEL,
                    CLS_E1, CLS_E2, CLS_F, CLS_D1]
        assert labels.tolist()[0] == expected

    def test_label_map_classes_at_injections(self):
        d = synth_scene(pit_amp=10 * SIGMA, mound_amp=4.5 * SIGMA)
        fit, z, d_surf = fitted(d)
        lm = compute_label_map(z, fit.inlier_mask, d_surf)
        # pit: deep depression (D2 majority); mound: moderate elevation (E1)
        assert (lm.labels[PIT] == CLS_D2).mean() > 0.8
        assert np.isin(lm.labels[MOUND], (CLS_E1, CLS_E2)).mean() > 0.8
        # flat near-field road is mostly F
        flat_block = lm.labels[100:115, 30:90]
        assert (flat_block == CLS_F).mean() > 0.8

    def test_far_field_invalidated(self):
        d = synth_scene()
        fit, z, d_surf = fitted(d)
        lm = compute_label_map(z, fit.inlier_mask, d_surf)
        assert lm.far_field_mask.any()
        # far field is the LOW-disparity (top) part of the road; all X
        far_rows = np.nonzero(lm.far_field_mask.any(axis=1))[0]
        near_rows = np.nonzero((lm.road_region & ~lm.far_field_mask).any(axis=1))[0]
        assert far_rows.mean() < near_rows.mean()
        assert (lm.labels[lm.far_field_mask] == CLS_X).all()

    def test_v_extent_gate_kills_tall_objects_keeps_bumps(self):
        d = synth_scene()
        # tall pole-like object: full road-height stripe of elevation
        d[40:158, 60:70] += 8 * SIGMA
        # speed bump: wide but short in v
        d[125:135, 130:200] += 8 * SIGMA
        fit, z, d_surf = fitted(d)
        lm = compute_label_map(z, fit.inlier_mask, d_surf)
        assert lm.suppressed_components >= 1
        assert (lm.labels[100:140, 62:68] == CLS_X).all()        # object -> X
        bump = lm.labels[127:133, 140:190]
        assert (np.isin(bump, (CLS_E1, CLS_E2))).mean() > 0.5    # bump survives

    def test_object_median_z_gate(self):
        d = synth_scene()
        d[125:140, 60:90] += 25 * SIGMA  # off-plane OBJECT amplitude
        fit, z, d_surf = fitted(d)
        lm = compute_label_map(z, fit.inlier_mask, d_surf)
        assert (lm.labels[128:137, 65:85] == CLS_X).all()


# ── box mining ────────────────────────────────────────────────────────────────


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def _rect_norm(sl_rows, sl_cols):
    return (sl_cols.start / W, sl_rows.start / H, sl_cols.stop / W, sl_rows.stop / H)


class TestBoxMining:
    def test_boxes_at_injected_locations(self):
        d = synth_scene(pit_amp=8 * SIGMA, mound_amp=8 * SIGMA)
        fit, z, d_surf = fitted(d)
        lm = compute_label_map(z, fit.inlier_mask, d_surf)
        boxes = mine_boxes(z, lm)
        dep = [b for b in boxes if b.cls == BOX_DEPRESSION]
        ele = [b for b in boxes if b.cls == BOX_ELEVATION]
        assert len(dep) == 1 and len(ele) == 1
        dep_rect = (dep[0].cx - dep[0].w / 2, dep[0].cy - dep[0].h / 2,
                    dep[0].cx + dep[0].w / 2, dep[0].cy + dep[0].h / 2)
        ele_rect = (ele[0].cx - ele[0].w / 2, ele[0].cy - ele[0].h / 2,
                    ele[0].cx + ele[0].w / 2, ele[0].cy + ele[0].h / 2)
        assert _iou(dep_rect, _rect_norm(*PIT)) > 0.5
        assert _iou(ele_rect, _rect_norm(*MOUND)) > 0.5
        assert dep[0].score > 6.0 and ele[0].score > 6.0

    def test_min_area_filter(self):
        d = synth_scene()
        d[130:136, 60:66] -= 8 * SIGMA  # 36 px << 256 px minimum
        fit, z, d_surf = fitted(d)
        lm = compute_label_map(z, fit.inlier_mask, d_surf)
        assert mine_boxes(z, lm) == []

    def test_yolo_lines_format(self):
        d = synth_scene(pit_amp=8 * SIGMA)
        geo = labels_from_inverse_depth(d, PlaneFitConfig(seed=0))
        lines = boxes_to_yolo_lines(geo.boxes)
        assert len(lines) == 1
        vals = lines[0].split()
        assert vals[0] == str(BOX_DEPRESSION)
        assert all(0.0 <= float(v) <= 1.0 for v in vals[1:])


# ── trust gates / full pipeline ───────────────────────────────────────────────


class TestTrustGates:
    def test_untrusted_fit_drops_all_supervision(self):
        rng = np.random.default_rng(3)
        geo = labels_from_inverse_depth(rng.uniform(0, 1, (H, W)),
                                        PlaneFitConfig(seed=0))
        assert not geo.use_dense and not geo.use_boxes
        assert geo.label_map is None and geo.boxes == []

    def test_sigma_mad_cap(self):
        d = synth_scene()
        cfg = ResidualLabelConfig(max_sigma_mad=SIGMA / 10)
        geo = labels_from_inverse_depth(d, PlaneFitConfig(seed=0), cfg)
        assert not geo.use_dense and "sigma_mad" in geo.reason

    def test_anomaly_area_cap_keeps_dense_drops_boxes(self):
        """Puddle-suspicion gate: anomaly area above the cap keeps F/X dense
        labels but switches anomaly pixels to ignore and mines no boxes."""
        d = synth_scene(pit_amp=8 * SIGMA)  # pit area 864 px ≈ 2.3% of road
        cfg = ResidualLabelConfig(max_anomaly_area_frac=0.005)
        geo = labels_from_inverse_depth(d, PlaneFitConfig(seed=0), cfg)
        assert geo.use_dense and not geo.use_boxes
        assert geo.boxes == []
        assert "anomaly_area" in geo.reason
        # anomaly pixels were switched to ignore, flat road kept
        assert (geo.label_map.labels[PIT] == IGNORE_LABEL).mean() > 0.5
        assert (geo.label_map.labels[100:115, 30:90] == CLS_F).mean() > 0.8
        assert not geo.label_map.anomaly_mask.any()

    def test_full_pipeline_ok(self):
        d = synth_scene(pit_amp=8 * SIGMA)
        geo = labels_from_inverse_depth(d, PlaneFitConfig(seed=0))
        assert geo.use_dense and geo.use_boxes and geo.reason == "ok"
        assert len(geo.boxes) == 1


# ── joint augmentation (R5) ───────────────────────────────────────────────────


class TestJointTransform:
    def _sample(self):
        img = torch.zeros(3, 80, 120)
        # channel 0 encodes normalized x, channel 1 normalized y
        xs = (torch.arange(120).float() + 0.5) / 120
        ys = (torch.arange(80).float() + 0.5) / 80
        img[0] = xs[None, :].expand(80, -1)
        img[1] = ys[:, None].expand(-1, 120)
        labels = torch.full((40, 60), CLS_X, dtype=torch.uint8)
        labels[20:36, 15:45] = CLS_F
        labels[26:32, 24:30] = CLS_D2
        boxes = torch.tensor([[0.0, 27 / 60, 29 / 40, 6 / 60, 6 / 40]])
        return img, labels, boxes

    def test_identity_crop(self):
        img, labels, boxes = self._sample()
        out_img, out_labels, out_boxes = joint_crop_flip(
            img, labels, boxes, (0.0, 0.0, 1.0, 1.0), False, 64)
        assert out_img.shape == (3, 64, 64)
        assert out_labels.shape == (64, 64)
        assert torch.allclose(out_boxes, boxes, atol=1e-6)
        # label classes preserved by nearest resize
        assert set(out_labels.unique().tolist()) == {CLS_D2, CLS_F, CLS_X}

    def test_flip_mirrors_all_three(self):
        img, labels, boxes = self._sample()
        a_img, a_lab, a_box = joint_crop_flip(img, labels, boxes,
                                              (0.0, 0.0, 1.0, 1.0), False, 64)
        b_img, b_lab, b_box = joint_crop_flip(img, labels, boxes,
                                              (0.0, 0.0, 1.0, 1.0), True, 64)
        assert torch.allclose(b_img, torch.flip(a_img, dims=[-1]))
        assert torch.equal(b_lab, torch.flip(a_lab, dims=[-1]))
        assert b_box[0, 1] == pytest.approx(1.0 - a_box[0, 1], abs=1e-6)
        assert torch.allclose(b_box[0, 2:], a_box[0, 2:], atol=1e-6)

    def test_crop_keeps_image_and_labels_aligned(self):
        """The coordinate-encoded image must agree with the label map after an
        arbitrary crop: anywhere the label says D2, the image x/y channels
        must read back coordinates inside the original D2 rectangle."""
        img, labels, boxes = self._sample()
        crop = (0.25, 0.45, 0.55, 0.5)
        out_img, out_labels, out_boxes = joint_crop_flip(
            img, labels, boxes, crop, False, 96)
        sel = out_labels == CLS_D2
        assert sel.any()
        xs, ys = out_img[0][sel], out_img[1][sel]
        # original D2 rect in normalized coords: x [24/60, 30/60), y [26/40, 32/40)
        tol = 0.03  # one label-grid cell
        assert xs.min() > 24 / 60 - tol and xs.max() < 30 / 60 + tol
        assert ys.min() > 26 / 40 - tol and ys.max() < 32 / 40 + tol
        # the box survived and overlaps the labeled pixels
        assert out_boxes.shape[0] == 1

    def test_box_dropped_when_cropped_out(self):
        img, labels, boxes = self._sample()
        # crop the left-top quarter — the box (right-bottom) disappears
        _, _, out_boxes = joint_crop_flip(img, labels, boxes,
                                          (0.0, 0.0, 0.3, 0.3), False, 64)
        assert out_boxes.shape[0] == 0


# ── dense loss machinery ──────────────────────────────────────────────────────


class TestOrdinalMachinery:
    def test_smoothing_matrix(self):
        t = ordinal_smoothing_matrix()
        assert t.shape == (NUM_CLASSES, NUM_CLASSES)
        assert torch.allclose(t.sum(dim=1), torch.ones(NUM_CLASSES))
        # interior bin: 0.8 true + 0.1 per neighbor
        assert t[CLS_F, CLS_F] == pytest.approx(0.8)
        assert t[CLS_F, CLS_D1] == pytest.approx(0.1)
        assert t[CLS_F, CLS_E1] == pytest.approx(0.1)
        assert t[CLS_F, CLS_X] == 0.0
        # endpoint bin: missing-neighbor mass returns to the true bin
        assert t[CLS_D2, CLS_D2] == pytest.approx(0.9)
        assert t[CLS_D2, CLS_D1] == pytest.approx(0.1)
        # X off-chain: one-hot
        assert t[CLS_X, CLS_X] == 1.0 and t[CLS_X].sum() == 1.0

    def test_majority_pool(self):
        labels = torch.full((1, 16, 16), CLS_F, dtype=torch.long)
        labels[0, :8, :8] = CLS_D2                     # pure D2 cell
        labels[0, 8:, :8] = IGNORE_LABEL               # pure ignore cell
        labels[0, :8, 8:12] = CLS_E2                   # half-cell: tie -> plurality
        pooled = majority_pool_labels(labels, (2, 2))
        assert pooled.shape == (1, 2, 2)
        assert pooled[0, 0, 0] == CLS_D2
        assert pooled[0, 1, 0] == IGNORE_LABEL
        assert pooled[0, 1, 1] == CLS_F
        assert pooled[0, 0, 1] in (CLS_E2, CLS_F)      # 50/50 tie

    def test_majority_pool_rejects_indivisible(self):
        with pytest.raises(ValueError, match="divisible"):
            majority_pool_labels(torch.zeros(1, 10, 10, dtype=torch.long), (3, 3))

    def test_balanced_sampling_ratios(self):
        torch.manual_seed(0)
        cells = torch.full((32, 32), CLS_F, dtype=torch.long)
        cells[:4, :4] = CLS_D1            # 16 anomaly cells
        cells[20:, 20:] = CLS_X           # plenty of X
        idx = sample_balanced_cells(cells, OrdinalLossConfig())
        lab = cells.reshape(-1)[idx]
        n_anom = sum(int((lab == c).sum()) for c in ANOMALY_CLASSES)
        assert n_anom == 16                              # ALL anomaly cells
        assert int((lab == CLS_F).sum()) == 48           # 3x per anomaly
        assert int((lab == CLS_X).sum()) == 16           # 1x per anomaly
        assert (lab != IGNORE_LABEL).all()

    def test_balanced_sampling_fallback_no_anomaly(self):
        torch.manual_seed(0)
        cells = torch.full((32, 32), CLS_F, dtype=torch.long)
        cells[16:, :] = CLS_X
        idx = sample_balanced_cells(cells, OrdinalLossConfig(fallback_cells=128))
        assert idx.numel() == 128
        lab = cells.reshape(-1)[idx]
        assert int((lab == CLS_F).sum()) == 64 and int((lab == CLS_X).sum()) == 64

    def test_balanced_sampling_cap(self):
        torch.manual_seed(0)
        cells = torch.full((64, 64), CLS_D1, dtype=torch.long)  # 4096 anomalies
        idx = sample_balanced_cells(cells, OrdinalLossConfig(max_cells_per_image=1024))
        assert idx.numel() == 1024

    def test_dense_head_shapes(self):
        head = DenseOrdinalHead(64)
        out = head(torch.randn(2, 64, 8, 8))
        assert out.shape == (2, NUM_CLASSES, 8, 8)
        n_params = sum(p.numel() for p in head.parameters())
        assert n_params < 60_000  # ~50K spec budget


# ── pool dataset / loader ─────────────────────────────────────────────────────


def _synth_channel_sample(seed: int):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(3, 96, 96, generator=g)
    labels = torch.full((48, 48), CLS_X, dtype=torch.uint8)
    labels[20:44, 8:40] = CLS_F
    labels[30:36, 14:20] = CLS_D2
    labels[31:35, 30:36] = CLS_E2
    boxes = torch.tensor([
        [float(BOX_DEPRESSION), 17 / 48, 33 / 48, 6 / 48, 6 / 48],
        [float(BOX_ELEVATION), 33 / 48, 33 / 48, 6 / 48, 4 / 48],
    ])
    return {"img": img, "labels": labels, "boxes": boxes}


class TestPoolDataset:
    def test_requires_exactly_one_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            TerraPoolDataset()
        with pytest.raises(ValueError, match="exactly one"):
            TerraPoolDataset(samples=[], root=".")

    def test_getitem_and_collate(self):
        ds = TerraPoolDataset(samples=[_synth_channel_sample(i) for i in range(3)],
                              imgsz=64, seed=0)
        items = [ds[i] for i in range(3)]
        for it in items:
            assert it["img"].shape == (3, 64, 64)
            assert it["img"].min() >= 0 and it["img"].max() <= 1
            assert it["labels"].shape == (64, 64)
            assert it["labels"].dtype == torch.long
            assert it["boxes"].shape[1] == 5
        batch = terra_collate(items)
        assert batch["img"].shape == (1 * 3, 3, 64, 64)
        n = batch["batch_idx"].shape[0]
        assert batch["cls"].shape == (n, 1) and batch["bboxes"].shape == (n, 4)
        if n:
            assert batch["batch_idx"].max() <= 2

    def test_photometric_jitter_touches_pixels_only(self):
        from yolo_contrastive.geoteach.channel import TerraAugConfig

        sample = _synth_channel_sample(0)
        fixed = dict(scale=(1.0, 1.0), ratio=(1.0, 1.0), hflip_prob=0.0)
        ds0 = TerraPoolDataset(samples=[sample], imgsz=64, seed=0,
                               aug=TerraAugConfig(photometric=0.0, **fixed))
        ds1 = TerraPoolDataset(samples=[sample], imgsz=64, seed=0,
                               aug=TerraAugConfig(photometric=0.5, **fixed))
        it0, it1 = ds0[0], ds1[0]
        # identical full-frame geometry -> labels/boxes identical (R5: jitter
        # is pixels-only, after geometry); pixels did change and stay in range
        assert torch.equal(it0["labels"], it1["labels"])
        assert torch.equal(it0["boxes"], it1["boxes"])
        assert not torch.equal(it0["img"], it1["img"])
        assert it1["img"].min() >= 0 and it1["img"].max() <= 1

    def test_label_classes_survive_augmentation(self):
        ds = TerraPoolDataset(samples=[_synth_channel_sample(0)], imgsz=64, seed=3)
        seen = set()
        for _ in range(8):
            seen |= set(ds[0]["labels"].unique().tolist())
        assert seen <= {CLS_D2, CLS_F, CLS_E2, CLS_X, IGNORE_LABEL}
        assert CLS_F in seen and CLS_X in seen

    def test_directory_mode(self, tmp_path):
        # Force the production environment: importing ultralytics monkeypatches
        # cv2.imread process-wide (grayscale reads become [H, W, 1]).
        import ultralytics  # noqa: F401
        import cv2

        root = tmp_path / "factory"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "boxes").mkdir()
        img = (np.random.default_rng(0).uniform(0, 255, (96, 96, 3))).astype(np.uint8)
        cv2.imwrite(str(root / "images" / "a.png"), img)
        lab = np.full((48, 48), CLS_X, dtype=np.uint8)
        lab[20:40, 10:40] = CLS_F
        cv2.imwrite(str(root / "labels" / "a.png"), lab)
        (root / "boxes" / "a.txt").write_text("0 0.5 0.6 0.1 0.1\n", encoding="utf-8")
        ds = TerraPoolDataset(root=str(root), imgsz=64, seed=0)
        assert len(ds) == 1
        item = ds[0]
        assert item["img"].shape == (3, 64, 64)
        assert item["labels"].shape == (64, 64)


# ── channel through the real anchored trainer ─────────────────────────────────


@pytest.fixture(scope="module")
def terra_trainer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("terra_run")
    torch.manual_seed(0)
    channel = TerraChannel(samples=[_synth_channel_sample(i) for i in range(4)], seed=0)
    trainer = AnchoredJointTrainer(
        model="yolov8n.yaml",  # offline: yaml ships inside ultralytics
        channels=[channel],
        lambda_aux=1.0,
        epochs=1,
        imgsz=64,
        batch=2,
        warmup_steps=1,
        device="cpu",
        amp=False,
        output_dir=str(tmp),
        sentinel_thresholds=RELAXED,
    )
    yield trainer, channel
    trainer.cleanup()


def _replay_batch(b: int = 2, s: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = 2 * b
    return {
        "img": torch.rand(b, 3, s, s, generator=g),
        "batch_idx": torch.arange(b).repeat_interleave(2).float(),
        "cls": torch.randint(0, 80, (n, 1), generator=g).float(),
        "bboxes": torch.cat(
            [0.3 + 0.4 * torch.rand(n, 2, generator=g),
             0.05 + 0.2 * torch.rand(n, 2, generator=g)], dim=1),
    }


class TestTerraChannel:
    def test_attach_built_both_heads(self, terra_trainer):
        trainer, channel = terra_trainer
        heads = trainer.heads["terra"]
        assert isinstance(heads, nn.ModuleList) and len(heads) == 2
        assert channel.dense_head is heads[0]
        assert channel.geo_head is heads[1]
        assert channel.geo_head.nc == 2

    def test_steps_produce_finite_terms_and_train_heads(self, terra_trainer):
        trainer, channel = terra_trainer
        loader = channel.build_loader(
            {"imgsz": 64, "batch": 2, "workers": 0, "device": "cpu"})
        batches = list(loader)
        assert len(batches) == 2
        w0_dense = channel.dense_head.conv1.weight.detach().clone()
        w0_geo = channel.geo_head.detect.cv2[0][0].conv.weight.detach().clone()
        for i, cb in enumerate(batches):
            m = trainer.step(_replay_batch(seed=i), {"terra": cb})
            for key in ("replay/det_loss", "terra/ordinal", "terra/geobox", "terra/total"):
                assert key in m and np.isfinite(m[key]), key
            assert m["terra/total"] == pytest.approx(
                m["terra/ordinal"] + m["terra/geobox"], rel=1e-5)
        assert not torch.allclose(channel.dense_head.conv1.weight, w0_dense)
        assert not torch.allclose(channel.geo_head.detect.cv2[0][0].conv.weight, w0_geo)

    def test_geo_dfl_stays_frozen(self, terra_trainer):
        _, channel = terra_trainer
        # E5 guard: re-frozen on the first loss() despite the trainer's
        # blanket requires_grad enable on channel heads
        assert channel.geo_head.detect.dfl.conv.weight.requires_grad is False

    def test_channel_heads_not_in_model_or_export(self, terra_trainer):
        trainer, channel = terra_trainer
        model_param_ids = {id(p) for p in trainer.model.parameters()}
        for head in (channel.dense_head, channel.geo_head):
            for p in head.parameters():
                assert id(p) not in model_param_ids

    def test_beta_scales_geobox(self):
        assert TerraChannel(samples=[], beta=0.0).beta == 0.0
        with pytest.raises(ValueError, match="beta"):
            TerraChannel(samples=[], beta=-1.0)


# ── depth cache ───────────────────────────────────────────────────────────────


class _StubPipe:
    """transformers depth-estimation pipeline stand-in (no downloads)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, images, batch_size=1):
        import torch as _t

        self.calls += len(images)
        out = []
        for im in images:
            w, h = im.size
            depth = _t.linspace(0.1, 1.0, h).view(h, 1).expand(h, w)
            out.append({"predicted_depth": depth.clone()})
        return out


class TestDepthCache:
    def test_save_load_roundtrip(self, tmp_path):
        cache = DepthCache(tmp_path, tag="t")
        d = synth_scene(pit_amp=8 * SIGMA)
        cache.save("sub/img1", d.astype(np.float32), meta={"model_name": "x"})
        assert cache.has("sub/img1") and "sub/img1" in cache
        loaded, meta = cache.load("sub/img1")
        assert loaded.shape == d.shape
        span = float(d.max() - d.min())
        assert np.max(np.abs(loaded - d)) <= span / 65535 * 1.01  # uint16 quantization
        assert meta["model_name"] == "x"
        assert len(cache) == 1 and cache.image_ids() == ["sub/img1"]

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DepthCache(tmp_path, tag="t").load("nope")

    def test_run_resumable_with_stub_pipe(self, tmp_path):
        rng = np.random.default_rng(0)
        imgs = [(f"im{i}", rng.integers(0, 255, (40, 60, 3), dtype=np.uint8).astype(np.uint8))
                for i in range(3)]
        cache = DepthCache(tmp_path, tag="stub")
        pipe = _StubPipe()
        stats = run_depth_anything(imgs, cache, batch_size=2, pipe=pipe,
                                   model_name="stub-relative")
        assert stats == {"scanned": 3, "skipped": 0, "computed": 3, "errors": 0}
        assert pipe.calls == 3 and len(cache) == 3
        d, meta = cache.load("im0")
        assert d.shape == (20, 30)  # half resolution
        assert meta["metric"] is False
        assert cache.load_metadata()["model_name"] == "stub-relative"
        # resume: nothing recomputed
        stats2 = run_depth_anything(imgs, cache, batch_size=2, pipe=pipe,
                                    model_name="stub-relative")
        assert stats2["skipped"] == 3 and stats2["computed"] == 0
        assert pipe.calls == 3

    def test_metric_model_inverted(self, tmp_path):
        imgs = [("m0", np.zeros((40, 60, 3), dtype=np.uint8))]
        cache = DepthCache(tmp_path, tag="metric")
        run_depth_anything(imgs, cache, pipe=_StubPipe(),
                           model_name="Depth-Anything-V2-Metric-Outdoor-Small-hf")
        d, meta = cache.load("m0")
        assert meta["metric"] is True and meta["depth_unit"] == "meters"
        # stub depth increases downward -> INVERSE depth must decrease downward
        assert d[0].mean() > d[-1].mean()

    def test_metric_clip_band_applied_and_synced_with_scalereal(self, tmp_path):
        from yolo_contrastive.geoteach.depth_cache import METRIC_Z_CLIP
        from yolo_contrastive.scalereal.depth_io import METRIC_Z_CLIP as SR_CLIP

        assert METRIC_Z_CLIP == SR_CLIP        # shared-cache contract
        imgs = [("c0", np.zeros((40, 60, 3), dtype=np.uint8))]
        cache = DepthCache(tmp_path, tag="metric_clip")
        run_depth_anything(imgs, cache, pipe=_StubPipe(),
                           model_name="Depth-Anything-V2-Metric-Outdoor-Small-hf")
        d, meta = cache.load("c0")
        lo, hi = METRIC_Z_CLIP
        assert meta["z_clip"] == [lo, hi]      # band recorded in the sidecar
        # stub depth spans 0.1..1.0 m -> clamped to [0.5, 1.0] BEFORE inversion,
        # so inverse depth is capped at 1/lo instead of blowing up to 10
        assert float(d.max()) == pytest.approx(1.0 / lo, rel=1e-3)
        assert float(d.min()) >= 1.0 / hi - 1e-6


# ── import hygiene ────────────────────────────────────────────────────────────


def test_package_import_stays_light():
    import yolo_contrastive  # noqa: F401 (E2)
    import yolo_contrastive.geoteach as geoteach

    assert hasattr(geoteach, "TerraChannel")
    assert hasattr(geoteach, "fit_road_plane")
