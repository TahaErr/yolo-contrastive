"""Tests for the REVISIT persistence channel (persistence/).

CPU-only, offline, no model downloads: alignment runs on synthetic textured
"road" images warped by a KNOWN homography, persistence labels are recovered
from analytic blob layouts, the correspondence loss is verified minimal at
the true alignment on a pixel-unshuffle toy backbone, and the channel is
integrated against a tiny conv stub plus (once) the real anchored trainer
built from ``yolov8n.yaml`` (ships inside ultralytics — no .pt download).

Network-dependent paths (Mapillary mining/download) are exercised against an
injected fake transport returning canned Graph-API JSON — zero network.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_contrastive.dense.multi_scale_tap import MultiScaleFeatureTap
from yolo_contrastive.exceptions import FeatureTapError
from yolo_contrastive.persistence import pair_manifest as pm
from yolo_contrastive.persistence.align import (
    AlignConfig,
    align_pair,
    degeneracy_ok,
    from_normalized_h,
    h_from_row,
    h_to_row,
    points_in_quad,
    to_normalized_h,
    warp_boxes_h,
    warp_points_h,
)
from yolo_contrastive.persistence.channel import PersistenceChannel
from yolo_contrastive.persistence.heads import PersistenceHead, simsiam_one_way
from yolo_contrastive.persistence.mapillary_pairs import (
    PairGateConfig,
    download_images,
    haversine_m,
    heading_diff_deg,
    mine_pairs,
    mine_tile_pairs,
)
from yolo_contrastive.persistence.pair_aug import (
    ViewAugConfig,
    apply_theta_to_images,
    make_theta,
    rasterize_label_map,
    sample_view_theta,
    transform_boxes,
    transform_points,
    valid_points_mask,
)
from yolo_contrastive.persistence.pair_dataset import PairDataset
from yolo_contrastive.persistence.persistence_labels import (
    LABEL_IGNORE,
    LABEL_PERSISTENT,
    LABEL_TRANSIENT,
    PersistenceLabelConfig,
    match_proposals,
)
from yolo_contrastive.persistence.proposals import (
    box_iou_matrix,
    cheap_proposals,
    nms_boxes,
)

cv2 = pytest.importorskip("cv2", reason="opencv (pretrain extra) required")

# one-meter offsets in degrees at the equator (tests place cities at lat 0)
M_LAT = 1.0 / 111_320.0


# ── import hygiene ────────────────────────────────────────────────────────────


def test_package_import_is_light():
    import yolo_contrastive  # noqa: F401  (E2)
    import yolo_contrastive.persistence as p

    assert "PersistenceChannel" in dir(p)
    assert p.PersistenceChannel is PersistenceChannel


# ── manifests (schema round trip, idempotent append, status transitions) ──────


def _pair_row(pair_id="p1", **kw):
    base = dict(
        pair_id=pair_id, img_a_id=f"{pair_id}a", img_b_id=f"{pair_id}b",
        lon_a=0.0, lat_a=0.0, lon_b=0.0, lat_b=0.0, dist_m=3.0,
        heading_a=10.0, heading_b=12.0, heading_diff=2.0,
        captured_at_a=0, captured_at_b=100 * 86_400_000, dt_days=100.0,
        seq_a="s1", seq_b="s2", city="test", tile_id="t0",
    )
    base.update(kw)
    return pm.new_pair_row(**base)


class TestManifest:
    def test_pairs_roundtrip_and_idempotent_append(self, tmp_path):
        path = tmp_path / "pairs.parquet"
        rows = [_pair_row("p1"), _pair_row("p2")]
        assert pm.append_pairs(path, rows) == 2
        assert pm.append_pairs(path, rows) == 0          # idempotent
        df = pm.read_pairs(path)
        assert list(df.columns) == pm.PAIRS_COLUMNS
        assert set(df["pair_id"]) == {"p1", "p2"}
        assert (df["status"] == "queued").all()

    def test_status_transition_and_h_columns(self, tmp_path):
        path = tmp_path / "pairs.parquet"
        pm.append_pairs(path, [_pair_row("p1")])
        h = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.02], [0.01, 0.0, 1.0]])
        n = pm.update_pairs(path, {"p1": {"status": "aligned", "align_ok": True,
                                          **h_to_row(h)}})
        assert n == 1
        row = pm.read_pairs(path).iloc[0]
        assert row["status"] == "aligned" and bool(row["align_ok"])
        assert np.allclose(h_from_row(row), h)

    def test_invalid_status_raises(self, tmp_path):
        path = tmp_path / "pairs.parquet"
        pm.append_pairs(path, [_pair_row("p1")])
        with pytest.raises(ValueError, match="status"):
            pm.update_pairs(path, {"p1": {"status": "bogus"}})

    def test_new_pair_row_validates(self):
        with pytest.raises(ValueError, match="missing"):
            pm.new_pair_row(pair_id="x")
        with pytest.raises(ValueError, match="unknown"):
            _pair_row("p1", not_a_column=1)

    def test_proposals_and_labels_dedup(self, tmp_path):
        prop = {"image_id": "i1", "prop_id": "i1_000", "x1": 0.1, "y1": 0.1,
                "x2": 0.2, "y2": 0.2, "score": 0.5, "backend": "cheap"}
        lab = {"label_id": "p1_a_000", "pair_id": "p1", "image_id": "i1",
               "side": "a", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2,
               "label": "persistent"}
        assert pm.append_proposals(tmp_path / "p.parquet", [prop, prop]) == 1
        assert pm.append_proposals(tmp_path / "p.parquet", [prop]) == 0
        assert pm.append_labels(tmp_path / "l.parquet", [lab]) == 1
        assert pm.labeled_pair_ids(tmp_path / "l.parquet") == {"p1"}


# ── miner against a fake Graph-API transport (zero network) ───────────────────


def _api_img(img_id, lat, lon, heading=0.0, days=0.0, seq="s1",
             pano=False, cam="perspective"):
    return {
        "id": img_id,
        "captured_at": int(days * 86_400_000),
        "compass_angle": heading,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "sequence": seq,
        "is_pano": pano,
        "camera_type": cam,
    }


def _norm(entries):
    from yolo_contrastive.persistence.mapillary_pairs import _normalize_entry

    return [_normalize_entry(e) for e in entries]


class TestMiner:
    def test_geo_helpers(self):
        assert abs(haversine_m(0.0, 0.0, M_LAT, 0.0) - 1.0) < 0.01
        assert heading_diff_deg(350.0, 10.0) == pytest.approx(20.0)
        assert heading_diff_deg(0.0, 180.0) == pytest.approx(180.0)

    def test_pair_gates(self):
        # each group is placed ~1.1 km from the others so groups can't pair up
        entries = [
            # valid: different seq, 5 m apart, dt 60 d, heading diff 10
            _api_img("a1", 0.0, 0.0, heading=0, days=0, seq="s1"),
            _api_img("a2", 0.0, 5 * M_LAT, heading=10, days=60, seq="s2"),
            # same sequence -> rejected
            _api_img("b1", 0.01, 0.0, days=0, seq="s3"),
            _api_img("b2", 0.01, 2 * M_LAT, days=60, seq="s3"),
            # dt 5 days -> rejected
            _api_img("c1", 0.02, 0.0, days=0, seq="s4"),
            _api_img("c2", 0.02, 2 * M_LAT, days=5, seq="s5"),
            # heading diff 90 -> rejected
            _api_img("d1", 0.03, 0.0, heading=0, days=0, seq="s6"),
            _api_img("d2", 0.03, 2 * M_LAT, heading=90, days=60, seq="s7"),
            # pano -> rejected
            _api_img("e1", 0.04, 0.0, days=0, seq="s8"),
            _api_img("e2", 0.04, 2 * M_LAT, days=60, seq="s9", pano=True),
            # 100 m apart -> rejected
            _api_img("f1", 0.05, 0.0, days=0, seq="s10"),
            _api_img("f2", 0.05, 100 * M_LAT, days=60, seq="s11"),
            # fisheye -> rejected
            _api_img("g1", 0.06, 0.0, days=0, seq="s12"),
            _api_img("g2", 0.06, 2 * M_LAT, days=60, seq="s13", cam="fisheye"),
        ]
        pairs = mine_tile_pairs(_norm(entries))
        assert [c["pair_id"] for c in pairs] == ["a1_a2"]

    def test_midpoint_nms_suppresses_co_traversed_street(self):
        # two sequences driving the same 7 m of street -> many raw candidates,
        # greedy 8 m midpoint NMS keeps exactly one
        entries = []
        for i, x in enumerate((0, 2, 4, 6)):
            entries.append(_api_img(f"a{i}", 0.0, x * M_LAT, days=0, seq="sa"))
        for i, x in enumerate((1, 3, 5, 7)):
            entries.append(_api_img(f"b{i}", 0.0, x * M_LAT, days=60, seq="sb"))
        pairs = mine_tile_pairs(_norm(entries))
        assert len(pairs) == 1

    def test_location_cell_cap_with_many_traversals(self):
        # 4 different sequences at one location -> 6 valid pairs, capped at 3
        entries = [
            _api_img(f"x{i}", 0.0, 0.0, days=i * 45, seq=f"q{i}") for i in range(4)
        ]
        pairs = mine_tile_pairs(_norm(entries))
        assert len(pairs) == PairGateConfig().max_pairs_per_cell == 3

    def test_mine_pairs_end_to_end_and_idempotent(self, tmp_path):
        calls = []

        def fake_json(url, params):
            calls.append(url)
            assert url.endswith("/images") and "bbox" in params
            return {"data": [
                _api_img("a1", 0.0, 0.0, days=0, seq="s1"),
                _api_img("a2", 0.0, 5 * M_LAT, heading=5, days=60, seq="s2"),
            ]}

        city = [{"name": "test", "lat": 0.0, "lon": 0.0, "radius_km": 0.05}]
        path = tmp_path / "pairs.parquet"
        assert mine_pairs(path, cities=city, fetch_json=fake_json) == 1
        assert len(calls) == 1                            # 0.05 km -> single tile
        assert mine_pairs(path, cities=city, fetch_json=fake_json) == 0  # dedup
        row = pm.read_pairs(path).iloc[0]
        assert row["pair_id"] == "a1_a2" and row["status"] == "queued"
        assert row["dt_days"] == pytest.approx(60.0)
        assert row["dist_m"] == pytest.approx(5.0, abs=0.05)

    def test_download_with_fake_transport(self, tmp_path):
        from PIL import Image

        path = tmp_path / "pairs.parquet"
        pm.append_pairs(path, [_pair_row("p1", img_a_id="img_a", img_b_id="img_b")])

        buf = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(buf, format="JPEG")
        jpeg = buf.getvalue()

        def fake_json(url, params):
            assert params["fields"].startswith("thumb_2048")
            return {"thumb_2048_url": f"{url}/thumb.jpg"}

        fetched = download_images(path, tmp_path / "pool", fetch_json=fake_json,
                                  fetch_bytes=lambda url: jpeg, sleep_s=0.0)
        assert fetched == 2
        row = pm.read_pairs(path).iloc[0]
        assert row["status"] == "downloaded"
        assert (tmp_path / "pool" / "images" / "img_a.jpg").exists()
        assert row["path_a"].endswith("img_a.jpg")
        # resumable: second run touches nothing
        assert download_images(path, tmp_path / "pool", fetch_json=fake_json,
                               fetch_bytes=lambda url: jpeg, sleep_s=0.0) == 0


# ── alignment: analytic homography recovery + gates ───────────────────────────


def _road_image(seed: int, size: int = 256) -> np.ndarray:
    """Synthetic textured road: smoothed noise + dark/bright discs (uint8)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(70, 190, (size, size)).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    cv2.circle(img, (int(0.4 * size), int(0.8 * size)), size // 18, 30, -1)
    cv2.circle(img, (int(0.7 * size), int(0.65 * size)), size // 24, 230, -1)
    cv2.circle(img, (int(0.2 * size), int(0.6 * size)), size // 28, 40, -1)
    return img


def _gt_homography(size: int = 256) -> np.ndarray:
    """Mild perspective warp (corner shifts of a few px)."""
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    dst = np.float32([[4, 2], [size - 3, -2], [size + 2, size - 4], [-2, size + 3]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


class TestAlign:
    CFG = AlignConfig(long_side=256)

    def test_recovers_known_homography(self):
        img_a = _road_image(0)
        h_gt = _gt_homography()
        img_b = cv2.warpPerspective(img_a, h_gt, (256, 256))
        # mild brightness shift + noise (real cross-session variation proxy)
        rng = np.random.default_rng(1)
        img_b = np.clip(img_b.astype(np.int16) + 10
                        + rng.normal(0, 3, img_b.shape), 0, 255).astype(np.uint8)

        res = align_pair(img_a, img_b, self.CFG)
        assert res.ok, res.reason
        assert res.n_inliers >= self.CFG.min_inliers
        assert res.reproj_rmse <= self.CFG.max_rmse_px

        # corner-transfer error on the ROI region, at working resolution
        h_rec = from_normalized_h(res.h_norm, (256, 256), (256, 256))
        pts = np.array([[30.0, 110.0], [226.0, 110.0], [226.0, 246.0], [30.0, 246.0]])
        err = np.linalg.norm(warp_points_h(h_rec, pts) - warp_points_h(h_gt, pts),
                             axis=1).mean()
        assert err < 1.5, f"corner transfer error {err:.2f}px"

    def test_normalized_h_roundtrip(self):
        h = np.array([[1.05, 0.02, 12.0], [-0.01, 0.97, -5.0], [1e-5, -2e-5, 1.0]])
        hn = to_normalized_h(h, (480, 640), (512, 384))
        hp = from_normalized_h(hn, (480, 640), (512, 384))
        assert np.allclose(hp, h / h[2, 2], atol=1e-9)
        # direction lock: x_b = H_norm @ x_a matches the pixel-space mapping
        p_px = np.array([[320.0, 360.0]])
        p_norm = p_px / np.array([640.0, 480.0])
        out_norm = warp_points_h(hn, p_norm)
        out_px = warp_points_h(h, p_px)
        assert np.allclose(out_norm * np.array([384.0, 512.0]), out_px, atol=1e-6)

    def test_unrelated_noise_pair_rejected(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 255, (256, 256)).astype(np.uint8)
        b = rng.integers(0, 255, (256, 256)).astype(np.uint8)
        res = align_pair(a, b, self.CFG)
        assert not res.ok
        assert res.reason  # a gate must have fired

    def test_degeneracy_guard(self):
        cfg = AlignConfig()
        # rank-deficient upper 2x2
        bad = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ok, reason = degeneracy_ok(bad, cfg)
        assert not ok
        # extreme scale collapse
        tiny = np.diag([0.01, 0.01, 1.0])
        ok, reason = degeneracy_ok(tiny, cfg)
        assert not ok and "scale" in reason or "det" in reason
        # identity passes
        ok, _ = degeneracy_ok(np.eye(3), cfg)
        assert ok

    def test_warp_boxes_axis_aligned_envelope(self):
        h = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, -0.05], [0.0, 0.0, 1.0]])
        out = warp_boxes_h(h, np.array([[0.2, 0.4, 0.3, 0.5]]))
        assert np.allclose(out, [[0.3, 0.35, 0.4, 0.45]])

    def test_points_in_quad(self):
        quad = np.array([[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]])
        pts = np.array([[0.5, 0.7], [0.5, 0.2], [1.5, 0.7], [0.5, np.inf]])
        assert points_in_quad(pts, quad).tolist() == [True, False, False, False]
        # reversed orientation must give the same answer
        assert points_in_quad(pts, quad[::-1]).tolist() == [True, False, False, False]


# ── persistence labels: analytic recovery ─────────────────────────────────────


class TestPersistenceLabels:
    CFG = PersistenceLabelConfig()

    def test_matcher_geometry_identity_h(self):
        h = np.eye(3)
        boxes_a = np.array([
            [0.40, 0.70, 0.50, 0.80],    # present in both -> PERSISTENT
            [0.20, 0.65, 0.30, 0.75],    # only in A, inside overlap -> TRANSIENT
            [0.60, 0.60, 0.70, 0.70],    # IoU ~0.2 band vs b[1] -> ignore
            [0.40, 0.10, 0.50, 0.20],    # outside the ROI/overlap -> ignore
        ])
        boxes_b = np.array([
            [0.40, 0.70, 0.50, 0.80],
            [0.667, 0.60, 0.767, 0.70],  # IoU ~0.2 with a[2]
        ])
        la, lb = match_proposals(boxes_a, boxes_b, h, self.CFG)
        assert la.tolist() == [LABEL_PERSISTENT, LABEL_TRANSIENT,
                               LABEL_IGNORE, LABEL_IGNORE]
        assert lb.tolist() == [LABEL_PERSISTENT, LABEL_IGNORE]

    def test_matcher_geometry_translation_h(self):
        # x_b = x_a + 0.1: the persistent box appears shifted in B
        h = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        boxes_a = np.array([
            [0.40, 0.70, 0.50, 0.80],    # counterpart at +0.1 in B -> PERSISTENT
            [0.20, 0.65, 0.30, 0.75],    # no counterpart, inside overlap -> TRANSIENT
        ])
        boxes_b = np.array([[0.50, 0.70, 0.60, 0.80]])
        la, lb = match_proposals(boxes_a, boxes_b, h, self.CFG)
        assert la.tolist() == [LABEL_PERSISTENT, LABEL_TRANSIENT]
        assert lb.tolist() == [LABEL_PERSISTENT]

    def test_asymmetric_evidence_no_transient_off_frame(self):
        # a box whose location is NOT visible in B (warps off-frame) must be
        # ignored, never marked transient
        h = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        boxes_a = np.array([[0.80, 0.70, 0.90, 0.80]])   # -> x_b 1.3..1.4, off-frame
        la, _ = match_proposals(boxes_a, np.zeros((0, 4)), h, self.CFG)
        assert la.tolist() == [LABEL_IGNORE]

    def test_end_to_end_blobs(self):
        # dark discs on light asphalt; one persists across sessions, one is
        # transient (only in A). H = pure translation of +32 px in x.
        size = 256
        base = np.full((size, size), 170, np.uint8)
        img_a, img_b = base.copy(), base.copy()
        cv2.circle(img_a, (128, 180), 12, 60, -1)        # persistent (A)
        cv2.circle(img_a, (80, 150), 10, 50, -1)         # transient (A only)
        cv2.circle(img_b, (160, 180), 12, 60, -1)        # persistent (B, shifted)
        h = np.array([[1.0, 0.0, 32.0 / size], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        props_a = cheap_proposals(img_a)
        props_b = cheap_proposals(img_b)
        assert props_a.shape[0] >= 2 and props_b.shape[0] >= 1

        la, lb = match_proposals(props_a[:, :4], props_b[:, :4], h, self.CFG)

        disc_p = np.array([116, 168, 140, 192]) / size   # persistent disc in A
        disc_t = np.array([70, 140, 90, 160]) / size     # transient disc in A
        iou_p = box_iou_matrix(props_a[:, :4], disc_p[None]).ravel()
        iou_t = box_iou_matrix(props_a[:, :4], disc_t[None]).ravel()
        assert iou_p.max() > 0.5 and iou_t.max() > 0.5   # both discs proposed
        assert set(la[iou_p > 0.5].tolist()) == {LABEL_PERSISTENT}
        assert set(la[iou_t > 0.5].tolist()) == {LABEL_TRANSIENT}
        # and the persistent disc is recovered from the B side too
        disc_pb = np.array([148, 168, 172, 192]) / size
        iou_pb = box_iou_matrix(props_b[:, :4], disc_pb[None]).ravel()
        assert set(lb[iou_pb > 0.5].tolist()) == {LABEL_PERSISTENT}

    def test_nms_caps_and_dedups(self):
        boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.11, 0.11, 0.31, 0.31],
                          [0.6, 0.6, 0.8, 0.8]])
        keep = nms_boxes(boxes, np.array([0.9, 0.8, 0.7]), iou_thresh=0.7)
        assert keep.tolist() == [0, 2]


# ── pair augmentation: R5 round-trip properties ───────────────────────────────


def _coord_image(size: int = 64) -> torch.Tensor:
    """[1, 3, S, S] image whose channels 0/1 encode each pixel's (x, y)
    center coordinate — bilinear sampling of a linear function is exact."""
    c = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    xs = c.view(1, -1).expand(size, size)
    ys = c.view(-1, 1).expand(size, size)
    return torch.stack([xs, ys, torch.full_like(xs, 0.5)])[None]


class TestPairAug:
    def test_points_land_on_same_content(self):
        img = _coord_image(64)
        theta = make_theta(cx=0.55, cy=0.5, half_w=0.3, half_h=0.25, flip=False)
        view = apply_theta_to_images(img, torch.from_numpy(theta)[None], (64, 64))

        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(0.30, 0.80, 24), rng.uniform(0.30, 0.70, 24)], 1)
        v = transform_points(theta, pts)
        assert valid_points_mask(v, 0.02).all()

        grid = torch.from_numpy(v).float().view(1, 1, -1, 2) * 2.0 - 1.0
        sampled = F.grid_sample(view, grid, mode="bilinear", align_corners=False)
        sampled = sampled.squeeze().permute(1, 0).numpy()    # [N, 3]
        # view content at the transformed coords == original content (coords)
        assert np.abs(sampled[:, 0] - pts[:, 0]).max() < 0.02
        assert np.abs(sampled[:, 1] - pts[:, 1]).max() < 0.02

    def test_hflip_exact_sign_flip(self):
        theta = make_theta(0.5, 0.5, 0.5, 0.5, flip=True)   # full-frame flip
        pts = np.array([[0.2, 0.3], [0.7, 0.9]])
        v = transform_points(theta, pts)
        assert np.allclose(v[:, 0], 1.0 - pts[:, 0], atol=1e-12)
        assert np.allclose(v[:, 1], pts[:, 1], atol=1e-12)
        boxes = transform_boxes(theta, np.array([[0.1, 0.2, 0.4, 0.5]]))
        assert np.allclose(boxes, [[0.6, 0.2, 0.9, 0.5]], atol=1e-12)

    def test_validity_mask_excludes_cropped_points(self):
        theta = make_theta(0.25, 0.5, 0.25, 0.5, flip=False)  # left half crop
        pts = np.array([[0.1, 0.5], [0.4, 0.5], [0.6, 0.5], [0.9, 0.5]])
        v = transform_points(theta, pts)
        assert valid_points_mask(v, 0.02).tolist() == [True, True, False, False]

    def test_raster_identity_conservation_and_bg_cap(self):
        g = 64
        boxes = np.array([[0.3, 0.3, 0.55, 0.55]])
        classes = np.array([1])
        quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        roi = np.array([0.0, 0.4, 1.0, 1.0])
        rng = np.random.default_rng(0)

        direct = rasterize_label_map(g, boxes, classes, quad, roi, rng)
        # analytic: 16 cells span -> 14^2 interior after the 1-cell erode
        assert int((direct == 1).sum()) == 14 * 14
        assert int((direct == 2).sum()) == 0
        # background cap: exactly 3x foreground (plenty of candidates here)
        assert int((direct == 0).sum()) == 3 * 14 * 14

        # identity theta reproduces the direct rasterization (fg conserved)
        tid = make_theta(0.5, 0.5, 0.5, 0.5, flip=False)
        via = rasterize_label_map(
            g, transform_boxes(tid, boxes), classes, quad, roi,
            np.random.default_rng(0))
        assert int((via == 1).sum()) == int((direct == 1).sum())

    def test_raster_fg_scales_with_zoom(self):
        g = 64
        boxes = np.array([[0.3, 0.3, 0.55, 0.55]])
        classes = np.array([1])
        rng = np.random.default_rng(0)
        n_id = int((rasterize_label_map(
            g, boxes, classes, None, None, rng) == 1).sum())
        # zoom-in crop (quarter area) containing the box -> ~4x the fg cells
        zoom = make_theta(0.425, 0.425, 0.25, 0.25, flip=False)
        n_zoom = int((rasterize_label_map(
            g, transform_boxes(zoom, boxes), classes, None, None, rng) == 1).sum())
        assert n_zoom > 2 * n_id

    def test_raster_bg_floor_when_no_proposals(self):
        g = 64
        quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        roi = np.array([0.0, 0.4, 1.0, 1.0])
        no_boxes = np.zeros((0, 4))
        no_classes = np.zeros(0, dtype=np.int64)
        # proposal-free view still supplies negatives: exactly the floor budget
        lm = rasterize_label_map(g, no_boxes, no_classes, quad, roi,
                                 np.random.default_rng(0), bg_floor=32)
        assert int((lm == 0).sum()) == 32
        assert set(np.unique(lm)) <= {0, 255}
        # floor above the candidate count -> every available bg cell is kept
        small_roi = np.array([0.0, 0.9, 0.2, 1.0])
        lm2 = rasterize_label_map(g, no_boxes, no_classes, quad, small_roi,
                                  np.random.default_rng(0), bg_floor=10_000)
        assert 0 < int((lm2 == 0).sum()) < 10_000

    def test_sampled_theta_is_well_formed(self):
        rng = np.random.default_rng(3)
        cfg = ViewAugConfig()
        for _ in range(20):
            th = sample_view_theta(rng, cfg)
            assert 2 * abs(th[0, 0]) <= 2.0 + 1e-9   # crop fits the unit square
            area = abs(th[0, 0]) * th[1, 1]          # (2hw)(2hh)/4... proportional
            assert cfg.scale[0] * 0.5 <= area        # not degenerate


# ── Signal A: loss minimal at the true alignment (toy backbone) ───────────────


def _unshuffle_stub() -> nn.Sequential:
    """Identity 1x1 conv (trainable probe) + PixelUnshuffle(8): each P3 cell's
    feature uniquely encodes its 8x8 source pixel block."""
    conv = nn.Conv2d(3, 3, 1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
    return nn.Sequential(conv, nn.PixelUnshuffle(8))


def _cell_centers(cells_x, cells_y, g: int = 8) -> np.ndarray:
    pts = [((cx + 0.5) / g, (cy + 0.5) / g) for cy in cells_y for cx in cells_x]
    return np.array(pts, dtype=np.float32)


def _corr_batch(shift_cells_y: int = 0):
    """A/B pair related by a +2-cell x-translation; pts at exact cell centers."""
    torch.manual_seed(0)
    img_a = torch.rand(3, 64, 64)
    img_b = torch.roll(img_a, shifts=16, dims=-1)        # content x_b = x_a + 16px
    pa = _cell_centers(range(1, 6), range(2, 7))         # 25 points, no wrap zone
    pb = pa.copy()
    pb[:, 0] += 2.0 / 8.0
    pb[:, 1] += shift_cells_y / 8.0
    k = pa.shape[0]
    return {
        "img": torch.stack([img_a, img_b]),
        "pts": torch.from_numpy(np.stack([pa, pb]))[None],   # [1, 2, K, 2]
        "valid": torch.ones(1, k, dtype=torch.bool),
        "labels": torch.full((2, 8, 8), 255, dtype=torch.long),
    }


class TestCorrMinimum:
    @pytest.fixture()
    def setup(self):
        stub = _unshuffle_stub()
        taps = MultiScaleFeatureTap(stub, levels=("P3",), layer_indices={"P3": 1})
        taps.setup()
        ch = PersistenceChannel(passthrough_heads=True, k_min=4)
        ch.attach(stub, taps)
        yield stub, taps, ch
        taps.close()

    def _loss(self, stub, taps, ch, batch):
        taps.clear()
        _ = stub(batch["img"])
        return ch.loss(batch, taps)

    def test_loss_zero_at_true_alignment(self, setup):
        stub, taps, ch = setup
        terms = self._loss(stub, taps, ch, _corr_batch())
        assert set(terms) == {"corr"}      # labels all-ignore -> pers skipped
        assert float(terms["corr"].detach()) < 1e-5

    def test_loss_strictly_higher_when_perturbed(self, setup):
        stub, taps, ch = setup
        true = float(self._loss(stub, taps, ch, _corr_batch())["corr"].detach())
        pert = float(self._loss(stub, taps, ch,
                                _corr_batch(shift_cells_y=2))["corr"].detach())
        assert pert > true + 0.05

    def test_stop_grad_target_branch(self):
        za = torch.randn(8, 16, requires_grad=True)
        zb = torch.randn(8, 16, requires_grad=True)
        loss = simsiam_one_way(za, zb)     # stop-grad is INSIDE the helper
        loss.backward()
        assert zb.grad is None             # target branch: zero grad flow
        assert za.grad is not None and za.grad.abs().sum() > 0

    def test_backbone_receives_grad(self, setup):
        stub, taps, ch = setup
        stub.zero_grad()
        terms = self._loss(stub, taps, ch, _corr_batch(shift_cells_y=2))
        terms["corr"].backward()
        assert stub[0].weight.grad is not None
        assert stub[0].weight.grad.abs().sum() > 0


# ── channel integration on a fake backbone ────────────────────────────────────


def _stride8_stub() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
    )


def _channel_batch(b_pairs: int = 2, k: int = 40, valid: bool = True):
    torch.manual_seed(1)
    g = 8
    labels = torch.full((2 * b_pairs, g, g), 255, dtype=torch.long)
    labels[:, 5:7, 2:5] = 1
    labels[:, 7, 5] = 2
    labels[:, 4, 0:3] = 0
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.1, 0.9, size=(b_pairs, 2, k, 2)).astype(np.float32)
    return {
        "img": torch.rand(2 * b_pairs, 3, 64, 64),
        "pts": torch.from_numpy(pts),
        "valid": torch.full((b_pairs, k), valid, dtype=torch.bool),
        "labels": labels,
    }


class TestChannelOnStub:
    @pytest.fixture()
    def setup(self):
        stub = _stride8_stub()
        taps = MultiScaleFeatureTap(stub, levels=("P3",), layer_indices={"P3": 5})
        taps.setup()
        ch = PersistenceChannel()
        heads = ch.attach(stub, taps)
        yield stub, taps, ch, heads
        taps.close()

    def test_attach_probes_width_and_returns_heads(self, setup):
        _, _, ch, heads = setup
        assert ch.p3_channels == 64
        assert isinstance(heads, nn.ModuleList) and len(heads) == 3
        head_param_ids = {id(p) for p in heads.parameters()}
        own = {id(p) for m in (ch.projector, ch.predictor, ch.pers_head)
               for p in m.parameters()}
        assert head_param_ids == own       # every trainable is handed over

    def test_attach_rejects_wrong_tap_layer(self):
        stub = _stride8_stub()
        taps = MultiScaleFeatureTap(stub, levels=("P3",), layer_indices={"P3": 3})
        taps.setup()                       # stride 4 layer — wrong by construction
        try:
            with pytest.raises(FeatureTapError, match="stride"):
                PersistenceChannel().attach(stub, taps)
        finally:
            taps.close()

    def test_loss_forward_backward_reaches_everything(self, setup):
        stub, taps, ch, heads = setup
        batch = _channel_batch()
        taps.clear()
        _ = stub(batch["img"])
        terms = ch.loss(batch, taps)
        assert set(terms) == {"corr", "pers"}
        total = sum(terms.values())
        assert torch.isfinite(total)
        total.backward()
        for name, module in (("backbone", stub[0]), ("projector", ch.projector),
                             ("predictor", ch.predictor), ("pers_head", ch.pers_head)):
            gsum = sum(p.grad.abs().sum() for p in module.parameters()
                       if p.grad is not None)
            assert gsum > 0, f"no gradient reached {name}"

    def test_ignore_cells_contribute_zero_gradient(self):
        torch.manual_seed(0)
        head = PersistenceHead(8)
        feats = torch.randn(1, 8, 4, 4)
        labels = torch.full((1, 4, 4), 255, dtype=torch.long)
        labels[0, 1, 1] = 1
        labels[0, 2, 3] = 0
        logits = head(feats)
        logits.retain_grad()
        F.cross_entropy(logits, labels, ignore_index=255).backward()
        g = logits.grad[0]                                  # [3, 4, 4]
        assert g[:, labels[0] == 255].abs().max() == 0
        assert g[:, labels[0] != 255].abs().max() > 0

    def test_signal_a_skip_path(self, setup):
        stub, taps, ch, _ = setup
        batch = _channel_batch(valid=False)                 # 0 valid points
        taps.clear()
        _ = stub(batch["img"])
        terms = ch.loss(batch, taps)
        assert set(terms) == {"pers"}                       # corr skipped, no error
        assert ch.sentinel_metrics()["skip_rate"] == 1.0

        batch["labels"][:] = 255                            # nothing left at all
        taps.clear()
        _ = stub(batch["img"])
        assert ch.loss(batch, taps) == {}

    def test_sentinel_metrics_populate(self, setup):
        stub, taps, ch, _ = setup
        ch.reset_epoch_stats()
        batch = _channel_batch()
        taps.clear()
        _ = stub(batch["img"])
        ch.loss(batch, taps)
        m = ch.sentinel_metrics()
        assert math.isfinite(m["proj_std"]) and m["proj_std"] > 0
        assert m["skip_rate"] == 0.0
        freq = sum(m[f"class_freq_{c}"] for c in ("background", "persistent",
                                                  "transient"))
        assert freq == pytest.approx(1.0)

    def test_on_epoch_end_reports_then_resets(self, setup):
        stub, taps, ch, _ = setup
        ch.reset_epoch_stats()
        batch = _channel_batch()
        taps.clear()
        _ = stub(batch["img"])
        ch.loss(batch, taps)
        m = ch.on_epoch_end(1)                 # R9 trainer hook
        assert math.isfinite(m["proj_std"]) and m["skip_rate"] == 0.0
        after = ch.sentinel_metrics()          # accumulators were reset
        assert math.isnan(after["skip_rate"])
        assert after["valid_points"] == 0.0

    def test_loss_before_attach_raises(self):
        with pytest.raises(RuntimeError, match="attach"):
            PersistenceChannel().loss({"img": torch.rand(2, 3, 64, 64)}, None)

    def test_odd_batch_raises(self, setup):
        stub, taps, ch, _ = setup
        taps.clear()
        _ = stub(torch.rand(3, 3, 64, 64))
        with pytest.raises(ValueError, match="2B"):
            ch.loss({"img": torch.rand(3, 3, 64, 64)}, taps)

    def test_state_dict_roundtrip_value_copy(self, setup):
        _, _, ch, _ = setup
        stub2 = _stride8_stub()
        taps2 = MultiScaleFeatureTap(stub2, levels=("P3",), layer_indices={"P3": 5})
        taps2.setup()
        try:
            ch2 = PersistenceChannel()
            ch2.attach(stub2, taps2)
            w_src = ch.projector.net[0].weight
            with torch.no_grad():           # make the two inits distinguishable
                w_src.add_(0.5)
            assert not torch.allclose(w_src, ch2.projector.net[0].weight)
            ch2.load_state_dict(ch.state_dict())
            w_dst = ch2.projector.net[0].weight
            assert torch.allclose(w_src, w_dst)
            assert w_src.data_ptr() != w_dst.data_ptr()     # Risk-16: no aliasing
        finally:
            taps2.close()


# ── dataset end-to-end (manifests + JPEGs on disk -> loader -> loss) ──────────


def _write_jpeg(path, seed: int, size: int = 128):
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, (size, size, 3)).astype(np.uint8)
    cv2.circle(arr, (size // 2, int(0.75 * size)), size // 10, (40, 40, 40), -1)
    Image.fromarray(arr).save(path, quality=95)


def _make_pool(tmp_path):
    """Two aligned pairs (translation H) + labels, all on disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    h = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rows, labels = [], []
    for i in range(2):
        pa, pb = img_dir / f"a{i}.jpg", img_dir / f"b{i}.jpg"
        _write_jpeg(pa, seed=10 + i)
        _write_jpeg(pb, seed=20 + i)
        rows.append(_pair_row(
            f"pr{i}", path_a=str(pa), path_b=str(pb), status="aligned",
            align_ok=True, n_persistent=1, n_transient=1, **h_to_row(h),
        ))
        labels.append({"label_id": f"pr{i}_a_000", "pair_id": f"pr{i}",
                       "image_id": f"pr{i}a", "side": "a",
                       "x1": 0.25, "y1": 0.5, "x2": 0.75, "y2": 0.9,
                       "label": "persistent"})
        labels.append({"label_id": f"pr{i}_a_001", "pair_id": f"pr{i}",
                       "image_id": f"pr{i}a", "side": "a",
                       "x1": 0.05, "y1": 0.45, "x2": 0.45, "y2": 0.8,
                       "label": "transient"})
        labels.append({"label_id": f"pr{i}_b_000", "pair_id": f"pr{i}",
                       "image_id": f"pr{i}b", "side": "b",
                       "x1": 0.35, "y1": 0.5, "x2": 0.85, "y2": 0.9,
                       "label": "persistent"})
    pairs_path = tmp_path / "pairs.parquet"
    labels_path = tmp_path / "labels.parquet"
    pm.append_pairs(pairs_path, rows)
    pm.append_labels(labels_path, labels)
    return pairs_path, labels_path


GENTLE_AUG = ViewAugConfig(scale=(0.9, 1.0), hflip_prob=0.0, photometric=0.1)


class TestDatasetEndToEnd:
    def test_item_shapes_and_label_semantics(self, tmp_path):
        pairs_path, labels_path = _make_pool(tmp_path)
        ds = PairDataset(pairs_path, labels=labels_path, imgsz=64,
                         aug=GENTLE_AUG, seed=0)
        assert len(ds) == 2
        it = ds[0]
        assert it["img_a"].shape == (3, 64, 64) and it["img_b"].shape == (3, 64, 64)
        assert it["img_a"].min() >= 0 and it["img_a"].max() <= 1
        assert it["pts"].shape == (2, 128, 2) and it["valid"].shape == (128,)
        assert int(it["valid"].sum()) >= 32                 # near-full crops
        assert it["labels_a"].shape == (8, 8)
        la = it["labels_a"].numpy()
        assert (la == 1).sum() > 0                          # persistent cells
        assert set(np.unique(la)) <= {0, 1, 2, 255}
        # determinism with a fixed seed
        it2 = ds[0]
        assert torch.equal(it["pts"], it2["pts"])

    def test_collate_and_channel_loss(self, tmp_path):
        pairs_path, labels_path = _make_pool(tmp_path)
        stub = _stride8_stub()
        taps = MultiScaleFeatureTap(stub, levels=("P3",), layer_indices={"P3": 5})
        taps.setup()
        try:
            ch = PersistenceChannel(pairs_path=str(pairs_path),
                                    labels_path=str(labels_path),
                                    aug=GENTLE_AUG, seed=0, k_min=8)
            ch.attach(stub, taps)
            loader = ch.build_loader(
                {"imgsz": 64, "batch": 2, "workers": 0, "device": "cpu"})
            batch = next(iter(loader))
            assert batch["img"].shape == (4, 3, 64, 64)     # 2 pairs -> 4 images
            assert batch["labels"].shape == (4, 8, 8)
            taps.clear()
            _ = stub(batch["img"])
            terms = ch.loss(batch, taps)
            assert "corr" in terms and "pers" in terms
            assert all(torch.isfinite(v) for v in terms.values())
        finally:
            taps.close()

    def test_dataset_requires_aligned_pairs(self, tmp_path):
        path = tmp_path / "pairs.parquet"
        pm.append_pairs(path, [_pair_row("p1")])            # queued, not aligned
        with pytest.raises(ValueError, match="align_ok"):
            PairDataset(path, imgsz=64)


# ── one real anchored-trainer step (yolov8n.yaml, offline) ────────────────────


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_persistence_channel_on_anchored_trainer(tmp_path):
    pytest.importorskip("ultralytics")
    from yolo_contrastive.anchored import AnchoredJointTrainer, SentinelThresholds

    relaxed = SentinelThresholds(
        eff_rank_warn=0.0, eff_rank_abort=-1.0,
        cls_drift_warn=1e9, cls_drift_abort=1e9,
        cka_warn=-1.0, head_norm_growth_warn=1e9,
    )
    ch = PersistenceChannel(k_min=4)
    trainer = AnchoredJointTrainer(
        model="yolov8n.yaml", channels=[ch], lambda_aux=1.0,
        epochs=1, imgsz=64, batch=2, warmup_steps=1, device="cpu", amp=False,
        output_dir=str(tmp_path), sentinel_thresholds=relaxed,
    )
    try:
        assert ch.p3_channels == trainer.tap_channels["P3"]     # attach probed v8n
        g = torch.Generator().manual_seed(0)
        n = 4
        replay = {
            "img": torch.rand(2, 3, 64, 64, generator=g),
            "batch_idx": torch.arange(2).repeat_interleave(2).float(),
            "cls": torch.randint(0, 80, (n, 1), generator=g).float(),
            "bboxes": torch.cat([0.3 + 0.4 * torch.rand(n, 2, generator=g),
                                 0.05 + 0.2 * torch.rand(n, 2, generator=g)], dim=1),
        }
        m = trainer.step(replay, {"persistence": _channel_batch(b_pairs=1, k=8)})
        assert "persistence/corr" in m and "persistence/pers" in m
        for k, v in m.items():
            assert math.isfinite(v), f"{k} not finite"
        assert m["total"] == pytest.approx(
            m["replay/det_loss"] + m["persistence/total"])
    finally:
        trainer.cleanup()
