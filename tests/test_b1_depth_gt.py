"""Tests for the B1 depth-GT store + Cityscapes disparity ingestion (geometry half).

CPU-only, offline, synthetic: a hand-built uint16 disparity PNG inside an in-memory zip
exercises the real ingestion path without any Cityscapes download.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from yolo_contrastive.geoteach.depth_gt import DepthGT, masked_downsample
from yolo_contrastive.data.ssl_pool.cityscapes_disparity import (
    _image_id,
    disparity_to_inverse,
    ingest_disparity,
)


class TestDepthGT:
    def test_save_load_roundtrip(self, tmp_path):
        gt = DepthGT(str(tmp_path), tag="t")
        inv = np.random.default_rng(0).random((16, 24)).astype(np.float32) + 0.1
        valid = np.zeros((16, 24), bool)
        valid[4:12, 6:18] = True
        gt.save("cityscapes/train/aachen/x_leftImg8bit", inv, valid, meta={"k": "v"})
        assert gt.has("cityscapes/train/aachen/x_leftImg8bit")
        inv2, valid2, meta = gt.load("cityscapes/train/aachen/x_leftImg8bit")
        assert inv2.shape == (16, 24) and valid2.shape == (16, 24)
        assert np.array_equal(valid2, valid)
        assert meta["k"] == "v"
        # invalid pixels are zeroed on save; valid pixels survive (float16 tolerance)
        assert np.allclose(inv2[valid], inv[valid], atol=1e-2)
        assert float(inv2[~valid].max()) == 0.0

    def test_shape_mismatch_raises(self, tmp_path):
        gt = DepthGT(str(tmp_path))
        with pytest.raises(ValueError):
            gt.save("id", np.zeros((4, 4), np.float32), np.zeros((4, 5), bool))

    def test_image_ids_listing(self, tmp_path):
        gt = DepthGT(str(tmp_path), tag="t")
        for name in ("cityscapes/train/a/x_leftImg8bit", "cityscapes/val/b/y_leftImg8bit"):
            gt.save(name, np.ones((4, 4), np.float32), np.ones((4, 4), bool))
        assert len(gt) == 2
        assert "cityscapes/train/a/x_leftImg8bit" in gt.image_ids()


class TestMaskedDownsample:
    def test_no_hole_bleed(self):
        # left half valid (inv=2.0), right half a hole (invalid, inv=0 sentinel)
        inv = np.zeros((32, 32), np.float32)
        valid = np.zeros((32, 32), bool)
        inv[:, :16] = 2.0
        valid[:, :16] = True
        inv_ds, valid_ds = masked_downsample(inv, valid, (8, 8))
        # the fully-valid left columns must average to 2.0 (no sentinel bleed)
        left = valid_ds[:, :3]
        assert left.all()
        assert np.allclose(inv_ds[:, :3][left], 2.0, atol=1e-3)
        # the fully-hole right columns stay invalid
        assert not valid_ds[:, -3:].any()


class TestDisparityDecode:
    def test_sentinel_and_scale(self):
        png = np.array([[0, 257, 513]], np.uint16)   # p=0 invalid; p=257 -> disp 1.0; p=513 -> 2.0
        inv, valid = disparity_to_inverse(png)
        assert not valid[0, 0] and inv[0, 0] == 0.0
        assert valid[0, 1] and abs(inv[0, 1] - 1.0) < 1e-5
        assert abs(inv[0, 2] - 2.0) < 1e-5

    def test_image_id_mapping(self):
        assert _image_id("train", "aachen", "aachen_000000_000019_disparity") == \
            "cityscapes/train/aachen/aachen_000000_000019_leftImg8bit"


class TestIngestDisparity:
    def _zip_with_disparity(self, tmp_path):
        import cv2
        # a 64x128 uint16 disparity: valid blob in the middle, sentinel elsewhere
        d = np.zeros((64, 128), np.uint16)
        d[20:44, 40:90] = 600  # disp = (600-1)/256 ≈ 2.34
        ok, buf = cv2.imencode(".png", d)
        assert ok
        zpath = tmp_path / "disp.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("disparity/train/aachen/aachen_000000_000019_disparity.png", buf.tobytes())
            z.writestr("README", b"skip me")
        return zpath

    def test_ingest_writes_gt(self, tmp_path):
        zpath = self._zip_with_disparity(tmp_path)
        gt = DepthGT(str(tmp_path / "gtstore"), tag="cityscapes_disp")
        stats = ingest_disparity(zpath, gt, long_side=64, log_every=0)
        assert stats["materialized"] == 1 and stats["scanned"] == 1
        ids = gt.image_ids()
        assert ids == ["cityscapes/train/aachen/aachen_000000_000019_leftImg8bit"]
        inv, valid, meta = gt.load(ids[0])
        assert valid.any() and meta["source"] == "cityscapes_disparity"
        # downsampled to long-side 64 (from 128 wide) → 32x64
        assert inv.shape == (32, 64)

    def test_ingest_resume_skips(self, tmp_path):
        zpath = self._zip_with_disparity(tmp_path)
        gt = DepthGT(str(tmp_path / "gtstore"), tag="cityscapes_disp")
        ingest_disparity(zpath, gt, long_side=64, log_every=0)
        stats = ingest_disparity(zpath, gt, long_side=64, log_every=0)
        assert stats["skipped_existing"] == 1 and stats["materialized"] == 0
