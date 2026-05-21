"""Tests for TeacherCache — FP16 npz feature cache I/O.

All tests are fast: a tiny stub teacher returns small random feature maps,
no real model needed.
"""

from __future__ import annotations

import json

import pytest
import torch


_LEVELS = ("P3", "P4", "P5")


def _fake_features(batch=1):
    """Teacher-like multi-scale feature maps (tiny, for fast tests)."""
    return {
        "P3": torch.randn(batch, 16, 8, 8),
        "P4": torch.randn(batch, 32, 4, 4),
        "P5": torch.randn(batch, 64, 2, 2),
    }


class _StubTeacher:
    """Minimal teacher — extract_features returns fixed-shape random maps."""

    def __init__(self):
        self.calls = 0

    def extract_features(self, images):
        self.calls += 1
        b = images.shape[0]
        return _fake_features(batch=b)


# ═════════════════════════════════════════════════════════════════════════
# save / load roundtrip
# ═════════════════════════════════════════════════════════════════════════


class TestSaveLoad:
    def test_roundtrip_keys_and_shapes(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        feats = {lv: t[0] for lv, t in _fake_features().items()}
        cache.save("img_001", feats)
        loaded = cache.load("img_001")

        assert set(loaded.keys()) == set(_LEVELS)
        for lv in _LEVELS:
            assert loaded[lv].shape == feats[lv].shape

    def test_roundtrip_fp16_precision(self, tmp_path):
        """FP16 storage → FP32 load: values close within half-precision."""
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        feats = {lv: t[0] for lv, t in _fake_features().items()}
        cache.save("img_x", feats)
        loaded = cache.load("img_x")
        for lv in _LEVELS:
            # FP16 has ~3 decimal digits — loose tolerance
            assert torch.allclose(loaded[lv], feats[lv], atol=1e-2, rtol=1e-2)

    def test_loaded_tensors_are_fp32(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        cache.save("img_y", {lv: t[0] for lv, t in _fake_features().items()})
        loaded = cache.load("img_y")
        for t in loaded.values():
            assert t.dtype == torch.float32

    def test_save_missing_level_raises(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        with pytest.raises(ValueError, match="missing levels"):
            cache.save("img_z", {"P3": torch.randn(16, 8, 8)})  # P4/P5 missing

    def test_load_missing_raises(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Not cached"):
            cache.load("never_saved")


# ═════════════════════════════════════════════════════════════════════════
# has / __contains__ / __len__
# ═════════════════════════════════════════════════════════════════════════


class TestPresence:
    def test_has_false_then_true(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        assert cache.has("img_a") is False
        cache.save("img_a", {lv: t[0] for lv, t in _fake_features().items()})
        assert cache.has("img_a") is True
        assert "img_a" in cache

    def test_len_counts_cached(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        assert len(cache) == 0
        for i in range(3):
            cache.save(f"img_{i}", {lv: t[0] for lv, t in _fake_features().items()})
        assert len(cache) == 3


# ═════════════════════════════════════════════════════════════════════════
# nested image_id (slashes → sub-dirs)
# ═════════════════════════════════════════════════════════════════════════


class TestNestedImageId:
    def test_slash_image_id_creates_subdirs(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        cache.save(
            "bdd100k/train/abc123",
            {lv: t[0] for lv, t in _fake_features().items()},
        )
        assert cache.has("bdd100k/train/abc123")
        loaded = cache.load("bdd100k/train/abc123")
        assert set(loaded.keys()) == set(_LEVELS)

    def test_no_collision_between_datasets(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        f1 = {lv: t[0] for lv, t in _fake_features().items()}
        f2 = {lv: t[0] for lv, t in _fake_features().items()}
        cache.save("a2d2/img_1", f1)
        cache.save("bdd100k/img_1", f2)
        # Same basename, different dataset — must not overwrite
        l1 = cache.load("a2d2/img_1")
        assert torch.allclose(l1["P3"], f1["P3"], atol=1e-2)
        assert len(cache) == 2


# ═════════════════════════════════════════════════════════════════════════
# metadata
# ═════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_save_and_load_metadata(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        cache.save_metadata(extra={"teacher_channels": {"P3": 256}})
        meta = cache.load_metadata()
        assert meta["teacher_tag"] == "yolov8x_coco_p3p4p5"
        assert meta["levels"] == ["P3", "P4", "P5"]
        assert meta["cache_version"] == 1
        assert meta["teacher_channels"] == {"P3": 256}

    def test_load_metadata_missing_raises(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No metadata"):
            cache.load_metadata()


# ═════════════════════════════════════════════════════════════════════════
# build — bulk + idempotency
# ═════════════════════════════════════════════════════════════════════════


class TestBuild:
    def test_build_populates_cache(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        teacher = _StubTeacher()
        images = [(f"img_{i}", torch.rand(3, 16, 16)) for i in range(5)]

        stats = cache.build(teacher, images, log_every=0)
        assert stats == {"scanned": 5, "skipped": 0, "computed": 5, "errors": 0}
        assert len(cache) == 5

    def test_build_idempotent_skips_existing(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        teacher = _StubTeacher()
        images = [(f"img_{i}", torch.rand(3, 16, 16)) for i in range(4)]

        cache.build(teacher, images, log_every=0)
        # Second run — everything already cached
        stats2 = cache.build(teacher, list(images), log_every=0)
        assert stats2["skipped"] == 4
        assert stats2["computed"] == 0

    def test_build_writes_metadata(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        teacher = _StubTeacher()
        images = [("img_0", torch.rand(3, 16, 16))]
        cache.build(teacher, images, log_every=0)
        meta = cache.load_metadata()
        assert meta["n_cached"] == 1

    def test_build_cached_features_loadable(self, tmp_path):
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path))
        teacher = _StubTeacher()
        cache.build(teacher, [("img_q", torch.rand(3, 16, 16))], log_every=0)
        loaded = cache.load("img_q")
        # Stub teacher feature shapes (batch dim dropped)
        assert loaded["P3"].shape == (16, 8, 8)
        assert loaded["P5"].shape == (64, 2, 2)


# ═════════════════════════════════════════════════════════════════════════
# subset levels
# ═════════════════════════════════════════════════════════════════════════


class TestSubsetLevels:
    def test_p5_only_cache(self, tmp_path):
        """P5-only cache strategy (§2.4 alternative)."""
        from yolo_contrastive.dual_teacher.teacher_cache import TeacherCache

        cache = TeacherCache(str(tmp_path), levels=("P5",))
        cache.save("img_p5", {"P5": torch.randn(64, 2, 2)})
        loaded = cache.load("img_p5")
        assert set(loaded.keys()) == {"P5"}
