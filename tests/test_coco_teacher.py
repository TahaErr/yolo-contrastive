"""Tests for CocoTeacher — frozen YOLOv8x feature teacher + per-scale adapter.

Mock tests use a synthetic 23-layer YOLOv8-like encoder (layer indices
15/18/21 valid for MultiScaleFeatureTap) with teacher-scale channel widths.
One slow test exercises a real COCO YOLOv8x download.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# Teacher-scale channels (wider than student — mimics YOLOv8x vs YOLOv8n).
_TEACHER_CH = (256, 512, 512)   # P3, P4, P5
# Student-scale channels (YOLOv8n-like).
_STUDENT_CH = {"P3": 64, "P4": 128, "P5": 256}


def _mock_yolo_encoder(channels=_TEACHER_CH) -> nn.Sequential:
    """23-layer Sequential mimicking YOLOv8 FPN — P3/P4/P5 at layers 15/18/21.

    Downsamples at layers 0, 6, 12, 16, 19; channel shifts placed so the
    hooks at 15/18/21 capture (p3, p4, p5) channel counts.
    """
    p3, p4, p5 = channels
    layers = []
    layers.append(nn.Conv2d(3, p3, 3, stride=2, padding=1))    # 0  /2
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))         # 1-5
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))   # 6  /4
    for _ in range(5):
        layers.append(nn.Conv2d(p3, p3, 3, padding=1))         # 7-11
    layers.append(nn.Conv2d(p3, p3, 3, stride=2, padding=1))   # 12 /8
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 13
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 14
    layers.append(nn.Conv2d(p3, p3, 3, padding=1))             # 15 P3 out
    layers.append(nn.Conv2d(p3, p4, 3, stride=2, padding=1))   # 16 /16, p4
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))             # 17
    layers.append(nn.Conv2d(p4, p4, 3, padding=1))             # 18 P4 out
    layers.append(nn.Conv2d(p4, p5, 3, stride=2, padding=1))   # 19 /32, p5
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))             # 20
    layers.append(nn.Conv2d(p5, p5, 3, padding=1))             # 21 P5 out
    layers.append(nn.Conv2d(p5, p5, 1))                        # 22 dummy Detect
    return nn.Sequential(*layers)


# ═════════════════════════════════════════════════════════════════════════
# Construction
# ═════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_builds_from_module_no_adapter(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            assert teacher.adapter is None
            assert teacher.levels == ("P3", "P4", "P5")
        finally:
            teacher.cleanup()

    def test_builds_with_adapter(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), student_channels=_STUDENT_CH,
            device="cpu",
        )
        try:
            assert teacher.adapter is not None
            assert set(teacher.adapter.keys()) == {"P3", "P4", "P5"}
        finally:
            teacher.cleanup()

    def test_teacher_channels_probed(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            assert teacher.teacher_channels == {"P3": 256, "P4": 512, "P5": 512}
        finally:
            teacher.cleanup()

    def test_student_channels_missing_level_raises(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        with pytest.raises(ValueError, match="missing levels"):
            CocoTeacher(
                weights=_mock_yolo_encoder(),
                student_channels={"P3": 64, "P4": 128},  # P5 missing
                device="cpu",
            )

    def test_subset_levels_p5_only(self):
        """P5-only cache strategy (§2.4 alternative)."""
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), levels=("P5",), device="cpu",
        )
        try:
            assert teacher.levels == ("P5",)
            assert set(teacher.teacher_channels.keys()) == {"P5"}
        finally:
            teacher.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Freeze / trainability invariants
# ═════════════════════════════════════════════════════════════════════════


class TestFreezeInvariants:
    def test_teacher_backbone_frozen(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            for p in teacher.backbone.parameters():
                assert p.requires_grad is False
        finally:
            teacher.cleanup()

    def test_adapter_is_trainable(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), student_channels=_STUDENT_CH,
            device="cpu",
        )
        try:
            for p in teacher.adapter.parameters():
                assert p.requires_grad is True
        finally:
            teacher.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# extract_features — cache build path
# ═════════════════════════════════════════════════════════════════════════


class TestExtractFeatures:
    def test_shape_and_keys(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            feats = teacher.extract_features(torch.rand(2, 3, 64, 64))
            assert set(feats.keys()) == {"P3", "P4", "P5"}
            # teacher channels, 4D maps
            assert feats["P3"].shape[:2] == (2, 256)
            assert feats["P4"].shape[:2] == (2, 512)
            assert feats["P5"].shape[:2] == (2, 512)
        finally:
            teacher.cleanup()

    def test_output_detached(self):
        """Cache-bound features must carry no gradient."""
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            feats = teacher.extract_features(torch.rand(2, 3, 64, 64))
            for t in feats.values():
                assert t.requires_grad is False
                assert t.grad_fn is None
        finally:
            teacher.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# adapt — train-time path
# ═════════════════════════════════════════════════════════════════════════


class TestAdapt:
    def test_maps_teacher_to_student_channels(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), student_channels=_STUDENT_CH,
            device="cpu",
        )
        try:
            raw = teacher.extract_features(torch.rand(2, 3, 64, 64))
            adapted = teacher.adapt(raw)
            assert adapted["P3"].shape[1] == 64
            assert adapted["P4"].shape[1] == 128
            assert adapted["P5"].shape[1] == 256
            # Spatial preserved
            for lv in ("P3", "P4", "P5"):
                assert adapted[lv].shape[2:] == raw[lv].shape[2:]
        finally:
            teacher.cleanup()

    def test_gradient_flows_through_adapter(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), student_channels=_STUDENT_CH,
            device="cpu",
        )
        try:
            raw = teacher.extract_features(torch.rand(2, 3, 64, 64))
            adapted = teacher.adapt(raw)
            loss = sum(t.mean() for t in adapted.values())
            loss.backward()
            for p in teacher.adapter.parameters():
                assert p.grad is not None
                assert p.grad.abs().sum() > 0
        finally:
            teacher.cleanup()

    def test_adapt_without_adapter_raises(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            raw = teacher.extract_features(torch.rand(1, 3, 64, 64))
            with pytest.raises(ValueError, match="student_channels"):
                teacher.adapt(raw)
        finally:
            teacher.cleanup()

    def test_adapt_wrong_channel_raises(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), student_channels=_STUDENT_CH,
            device="cpu",
        )
        try:
            bad = {
                "P3": torch.rand(1, 99, 8, 8),   # wrong channel count
                "P4": torch.rand(1, 512, 4, 4),
                "P5": torch.rand(1, 512, 2, 2),
            }
            with pytest.raises(ValueError, match="channel mismatch"):
                teacher.adapt(bad)
        finally:
            teacher.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# forward — extract + adapt convenience
# ═════════════════════════════════════════════════════════════════════════


class TestForward:
    def test_forward_equals_extract_then_adapt(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(
            weights=_mock_yolo_encoder(), student_channels=_STUDENT_CH,
            device="cpu",
        )
        try:
            out = teacher.forward(torch.rand(2, 3, 64, 64))
            assert set(out.keys()) == {"P3", "P4", "P5"}
            assert out["P3"].shape[1] == 64
            assert out["P5"].shape[1] == 256
        finally:
            teacher.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_cleanup_idempotent(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        teacher.cleanup()
        teacher.cleanup()   # no raise
        assert teacher.tap._is_setup is False

    def test_repr(self):
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights=_mock_yolo_encoder(), device="cpu")
        try:
            r = repr(teacher)
            assert "CocoTeacher" in r
            assert "teacher_channels" in r
        finally:
            teacher.cleanup()


# ═════════════════════════════════════════════════════════════════════════
# Real COCO YOLOv8x
# ═════════════════════════════════════════════════════════════════════════


class TestRealYOLOv8x:
    @pytest.mark.slow
    def test_real_yolov8x_extract(self):
        """Download real COCO YOLOv8x, extract P3/P4/P5 features."""
        from yolo_contrastive.dual_teacher import CocoTeacher

        teacher = CocoTeacher(weights="yolov8x.pt", device="cpu")
        try:
            feats = teacher.extract_features(torch.rand(1, 3, 320, 320))
            assert set(feats.keys()) == {"P3", "P4", "P5"}
            # YOLOv8x is wide — every level should have a healthy channel count
            for lv in ("P3", "P4", "P5"):
                assert feats[lv].dim() == 4
                assert feats[lv].shape[1] >= 128
            # FPN stride sanity: P3 > P4 > P5 spatial
            assert feats["P3"].shape[2] > feats["P4"].shape[2] > feats["P5"].shape[2]
        finally:
            teacher.cleanup()
