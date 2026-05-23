"""Tests for MultiScaleFeatureTap.

Synthetic tests cover hooks, lifecycle, errors, model drilling.
A separate test attempts real YOLOv8 if ultralytics is installed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from yolo_contrastive.dense import (
    MultiScaleFeatureTap,
    YOLOV8_FPN_LAYERS,
    YOLOV8_FPN_STRIDES,
)
from yolo_contrastive.exceptions import FeatureTapError


# ── helpers ──────────────────────────────────────────────────────────────


def _dummy_seq(n_layers: int = 23) -> nn.Sequential:
    """Sequential of N pointwise convs — preserves shape, hooks attach cleanly."""
    return nn.Sequential(*[nn.Conv2d(3, 3, kernel_size=1) for _ in range(n_layers)])


# ── setup / hooks ─────────────────────────────────────────────────────────


class TestSetupAndHooks:
    def test_setup_installs_three_hooks(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        tap.setup()
        assert len(tap._hooks) == 3
        tap.close()

    def test_close_removes_hooks(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        tap.setup()
        tap.close()
        assert len(tap._hooks) == 0
        assert not tap._is_setup

    def test_setup_idempotent(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        tap.setup()
        tap.setup()
        assert len(tap._hooks) == 3  # not 6
        tap.close()

    def test_features_populated_after_forward(self):
        seq = _dummy_seq()
        tap = MultiScaleFeatureTap(seq)
        tap.setup()
        _ = seq(torch.randn(2, 3, 32, 32))
        feats = tap.get_features()
        assert set(feats.keys()) == {"P3", "P4", "P5"}
        for t in feats.values():
            assert t is not None
            assert t.shape[0] == 2

    def test_features_none_before_forward(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        tap.setup()
        with pytest.raises(FeatureTapError, match="None"):
            tap.get_features()
        tap.close()

    def test_get_features_before_setup_raises(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        with pytest.raises(FeatureTapError, match="setup"):
            tap.get_features()


# ── context manager ──────────────────────────────────────────────────────


class TestContextManager:
    def test_context_installs_and_closes(self):
        seq = _dummy_seq()
        with MultiScaleFeatureTap(seq) as tap:
            tap.setup()
            assert tap._is_setup
            _ = seq(torch.randn(1, 3, 32, 32))
            feats = tap.get_features()
            assert all(v is not None for v in feats.values())
        assert not tap._is_setup
        assert len(tap._hooks) == 0


# ── clear ─────────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_resets_features_keeps_hooks(self):
        seq = _dummy_seq()
        tap = MultiScaleFeatureTap(seq)
        tap.setup()
        _ = seq(torch.randn(1, 3, 32, 32))
        tap.clear()
        # Hooks still installed
        assert len(tap._hooks) == 3
        with pytest.raises(FeatureTapError, match="None"):
            tap.get_features()
        # Re-forward repopulates
        _ = seq(torch.randn(1, 3, 32, 32))
        feats = tap.get_features()
        assert all(v is not None for v in feats.values())
        tap.close()


# ── error paths ───────────────────────────────────────────────────────────


class TestErrors:
    def test_layer_out_of_range_raises_and_rollbacks(self):
        seq = _dummy_seq(n_layers=10)
        tap = MultiScaleFeatureTap(seq)
        with pytest.raises(FeatureTapError, match="out of range"):
            tap.setup()
        # Rollback: no dangling hooks
        assert len(tap._hooks) == 0
        assert not tap._is_setup

    def test_unknown_level_without_indices_raises(self):
        with pytest.raises(FeatureTapError, match="No layer index"):
            MultiScaleFeatureTap(_dummy_seq(), levels=("P3", "P4", "P6"))

    def test_unwrappable_model_raises(self):
        class NotASequential(nn.Module):
            pass

        with pytest.raises(FeatureTapError, match="locate nn.Sequential"):
            MultiScaleFeatureTap(NotASequential()).setup()

    def test_negative_index_raises(self):
        tap = MultiScaleFeatureTap(
            _dummy_seq(),
            levels=("L1",),
            layer_indices={"L1": -1},
        )
        with pytest.raises(FeatureTapError, match="out of range"):
            tap.setup()


# ── customization ─────────────────────────────────────────────────────────


class TestCustomConfig:
    def test_custom_levels_and_indices(self):
        seq = _dummy_seq()
        tap = MultiScaleFeatureTap(
            seq,
            levels=("L1", "L2"),
            layer_indices={"L1": 5, "L2": 10},
        )
        tap.setup()
        _ = seq(torch.randn(1, 3, 32, 32))
        feats = tap.get_features()
        assert set(feats.keys()) == {"L1", "L2"}
        tap.close()

    def test_subset_of_default_levels(self):
        # Just P3 and P5 — skip P4
        seq = _dummy_seq()
        tap = MultiScaleFeatureTap(seq, levels=("P3", "P5"))
        tap.setup()
        _ = seq(torch.randn(1, 3, 32, 32))
        feats = tap.get_features()
        assert set(feats.keys()) == {"P3", "P5"}
        tap.close()


# ── model drilling ────────────────────────────────────────────────────────


class TestModelDrilling:
    def test_direct_sequential(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        tap.setup()
        tap.close()

    def test_wrapped_in_model_attr(self):
        class Wrapper(nn.Module):
            def __init__(self, seq):
                super().__init__()
                self.model = seq

        w = Wrapper(_dummy_seq())
        tap = MultiScaleFeatureTap(w)
        tap.setup()
        _ = w.model(torch.randn(1, 3, 32, 32))
        feats = tap.get_features()
        assert all(v is not None for v in feats.values())
        tap.close()

    def test_nested_model_dot_model(self):
        # Mimics ultralytics: YOLO.model is DetectionModel, DetectionModel.model is Sequential
        class DetectionModel(nn.Module):
            def __init__(self, seq):
                super().__init__()
                self.model = seq

        class YOLO(nn.Module):
            def __init__(self, seq):
                super().__init__()
                self.model = DetectionModel(seq)

        y = YOLO(_dummy_seq())
        tap = MultiScaleFeatureTap(y)
        tap.setup()
        _ = y.model.model(torch.randn(1, 3, 32, 32))
        feats = tap.get_features()
        assert all(v is not None for v in feats.values())
        tap.close()


# ── repr ──────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_pre_setup(self):
        r = repr(MultiScaleFeatureTap(_dummy_seq()))
        assert "MultiScaleFeatureTap" in r
        assert "not setup" in r

    def test_repr_post_setup(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        tap.setup()
        r = repr(tap)
        assert "status=setup" in r
        tap.close()


# ── constants ─────────────────────────────────────────────────────────────


def test_yolov8_fpn_layers_constants():
    """Verify hardcoded indices match YOLOv8 yaml head section."""
    assert YOLOV8_FPN_LAYERS == {"P3": 15, "P4": 18, "P5": 21}
    assert YOLOV8_FPN_STRIDES == {"P3": 8, "P4": 16, "P5": 32}


# ── tuple output handling ─────────────────────────────────────────────────


class TestTupleOutput:
    """Hook should unwrap tuple/list outputs (defensive — rare in YOLOv8 neck)."""

    def test_hook_handles_tuple(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        hook = tap._make_hook("P3")
        fake = torch.randn(1, 3, 8, 8)
        hook(None, None, (fake, torch.zeros_like(fake)))
        assert tap._features["P3"] is fake

    def test_hook_handles_list(self):
        tap = MultiScaleFeatureTap(_dummy_seq())
        hook = tap._make_hook("P3")
        fake = torch.randn(1, 3, 8, 8)
        hook(None, None, [fake])
        assert tap._features["P3"] is fake


# ── real YOLOv8 (skip if ultralytics missing) ─────────────────────────────


def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _ultralytics_available(), reason="ultralytics not installed")
class TestRealYOLOv8:
    def test_strides_match_expected(self):
        from ultralytics import YOLO

        y = YOLO("yolov8n.pt")
        det_model = y.model
        det_model.eval()

        tap = MultiScaleFeatureTap(det_model)
        tap.setup()

        imgsz = 640
        with torch.no_grad():
            _ = det_model(torch.randn(1, 3, imgsz, imgsz))
        feats = tap.get_features()
        tap.close()

        for level, expected_stride in YOLOV8_FPN_STRIDES.items():
            assert level in feats
            t = feats[level]
            assert t is not None
            actual_stride = imgsz // t.shape[-1]
            assert actual_stride == expected_stride, (
                f"{level}: expected stride {expected_stride}, got {actual_stride} "
                f"(shape={tuple(t.shape)})"
            )


# ── architecture-agnostic Detect.f layer detection (v9 — smoke bug fix) ──


class TestDetectFpnLayers:
    """detect_fpn_layers reads P3/P4/P5 indices from the Detect head, so the
    tap works across YOLOv8/9/10/11/12/26 — not just the hardcoded v8 table.

    These tests need ultralytics + real weights; skipped if unavailable.
    """

    # Detect.f indices verified against real models (Faz 5.2 smoke).
    EXPECTED = {
        "yolov8n.pt": [15, 18, 21],
        "yolov9t.pt": [15, 18, 21],
        "yolov10n.pt": [16, 19, 22],
        "yolo11n.pt": [16, 19, 22],
        "yolo12n.pt": [14, 17, 20],
        "yolo26n.pt": [16, 19, 22],
    }

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name,expected", list(EXPECTED.items()))
    def test_detect_fpn_layers_matches_architecture(self, model_name, expected):
        ultralytics = pytest.importorskip("ultralytics")
        from yolo_contrastive.dense import detect_fpn_layers

        model = ultralytics.YOLO(model_name).model
        idx = detect_fpn_layers(model, ("P3", "P4", "P5"))
        assert idx is not None, f"{model_name}: Detect head not found"
        assert [idx["P3"], idx["P4"], idx["P5"]] == expected

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", list(EXPECTED.keys()))
    def test_tap_extracts_correct_strides(self, model_name):
        """Tapped P3/P4/P5 are 4D feature maps at stride 8/16/32 — not the
        Detect head output (the v12 failure mode from Faz 5.2 smoke)."""
        ultralytics = pytest.importorskip("ultralytics")
        model = ultralytics.YOLO(model_name).model.eval()

        tap = MultiScaleFeatureTap(model)  # layer_indices=None -> auto-detect
        tap.setup()
        with torch.no_grad():
            model(torch.rand(2, 3, 320, 320))
        feats = tap.get_features()
        tap.close()

        for level, stride in [("P3", 8), ("P4", 16), ("P5", 32)]:
            t = feats[level]
            assert torch.is_tensor(t) and t.dim() == 4, (
                f"{model_name} {level}: not a 4D feature map "
                f"(got {type(t).__name__}) — tap hit the wrong layer"
            )
            assert 320 // t.shape[2] == stride, (
                f"{model_name} {level}: stride {320 // t.shape[2]} != {stride}"
            )

    def test_falls_back_to_v8_table_for_bare_sequential(self):
        """A bare nn.Sequential has no Detect head — detect_fpn_layers
        returns None and the tap falls back to YOLOV8_FPN_LAYERS."""
        from yolo_contrastive.dense import detect_fpn_layers

        seq = _dummy_seq()
        assert detect_fpn_layers(seq, ("P3", "P4", "P5")) is None
        tap = MultiScaleFeatureTap(seq)
        assert tap.layer_indices == YOLOV8_FPN_LAYERS
