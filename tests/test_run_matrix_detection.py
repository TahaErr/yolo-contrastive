"""Tests for eval/run_matrix.py::_run_detection runner (§13.7 implementation).

These tests are mock-based — we monkey-patch the Ultralytics import boundary
and the FinetuneDetectionTrainer import so we never actually load YOLO or
touch GPU. The contract under test is:

    (a) env var lifecycle:    set before train, restored after (even on error)
    (b) hp parameter forward: cell + hp → YOLO.train kwargs are correct
    (c) return shape:         results dict has metric, metric_value, mAP50,
                              precision, recall keys with float values
    (d) error propagation:    YOLO.train exception bubbles up to caller
    (e) defaults applied:     missing hp keys → paper-grade defaults
    (f) cell_id used:         run_name reflects cell_id when present
    (g) missing required:     backbone_ckpt/data_yaml empty → ValueError

Smoke-test against the real Ultralytics integration is covered by §11.8
(v2 production validation), not these unit tests.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Helpers — install mock modules so the runner's lazy imports succeed
# ─────────────────────────────────────────────────────────────────────────


class _MockBox:
    """Mimics ultralytics results.box with the four metrics _run_detection reads."""

    def __init__(self, map_=0.42, map50=0.78, mp=0.65, mr=0.55):
        # cast-to-float compatibility: use plain numbers (real Ultralytics
        # returns torch tensors; float(tensor) works the same)
        self.map = map_
        self.map50 = map50
        self.mp = mp
        self.mr = mr


class _MockResults:
    def __init__(self, box=None):
        self.box = box or _MockBox()


class _MockYOLO:
    """Stand-in for ultralytics.YOLO. Records constructor + train args."""

    last_init = None
    last_train_kwargs = None
    train_raises = None
    train_result = None
    train_call_count = 0

    @classmethod
    def reset(cls):
        cls.last_init = None
        cls.last_train_kwargs = None
        cls.train_raises = None
        cls.train_result = None
        cls.train_call_count = 0

    def __init__(self, model_spec):
        _MockYOLO.last_init = model_spec

    def train(self, **kwargs):
        _MockYOLO.train_call_count += 1
        _MockYOLO.last_train_kwargs = kwargs
        if _MockYOLO.train_raises is not None:
            raise _MockYOLO.train_raises
        return _MockYOLO.train_result or _MockResults()


@pytest.fixture(autouse=True)
def install_mock_ultralytics(monkeypatch):
    """Inject mock ultralytics + mock yolo_contrastive.finetune modules.

    The runner's body does:
        from ultralytics import YOLO              # → mock here
        from ..finetune import FinetuneDetectionTrainer  # → mock here too,
                                                    so we don't trigger the real
                                                    finetune __init__ chain
                                                    (which loads ultralytics
                                                    submodules our top-level
                                                    mock doesn't cover).
    """
    _MockYOLO.reset()

    # 1) Mock the top-level ultralytics module
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = _MockYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    # 2) Mock yolo_contrastive.finetune — runner imports
    #    FinetuneDetectionTrainer from here; we provide a stand-in so the
    #    real submodule chain (which imports ultralytics submodules) is never
    #    triggered.
    fake_finetune = types.ModuleType("yolo_contrastive.finetune")

    class _FakeTrainer:
        """Stand-in passed as `trainer=` kwarg; never actually instantiated."""
        pass

    fake_finetune.FinetuneDetectionTrainer = _FakeTrainer
    monkeypatch.setitem(sys.modules, "yolo_contrastive.finetune", fake_finetune)

    yield

    # Cleanup via monkeypatch fixture teardown


def _basic_cell(**overrides) -> Dict[str, Any]:
    """Build a minimal valid cell dict."""
    cell = {
        "method": {"name": "test_method", "backbone_ckpt": "/fake/backbone.pt"},
        "dataset": {"name": "test_ds", "data_yaml": "/fake/data.yaml"},
        "fraction": 1.0,
        "seed": 42,
        "task": "detection",
        "cell_id": "abc12345xyz0",
    }
    for k, v in overrides.items():
        cell[k] = v
    return cell


def _basic_hp(**overrides) -> Dict[str, Any]:
    """Default test hp dict (intentionally smaller than paper defaults for clarity)."""
    hp = {
        "base_model": "yolov8n.pt",
        "epochs": 5,
        "imgsz": 320,
        "batch": 8,
        "freeze": 10,
        "unfreeze_epoch": 2,
        "backbone_lr_scale": 0.5,
        "device": "cpu",
        "project": "/tmp/test_runs",
    }
    for k, v in overrides.items():
        hp[k] = v
    return hp


# ─────────────────────────────────────────────────────────────────────────
# (a) Env var lifecycle
# ─────────────────────────────────────────────────────────────────────────


class TestEnvVarLifecycle:
    def test_env_vars_set_during_train(self, monkeypatch):
        # Clear any existing env state
        for k in ("YCL_PRETRAINED", "YCL_FREEZE_BACKBONE",
                  "YCL_UNFREEZE_EPOCH", "YCL_BACKBONE_LR_SCALE"):
            monkeypatch.delenv(k, raising=False)

        # Inspect env *inside* train via mock
        captured_env = {}

        def capture_env_train(**kwargs):
            for k in ("YCL_PRETRAINED", "YCL_FREEZE_BACKBONE",
                      "YCL_UNFREEZE_EPOCH", "YCL_BACKBONE_LR_SCALE"):
                captured_env[k] = os.environ.get(k)
            return _MockResults()

        original_train = _MockYOLO.train
        _MockYOLO.train = lambda self, **kw: capture_env_train(**kw)

        try:
            from yolo_contrastive.eval.run_matrix import _run_detection
            _run_detection(_basic_cell(), _basic_hp(freeze=15, unfreeze_epoch=7))
        finally:
            _MockYOLO.train = original_train

        assert captured_env["YCL_PRETRAINED"] == "/fake/backbone.pt"
        assert captured_env["YCL_FREEZE_BACKBONE"] == "15"
        assert captured_env["YCL_UNFREEZE_EPOCH"] == "7"
        assert captured_env["YCL_BACKBONE_LR_SCALE"] == "0.5"

    def test_env_vars_restored_after_success(self, monkeypatch):
        """Env state before runner == env state after runner (success path)."""
        # Pre-set some vars to non-default values
        monkeypatch.setenv("YCL_PRETRAINED", "original_value")
        monkeypatch.delenv("YCL_FREEZE_BACKBONE", raising=False)

        from yolo_contrastive.eval.run_matrix import _run_detection
        _run_detection(_basic_cell(), _basic_hp())

        # Pre-existing var should be back to its original value
        assert os.environ.get("YCL_PRETRAINED") == "original_value"
        # Previously-unset var should be unset again
        assert "YCL_FREEZE_BACKBONE" not in os.environ

    def test_env_vars_restored_after_train_failure(self, monkeypatch):
        """Env must restore even if YOLO.train raises."""
        monkeypatch.setenv("YCL_PRETRAINED", "original_value")
        monkeypatch.delenv("YCL_UNFREEZE_EPOCH", raising=False)

        _MockYOLO.train_raises = RuntimeError("simulated CUDA OOM")

        from yolo_contrastive.eval.run_matrix import _run_detection
        with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
            _run_detection(_basic_cell(), _basic_hp())

        # Still restored despite exception
        assert os.environ.get("YCL_PRETRAINED") == "original_value"
        assert "YCL_UNFREEZE_EPOCH" not in os.environ


# ─────────────────────────────────────────────────────────────────────────
# (b) hp parameters forwarded to YOLO.train
# ─────────────────────────────────────────────────────────────────────────


class TestHyperparamForward:
    def test_train_kwargs_use_hp_values(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        _run_detection(
            _basic_cell(),
            _basic_hp(epochs=20, imgsz=512, batch=24),
        )
        kw = _MockYOLO.last_train_kwargs
        assert kw["epochs"] == 20
        assert kw["imgsz"] == 512
        assert kw["batch"] == 24
        assert kw["data"] == "/fake/data.yaml"

    def test_yolo_constructor_uses_base_model(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        _run_detection(_basic_cell(), _basic_hp(base_model="yolov8s.pt"))
        assert _MockYOLO.last_init == "yolov8s.pt"


# ─────────────────────────────────────────────────────────────────────────
# (c) Return shape contract
# ─────────────────────────────────────────────────────────────────────────


class TestReturnShape:
    def test_returns_expected_keys(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        _MockYOLO.train_result = _MockResults(_MockBox(0.33, 0.66, 0.5, 0.4))
        out = _run_detection(_basic_cell(), _basic_hp())
        assert set(out.keys()) == {"metric", "metric_value", "mAP50",
                                    "precision", "recall"}

    def test_metric_value_is_map50_95(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        _MockYOLO.train_result = _MockResults(_MockBox(map_=0.42, map50=0.78))
        out = _run_detection(_basic_cell(), _basic_hp())
        assert out["metric"] == "mAP50-95"
        assert out["metric_value"] == pytest.approx(0.42)
        assert out["mAP50"] == pytest.approx(0.78)

    def test_returns_floats_not_tensors(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        out = _run_detection(_basic_cell(), _basic_hp())
        for k in ("metric_value", "mAP50", "precision", "recall"):
            assert isinstance(out[k], float), f"{k!r} is {type(out[k]).__name__}"


# ─────────────────────────────────────────────────────────────────────────
# (e) Defaults
# ─────────────────────────────────────────────────────────────────────────


class TestDefaults:
    def test_empty_hp_uses_paper_grade_defaults(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        _run_detection(_basic_cell(), {})  # no hp at all
        kw = _MockYOLO.last_train_kwargs
        assert kw["epochs"] == 30           # plan §13.7 paper-grade default
        assert kw["imgsz"] == 640
        assert kw["batch"] == 16
        assert _MockYOLO.last_init == "yolov8n.pt"


# ─────────────────────────────────────────────────────────────────────────
# (f) Cell ID in run name
# ─────────────────────────────────────────────────────────────────────────


class TestRunName:
    def test_run_name_uses_cell_id_short(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        _run_detection(
            _basic_cell(cell_id="abc12345xyz0"),
            _basic_hp(),
        )
        assert _MockYOLO.last_train_kwargs["name"] == "cell_abc12345"

    def test_run_name_falls_back_without_cell_id(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        cell = _basic_cell()
        cell.pop("cell_id")
        _run_detection(cell, _basic_hp())
        kw = _MockYOLO.last_train_kwargs
        # Format: {method}_{dataset}_seed{seed}
        assert "test_method" in kw["name"]
        assert "test_ds" in kw["name"]
        assert "seed42" in kw["name"]


# ─────────────────────────────────────────────────────────────────────────
# (g) Required field validation
# ─────────────────────────────────────────────────────────────────────────


class TestRequiredFields:
    def test_baseline_no_backbone_ckpt_runs(self):
        """Baseline cell: no backbone_ckpt, method carries base_model — must NOT raise."""
        from yolo_contrastive.eval.run_matrix import _run_detection
        cell = _basic_cell()
        cell["method"] = {"name": "scratch", "base_model": "yolov8n.yaml"}  # no backbone_ckpt
        out = _run_detection(cell, _basic_hp(base_model="yolov8n.pt"))
        assert _MockYOLO.train_call_count == 1
        assert _MockYOLO.last_init == "yolov8n.yaml"   # per-method base_model overrides hp
        assert "metric_value" in out

    def test_missing_data_yaml_raises(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        cell = _basic_cell()
        cell["dataset"]["data_yaml"] = ""
        with pytest.raises(ValueError, match="data_yaml"):
            _run_detection(cell, _basic_hp())

    def test_invalid_results_no_box_raises(self):
        from yolo_contrastive.eval.run_matrix import _run_detection
        # Results object without .box attribute
        class _BadResults:
            pass
        _MockYOLO.train_result = _BadResults()
        with pytest.raises(RuntimeError, match="no .box attribute"):
            _run_detection(_basic_cell(), _basic_hp())


# ─────────────────────────────────────────────────────────────────────────
# (h) Integration with RunMatrix dispatcher (mock-based end-to-end)
# ─────────────────────────────────────────────────────────────────────────


class TestRunMatrixIntegration:
    def test_run_matrix_invokes_detection_runner(self, tmp_path):
        """Full RunMatrix.run() call with task=detection should hit _run_detection."""
        from yolo_contrastive.eval.run_matrix import RunMatrix

        cfg = {
            "task": "detection",
            "methods": [{"name": "m1", "backbone_ckpt": "/fake/m1.pt"}],
            "datasets": [{"name": "d1", "data_yaml": "/fake/d1.yaml"}],
            "fractions": [1.0],
            "seeds": [42],
            "hp": {"epochs": 1, "imgsz": 320, "batch": 8},
        }
        csv_path = str(tmp_path / "results.csv")
        rm = RunMatrix(config=cfg, output_csv=csv_path)

        results = rm.run(verbose=False)
        assert len(results) == 1
        assert results[0]["status"] == "ok"
        assert results[0]["metric"] == "mAP50-95"
        assert _MockYOLO.train_call_count == 1
