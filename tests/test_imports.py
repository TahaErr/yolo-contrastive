"""Top-level public API import smoke tests.

Verifies that every name in yolo_contrastive's lazy __all__ resolves, that
``import yolo_contrastive`` stays lightweight (no eager optional deps), and
that the exception hierarchy is intact.
"""

from __future__ import annotations

import subprocess
import sys

import yolo_contrastive


# ─────────────────────────────────────────────────────────────────────────
# version + __all__ surface
# ─────────────────────────────────────────────────────────────────────────


def test_version():
    assert yolo_contrastive.__version__ == "0.2.0"
    assert isinstance(yolo_contrastive.__version__, str)


def test_all_names_resolve():
    """Every name in __all__ is actually accessible via lazy __getattr__."""
    for name in yolo_contrastive.__all__:
        obj = getattr(yolo_contrastive, name)
        assert obj is not None, f"{name} resolved to None"


def test_dir_lists_lazy_names():
    """dir() exposes the lazy names (tab-completion / discoverability)."""
    listing = dir(yolo_contrastive)
    for name in ("DenseSSLPretrainer", "DualTeacherTrainer", "NTXentLoss"):
        assert name in listing, f"{name} missing from dir()"


def test_unknown_attribute_raises():
    """An unknown top-level name raises AttributeError, not ImportError."""
    import pytest
    with pytest.raises(AttributeError):
        _ = yolo_contrastive.NoSuchSymbol


# ─────────────────────────────────────────────────────────────────────────
# lazy import is lightweight
# ─────────────────────────────────────────────────────────────────────────


def test_import_does_not_pull_ultralytics():
    """`import yolo_contrastive` must not eagerly import ultralytics.

    Run in a fresh subprocess so other tests' imports don't pollute the
    check. Lazy __getattr__ means optional deps load only on first use.
    """
    code = (
        "import yolo_contrastive, sys; "
        "assert 'ultralytics' not in sys.modules, "
        "'import yolo_contrastive eagerly pulled ultralytics'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# ─────────────────────────────────────────────────────────────────────────
# framework groups — pretraining
# ─────────────────────────────────────────────────────────────────────────


def test_pretrainer_imports():
    from yolo_contrastive import DenseSSLPretrainer, SSLPretrainer
    assert DenseSSLPretrainer is not None
    assert SSLPretrainer is not None


def test_backbone_io_imports():
    from yolo_contrastive import save_backbone, load_backbone
    assert callable(save_backbone)
    assert callable(load_backbone)


# ─────────────────────────────────────────────────────────────────────────
# framework groups — dual-teacher (DT-SAPS)
# ─────────────────────────────────────────────────────────────────────────


def test_dual_teacher_imports():
    from yolo_contrastive import (
        DualTeacherTrainer, CocoTeacher, TeacherCache,
        ConsensusLoss, DisagreementWeighter,
    )
    for obj in (DualTeacherTrainer, CocoTeacher, TeacherCache,
                ConsensusLoss, DisagreementWeighter):
        assert obj is not None


# ─────────────────────────────────────────────────────────────────────────
# framework groups — external baselines
# ─────────────────────────────────────────────────────────────────────────


def test_baseline_imports():
    from yolo_contrastive import (
        SimCLRYOLOTrainer, MoCoV3YOLOTrainer, CoMADYOLOTrainer,
    )
    for obj in (SimCLRYOLOTrainer, MoCoV3YOLOTrainer, CoMADYOLOTrainer):
        assert obj is not None


# ─────────────────────────────────────────────────────────────────────────
# framework groups — evaluation
# ─────────────────────────────────────────────────────────────────────────


def test_eval_imports():
    from yolo_contrastive import LinearProbeTrainer, RunMatrix, run_leakage_check
    assert LinearProbeTrainer is not None
    assert RunMatrix is not None
    assert callable(run_leakage_check)


# ─────────────────────────────────────────────────────────────────────────
# framework groups — fine-tuning
# ─────────────────────────────────────────────────────────────────────────


def test_finetune_import():
    from yolo_contrastive import FinetuneDetectionTrainer
    assert FinetuneDetectionTrainer is not None


# ─────────────────────────────────────────────────────────────────────────
# framework groups — pipeline + core building blocks
# ─────────────────────────────────────────────────────────────────────────


def test_pipeline_imports():
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig, auto_train
    assert SSLFinetunePipeline is not None
    assert PipelineConfig is not None
    assert callable(auto_train)


def test_core_building_block_imports():
    from yolo_contrastive import NTXentLoss, build_contrastive_loss, FeatureTap
    assert NTXentLoss is not None
    assert callable(build_contrastive_loss)
    assert FeatureTap is not None


def test_discovery_imports():
    from yolo_contrastive import discover, DatasetInfo, TrainMode
    assert callable(discover)
    assert DatasetInfo is not None
    assert TrainMode is not None


# ─────────────────────────────────────────────────────────────────────────
# exception hierarchy
# ─────────────────────────────────────────────────────────────────────────


def test_exception_hierarchy():
    from yolo_contrastive import (
        YoloContrastiveError, FeatureTapError,
        ContrastiveLossError, ConfigError, PatchError,
    )
    assert issubclass(FeatureTapError, YoloContrastiveError)
    assert issubclass(ContrastiveLossError, YoloContrastiveError)
    assert issubclass(ConfigError, YoloContrastiveError)
    assert issubclass(PatchError, YoloContrastiveError)
    assert issubclass(YoloContrastiveError, Exception)
