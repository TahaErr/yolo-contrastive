"""Hat C — UX Façade Hat integration smoke tests.

Covers 17 scenarios from INVENTORY.md §2.3:
    C1-C6:   discovery.py — discover() + DatasetInfo + TrainMode (3 modes + errors)
    C7-C10:  pipeline.py — PipelineConfig + SSLFinetunePipeline construction/discover
    C11-C14: pipeline.py — run_ssl / run_finetune / run() end-to-end
    C15-C17: top-level __init__.py public API + §11.11 sentinel + exception hierarchy

Integration scope:
    Hat C is the user-facing entry layer — auto_train, SSLFinetunePipeline,
    discover. These smoke tests pin the public API so the §13.8 pipeline.py
    rewire (Adım 3, modern-hat integration) can't silently break the
    backward-compatible surface. The §11.11 sentinel (C16) is intentional:
    if a future change exports the modern hat at top level, C16 fails on
    purpose to force a conscious decision.

    C1-C10, C12, C15-C17 are fast (discovery + dataclass + imports).
    C11, C13, C14 run real SSLPretrainer / YOLO training — @pytest.mark.slow.

Note on device:
    Pipeline auto-detects device (0 if cuda else "cpu"). Slow tests pass
    device="cpu" explicitly so they're runtime-independent (lesson from
    the A15 GPU-masking bug).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ═════════════════════════════════════════════════════════════════════════
# C1 — discover() SSL_FINETUNE mode
# ═════════════════════════════════════════════════════════════════════════


class TestC1_DiscoverSSLFinetune:
    """labeled data.yaml + unlabeled dir → TrainMode.SSL_FINETUNE."""

    def test_ssl_finetune_mode(self, dummy_yolo_dataset, dummy_images_dir):
        from yolo_contrastive import discover, TrainMode

        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=2)
        unlabeled = dummy_images_dir(n=5, size=64, name="pool_c1")

        info = discover(data_yaml=ds["data_yaml"], unlabeled_dir=str(unlabeled))
        assert info.mode == TrainMode.SSL_FINETUNE
        assert info.n_train == 6
        assert info.n_unlabeled == 5
        assert info.num_classes == 2


# ═════════════════════════════════════════════════════════════════════════
# C2 — discover() DETECTION mode
# ═════════════════════════════════════════════════════════════════════════


class TestC2_DiscoverDetection:
    """labeled data.yaml only (no unlabeled) → TrainMode.DETECTION."""

    def test_detection_mode(self, dummy_yolo_dataset):
        from yolo_contrastive import discover, TrainMode

        ds = dummy_yolo_dataset(n_train=8, n_val=2, num_classes=3)
        info = discover(data_yaml=ds["data_yaml"])
        assert info.mode == TrainMode.DETECTION
        assert info.n_train == 8
        assert info.num_classes == 3
        assert info.n_unlabeled == 0


# ═════════════════════════════════════════════════════════════════════════
# C3 — discover() SSL_ONLY mode
# ═════════════════════════════════════════════════════════════════════════


class TestC3_DiscoverSSLOnly:
    """unlabeled dir only (no data.yaml) → TrainMode.SSL_ONLY."""

    def test_ssl_only_mode(self, dummy_images_dir):
        from yolo_contrastive import discover, TrainMode

        unlabeled = dummy_images_dir(n=7, size=64, name="pool_c3")
        info = discover(unlabeled_dir=str(unlabeled))
        assert info.mode == TrainMode.SSL_ONLY
        assert info.n_unlabeled == 7
        assert info.n_train == 0


# ═════════════════════════════════════════════════════════════════════════
# C4 — discover() raises when nothing found
# ═════════════════════════════════════════════════════════════════════════


class TestC4_DiscoverNoData:
    """Neither labeled nor unlabeled data → ConfigError."""

    def test_empty_raises_config_error(self, tmp_workspace):
        from yolo_contrastive import discover
        from yolo_contrastive.exceptions import ConfigError

        empty = tmp_workspace / "empty_dir"
        empty.mkdir()
        with pytest.raises(ConfigError):
            discover(dataset_dir=str(empty))


# ═════════════════════════════════════════════════════════════════════════
# C5 — DatasetInfo.summary() + dataset_dir convention
# ═════════════════════════════════════════════════════════════════════════


class TestC5_DatasetInfoSummary:
    """discover via dataset_dir convention; summary() is a readable string."""

    def test_summary_string_and_dataset_dir(self, dummy_yolo_dataset, tmp_workspace):
        from yolo_contrastive import discover, TrainMode

        # dummy_yolo_dataset writes data.yaml at a known path; point
        # dataset_dir at its parent so discover finds data.yaml by convention.
        ds = dummy_yolo_dataset(n_train=5, n_val=2, num_classes=2)
        data_yaml_dir = str(Path(ds["data_yaml"]).parent)

        info = discover(dataset_dir=data_yaml_dir)
        assert info.mode in (TrainMode.DETECTION, TrainMode.SSL_FINETUNE)

        summary = info.summary()
        assert isinstance(summary, str)
        assert "Mode:" in summary
        assert "Classes:" in summary


# ═════════════════════════════════════════════════════════════════════════
# C6 — discover() class names parsing (dict vs list)
# ═════════════════════════════════════════════════════════════════════════


class TestC6_DiscoverClassNames:
    """data.yaml `names` parsed into num_classes + class_names list."""

    def test_class_names_populated(self, dummy_yolo_dataset):
        from yolo_contrastive import discover

        ds = dummy_yolo_dataset(n_train=4, n_val=2, num_classes=3)
        info = discover(data_yaml=ds["data_yaml"])
        assert info.num_classes == 3
        assert info.class_names is not None
        assert len(info.class_names) == 3


# ═════════════════════════════════════════════════════════════════════════
# C7 — PipelineConfig defaults
# ═════════════════════════════════════════════════════════════════════════


class TestC7_PipelineConfigDefaults:
    """PipelineConfig dataclass exposes documented default fields."""

    def test_defaults(self):
        from yolo_contrastive import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.model == "yolov8n.pt"
        assert cfg.imgsz == 640
        # SSL + FT sections exist with sane defaults
        assert cfg.ssl_epochs > 0
        assert cfg.ft_epochs > 0
        assert cfg.ft_freeze_layers == 10
        assert cfg.backbone_path  # non-empty default path


# ═════════════════════════════════════════════════════════════════════════
# C8 — PipelineConfig.from_dict filters unknown keys
# ═════════════════════════════════════════════════════════════════════════


class TestC8_PipelineConfigFromDict:
    """from_dict keeps valid keys, silently drops unknown ones."""

    def test_unknown_keys_dropped(self):
        from yolo_contrastive import PipelineConfig

        cfg = PipelineConfig.from_dict({
            "model": "yolov8s.pt",
            "imgsz": 320,
            "ssl_epochs": 7,
            "totally_bogus_key": 999,   # must be ignored, not crash
        })
        assert cfg.model == "yolov8s.pt"
        assert cfg.imgsz == 320
        assert cfg.ssl_epochs == 7
        assert not hasattr(cfg, "totally_bogus_key")


# ═════════════════════════════════════════════════════════════════════════
# C9 — SSLFinetunePipeline construction (config= and **kwargs)
# ═════════════════════════════════════════════════════════════════════════


class TestC9_PipelineConstruction:
    """Pipeline accepts both an explicit config and loose **kwargs."""

    def test_construct_with_config(self):
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        cfg = PipelineConfig(model="yolov8n.pt", imgsz=320)
        pipe = SSLFinetunePipeline(config=cfg)
        assert pipe.cfg.imgsz == 320
        assert pipe.dataset_info is None
        assert pipe.backbone_path is None

    def test_construct_with_kwargs(self):
        from yolo_contrastive import SSLFinetunePipeline

        pipe = SSLFinetunePipeline(model="yolov8n.pt", imgsz=160, ssl_epochs=3)
        assert pipe.cfg.imgsz == 160
        assert pipe.cfg.ssl_epochs == 3


# ═════════════════════════════════════════════════════════════════════════
# C10 — pipeline.discover_dataset() wires DatasetInfo
# ═════════════════════════════════════════════════════════════════════════


class TestC10_PipelineDiscoverDataset:
    """discover_dataset() stores the DatasetInfo on the pipeline instance."""

    def test_discover_sets_dataset_info(self, dummy_yolo_dataset):
        from yolo_contrastive import SSLFinetunePipeline, TrainMode

        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=2)
        pipe = SSLFinetunePipeline(model="yolov8n.pt")
        info = pipe.discover_dataset(data_yaml=ds["data_yaml"])

        assert info is pipe.dataset_info
        assert info.mode == TrainMode.DETECTION
        assert info.n_train == 6


# ═════════════════════════════════════════════════════════════════════════
# C11 — run_ssl() smoke (real SSL pretraining)
# ═════════════════════════════════════════════════════════════════════════


class TestC11_RunSSL:
    """pipeline.run_ssl() runs a real 1-epoch SSL pretrain, writes backbone."""

    @pytest.mark.slow
    def test_run_ssl_writes_backbone(self, dummy_images_dir, tmp_workspace):
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        unlabeled = dummy_images_dir(n=6, size=64, name="pool_c11")
        backbone_out = tmp_workspace / "c11_backbone.pt"

        cfg = PipelineConfig(
            model="yolov8n.pt", imgsz=64, device="cpu",
            ssl_epochs=1, ssl_batch=2, ssl_warmup_epochs=0,
            ssl_num_workers=0, ssl_print_every=1, ssl_save_every=0,
            backbone_path=str(backbone_out),
        )
        pipe = SSLFinetunePipeline(config=cfg)
        result = pipe.run_ssl(images_dir=str(unlabeled))

        assert result == str(backbone_out)
        assert backbone_out.exists()
        assert pipe.backbone_path == str(backbone_out)
        assert pipe.ssl_time > 0


# ═════════════════════════════════════════════════════════════════════════
# C12 — run_finetune() without backbone raises
# ═════════════════════════════════════════════════════════════════════════


class TestC12_RunFinetuneNoBackbone:
    """run_finetune() with a missing backbone path → FileNotFoundError."""

    def test_missing_backbone_raises(self, dummy_yolo_dataset, tmp_workspace):
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        ds = dummy_yolo_dataset(n_train=4, n_val=2, num_classes=2)
        cfg = PipelineConfig(model="yolov8n.pt", device="cpu")
        pipe = SSLFinetunePipeline(config=cfg)

        # No run_ssl() called → backbone_path is None / nonexistent
        with pytest.raises(FileNotFoundError):
            pipe.run_finetune(
                data_yaml=ds["data_yaml"],
                backbone_path=str(tmp_workspace / "does_not_exist.pt"),
            )


# ═════════════════════════════════════════════════════════════════════════
# C13 — run() full SSL_FINETUNE pipeline
# ═════════════════════════════════════════════════════════════════════════


class TestC13_RunFullPipeline:
    """run() auto-discovers SSL_FINETUNE and executes pretrain → finetune."""

    @pytest.mark.slow
    def test_full_ssl_finetune(
        self, dummy_yolo_dataset, dummy_images_dir, env_isolation, tmp_workspace,
    ):
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=2, imgsz=160)
        unlabeled = dummy_images_dir(n=6, size=64, name="pool_c13")

        cfg = PipelineConfig(
            model="yolov8n.pt", imgsz=160, device="cpu",
            ssl_epochs=1, ssl_batch=2, ssl_warmup_epochs=0,
            ssl_num_workers=0, ssl_print_every=1, ssl_save_every=0,
            ft_epochs=1, ft_batch=2, ft_freeze_layers=10, ft_unfreeze_epoch=0,
            backbone_path=str(tmp_workspace / "c13_backbone.pt"),
            project=str(tmp_workspace / "c13_runs"), name="c13",
        )
        pipe = SSLFinetunePipeline(config=cfg)
        pipe.run(data_yaml=ds["data_yaml"], unlabeled_dir=str(unlabeled))

        # SSL ran → backbone exists; FT ran → ft_time recorded
        assert pipe.backbone_path is not None
        assert os.path.exists(pipe.backbone_path)
        assert pipe.ssl_time > 0
        assert pipe.ft_time > 0

        # summary() reflects a completed SSL_FINETUNE run
        s = pipe.summary()
        assert s["mode"] == "ssl_finetune"
        assert s["total_time_sec"] > 0


# ═════════════════════════════════════════════════════════════════════════
# C14 — run() DETECTION mode
# ═════════════════════════════════════════════════════════════════════════


class TestC14_RunDetectionMode:
    """run() with labeled-only data auto-selects DETECTION and trains."""

    @pytest.mark.slow
    def test_detection_run(self, dummy_yolo_dataset, env_isolation, tmp_workspace):
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        ds = dummy_yolo_dataset(n_train=6, n_val=2, num_classes=2, imgsz=160)
        cfg = PipelineConfig(
            model="yolov8n.pt", imgsz=160, device="cpu",
            ft_epochs=1, ft_batch=2,
            cl_lambda=0.0,  # base detection, no contrastive
            project=str(tmp_workspace / "c14_runs"), name="c14",
        )
        pipe = SSLFinetunePipeline(config=cfg)
        pipe.run(data_yaml=ds["data_yaml"])

        # DETECTION mode → no SSL, finetune time recorded
        assert pipe.dataset_info.mode.value == "detection"
        assert pipe.ft_time > 0
        assert pipe.ssl_time == 0.0


# ═════════════════════════════════════════════════════════════════════════
# C15 — top-level public API surface
# ═════════════════════════════════════════════════════════════════════════


class TestC15_TopLevelAPI:
    """Every symbol in yolo_contrastive.__all__ is importable + present."""

    def test_all_symbols_present(self):
        import yolo_contrastive

        for sym in yolo_contrastive.__all__:
            assert hasattr(yolo_contrastive, sym), (
                f"__all__ lists '{sym}' but it's not on the module"
            )

    def test_key_facade_symbols(self):
        # The documented UX entry points must be reachable from the top.
        from yolo_contrastive import (
            SSLFinetunePipeline, PipelineConfig, auto_train,
            discover, DatasetInfo, TrainMode,
        )
        assert callable(auto_train)
        assert callable(discover)


# ═════════════════════════════════════════════════════════════════════════
# C16 — §11.11 sentinel: modern hat NOT exported at top level
# ═════════════════════════════════════════════════════════════════════════


class TestC16_ModernHatExportSentinel:
    """WORK_PLAN_v9 §11.11 sentinel — the modern dense hat (DenseSSLPretrainer,
    dense.*) is intentionally NOT exported from the top-level package.

    If the §13.8 pipeline rewire later promotes the modern hat to the top
    level, THIS TEST FAILS ON PURPOSE — forcing a conscious __all__ update
    rather than a silent surface change."""

    def test_modern_hat_absent_from_top_level(self):
        import yolo_contrastive

        # Sentinel: these are NOT on the top-level namespace today.
        for modern_sym in ("DenseSSLPretrainer", "MultiScaleFeatureTap",
                            "PretrainMatrix", "RunMatrix"):
            assert not hasattr(yolo_contrastive, modern_sym), (
                f"§11.11 sentinel tripped: '{modern_sym}' is now exported at "
                f"top level. If intentional, update __all__ + this test."
            )

    def test_modern_hat_reachable_via_submodule(self):
        # ...but the modern hat IS reachable via its submodule path.
        from yolo_contrastive.pretrain import DenseSSLPretrainer
        from yolo_contrastive.dense import MultiScaleFeatureTap
        assert DenseSSLPretrainer is not None
        assert MultiScaleFeatureTap is not None


# ═════════════════════════════════════════════════════════════════════════
# C17 — exception hierarchy
# ═════════════════════════════════════════════════════════════════════════


class TestC17_ExceptionHierarchy:
    """All library exceptions derive from YoloContrastiveError — so callers
    can catch the base class and handle any library error uniformly."""

    def test_all_derive_from_base(self):
        from yolo_contrastive.exceptions import (
            YoloContrastiveError, FeatureTapError,
            ContrastiveLossError, ConfigError, PatchError,
        )
        for exc in (FeatureTapError, ContrastiveLossError, ConfigError, PatchError):
            assert issubclass(exc, YoloContrastiveError), (
                f"{exc.__name__} does not derive from YoloContrastiveError"
            )

    def test_base_is_exception(self):
        from yolo_contrastive.exceptions import YoloContrastiveError
        assert issubclass(YoloContrastiveError, Exception)
