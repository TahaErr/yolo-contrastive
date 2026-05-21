"""Tests for PipelineConfig.ssl_method + run_ssl dense/legacy dispatch.

Adım 3 (§13.8) — pipeline.py modern-hat rewire.

run_ssl() now dispatches on cfg.ssl_method:
    "dense"  → DenseSSLPretrainer (modern DT-SAPS hat — default)
    "legacy" → SSLPretrainer      (rotation/composite-pretext hat)
    other    → ConfigError

Mock tests (fast) verify the dispatch wiring + constructor kwargs without
real training, by monkeypatching the pretrainer classes on the
yolo_contrastive.pretrain module (run_ssl imports them lazily from there).

Slow tests run a real 1-epoch pretrain through each hat to confirm the
drop-in train() API parity holds end-to-end.
"""

from __future__ import annotations

import os

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Fake pretrainers — capture constructor + train kwargs, no real work
# ─────────────────────────────────────────────────────────────────────────


class _FakePretrainer:
    """Records init/train kwargs; train() touches a real file at `output`."""

    last_init: dict = {}
    last_train: dict = {}
    cleanup_called: bool = False

    def __init__(self, **kwargs):
        type(self).last_init = dict(kwargs)
        type(self).cleanup_called = False

    def train(self, **kwargs):
        type(self).last_train = dict(kwargs)
        output = kwargs["output"]
        # Produce a real (tiny) file so run_ssl's return value is usable.
        with open(output, "w") as f:
            f.write("fake-backbone")
        return output

    def cleanup(self):
        type(self).cleanup_called = True


class _FakeDense(_FakePretrainer):
    last_init: dict = {}
    last_train: dict = {}
    cleanup_called: bool = False


class _FakeLegacy(_FakePretrainer):
    last_init: dict = {}
    last_train: dict = {}
    cleanup_called: bool = False


@pytest.fixture
def patched_pretrainers(monkeypatch):
    """Swap DenseSSLPretrainer + SSLPretrainer on the pretrain module.

    run_ssl does `from .pretrain import DenseSSLPretrainer` / `SSLPretrainer`
    lazily, so patching the attribute on the module object is sufficient.
    """
    import yolo_contrastive.pretrain as pretrain_mod

    monkeypatch.setattr(pretrain_mod, "DenseSSLPretrainer", _FakeDense)
    monkeypatch.setattr(pretrain_mod, "SSLPretrainer", _FakeLegacy)
    # Reset capture state
    for cls in (_FakeDense, _FakeLegacy):
        cls.last_init = {}
        cls.last_train = {}
        cls.cleanup_called = False
    return pretrain_mod


# ═════════════════════════════════════════════════════════════════════════
# 1 — PipelineConfig.ssl_method default
# ═════════════════════════════════════════════════════════════════════════


def test_ssl_method_default_is_dense():
    from yolo_contrastive import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.ssl_method == "dense"


# ═════════════════════════════════════════════════════════════════════════
# 2 — dense/SAPS config fields exist with production-validated defaults
# ═════════════════════════════════════════════════════════════════════════


def test_dense_saps_config_defaults():
    from yolo_contrastive import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.ssl_out_dim == 128
    assert cfg.ssl_queue_size == 4096
    assert cfg.ssl_momentum == 0.99
    assert cfg.ssl_n_query == 128
    assert cfg.ssl_pos_radius == 0.07
    assert cfg.ssl_match_mode == "threshold"
    assert cfg.ssl_saps_mode == "both"
    assert cfg.ssl_saps_t_scale == 1.0
    assert cfg.ssl_saps_both_lambda == 1.0
    assert cfg.ssl_queue_update_strategy == "pooled"
    # Legacy fields still present (ablation/baseline hat)
    assert cfg.ssl_aug_preset == "simclr_v2"
    assert cfg.ssl_lambda_rot == 0.5


# ═════════════════════════════════════════════════════════════════════════
# 3 — from_dict accepts new keys, still drops unknown ones
# ═════════════════════════════════════════════════════════════════════════


def test_from_dict_accepts_ssl_method_and_saps():
    from yolo_contrastive import PipelineConfig

    cfg = PipelineConfig.from_dict({
        "ssl_method": "legacy",
        "ssl_saps_mode": "within",
        "ssl_queue_size": 8192,
        "bogus_unknown_key": 123,   # must be dropped
    })
    assert cfg.ssl_method == "legacy"
    assert cfg.ssl_saps_mode == "within"
    assert cfg.ssl_queue_size == 8192
    assert not hasattr(cfg, "bogus_unknown_key")


# ═════════════════════════════════════════════════════════════════════════
# 4 — run_ssl dispatches to DenseSSLPretrainer when ssl_method="dense"
# ═════════════════════════════════════════════════════════════════════════


def test_run_ssl_dense_dispatch(patched_pretrainers, tmp_path):
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

    images = tmp_path / "imgs"
    images.mkdir()
    out = tmp_path / "dense_bb.pt"

    cfg = PipelineConfig(ssl_method="dense", backbone_path=str(out))
    pipe = SSLFinetunePipeline(config=cfg)
    result = pipe.run_ssl(images_dir=str(images))

    # Dense fake was used, legacy fake untouched
    assert _FakeDense.last_init != {}
    assert _FakeLegacy.last_init == {}
    assert result == str(out)
    assert os.path.exists(out)
    assert _FakeDense.cleanup_called is True


# ═════════════════════════════════════════════════════════════════════════
# 5 — run_ssl dispatches to SSLPretrainer when ssl_method="legacy"
# ═════════════════════════════════════════════════════════════════════════


def test_run_ssl_legacy_dispatch(patched_pretrainers, tmp_path):
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

    images = tmp_path / "imgs"
    images.mkdir()
    out = tmp_path / "legacy_bb.pt"

    cfg = PipelineConfig(ssl_method="legacy", backbone_path=str(out))
    pipe = SSLFinetunePipeline(config=cfg)
    result = pipe.run_ssl(images_dir=str(images))

    assert _FakeLegacy.last_init != {}
    assert _FakeDense.last_init == {}
    assert result == str(out)
    assert _FakeLegacy.cleanup_called is True


# ═════════════════════════════════════════════════════════════════════════
# 6 — unknown ssl_method raises ConfigError
# ═════════════════════════════════════════════════════════════════════════


def test_run_ssl_unknown_method_raises(patched_pretrainers, tmp_path):
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig
    from yolo_contrastive.exceptions import ConfigError

    images = tmp_path / "imgs"
    images.mkdir()

    cfg = PipelineConfig(ssl_method="bogus_hat", backbone_path=str(tmp_path / "x.pt"))
    pipe = SSLFinetunePipeline(config=cfg)
    with pytest.raises(ConfigError, match="ssl_method"):
        pipe.run_ssl(images_dir=str(images))


# ═════════════════════════════════════════════════════════════════════════
# 7 — dense SAPS params forwarded to DenseSSLPretrainer constructor
# ═════════════════════════════════════════════════════════════════════════


def test_dense_constructor_receives_saps_params(patched_pretrainers, tmp_path):
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

    images = tmp_path / "imgs"
    images.mkdir()

    cfg = PipelineConfig(
        ssl_method="dense", backbone_path=str(tmp_path / "bb.pt"),
        ssl_saps_mode="cross", ssl_queue_size=2048, ssl_n_query=64,
        ssl_match_mode="nearest", imgsz=320,
    )
    pipe = SSLFinetunePipeline(config=cfg)
    pipe.run_ssl(images_dir=str(images))

    init = _FakeDense.last_init
    assert init["saps_mode"] == "cross"
    assert init["queue_size"] == 2048
    assert init["n_query"] == 64
    assert init["match_mode"] == "nearest"
    assert init["imgsz"] == 320
    # Dense constructor must NOT receive legacy-only kwargs
    assert "aug_preset" not in init
    assert "lambda_rot" not in init


# ═════════════════════════════════════════════════════════════════════════
# 8 — legacy params forwarded to SSLPretrainer constructor
# ═════════════════════════════════════════════════════════════════════════


def test_legacy_constructor_receives_legacy_params(patched_pretrainers, tmp_path):
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

    images = tmp_path / "imgs"
    images.mkdir()

    cfg = PipelineConfig(
        ssl_method="legacy", backbone_path=str(tmp_path / "bb.pt"),
        ssl_aug_preset="byol", ssl_lambda_rot=0.3, ssl_lambda_cl=0.8,
    )
    pipe = SSLFinetunePipeline(config=cfg)
    pipe.run_ssl(images_dir=str(images))

    init = _FakeLegacy.last_init
    assert init["aug_preset"] == "byol"
    assert init["lambda_rot"] == 0.3
    assert init["lambda_cl"] == 0.8
    # Legacy constructor must NOT receive dense-only kwargs
    assert "saps_mode" not in init
    assert "queue_size" not in init


# ═════════════════════════════════════════════════════════════════════════
# 9 — train() kwargs identical regardless of hat (API parity)
# ═════════════════════════════════════════════════════════════════════════


def test_train_kwargs_parity_across_hats(patched_pretrainers, tmp_path):
    from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

    images = tmp_path / "imgs"
    images.mkdir()

    common = dict(
        ssl_epochs=3, ssl_batch=8, ssl_lr=2e-3, ssl_warmup_epochs=1,
        ssl_num_workers=0, ssl_save_every=0, ssl_print_every=2,
    )
    for method, fake in (("dense", _FakeDense), ("legacy", _FakeLegacy)):
        cfg = PipelineConfig(
            ssl_method=method, backbone_path=str(tmp_path / f"{method}.pt"),
            **common,
        )
        SSLFinetunePipeline(config=cfg).run_ssl(images_dir=str(images))
        tr = fake.last_train
        assert tr["epochs"] == 3
        assert tr["batch_size"] == 8
        assert tr["lr"] == 2e-3
        assert tr["warmup_epochs"] == 1
        assert tr["num_workers"] == 0
        assert tr["save_every"] == 0
        assert tr["print_every"] == 2


# ═════════════════════════════════════════════════════════════════════════
# 10 — real dense 1-epoch pretrain smoke (drop-in train() works)
# ═════════════════════════════════════════════════════════════════════════


class TestRealPretrainSmoke:
    """End-to-end: a real 1-epoch run through each hat via the pipeline."""

    @pytest.mark.slow
    def test_dense_real_1epoch(self, dummy_images, tmp_path):
        import torch
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        out = tmp_path / "dense_real.pt"
        cfg = PipelineConfig(
            ssl_method="dense", model="yolov8n.pt", imgsz=64, device="cpu",
            ssl_epochs=1, ssl_batch=2, ssl_warmup_epochs=0,
            ssl_num_workers=0, ssl_print_every=1, ssl_save_every=0,
            ssl_out_dim=16, ssl_queue_size=8, ssl_n_query=4,
            backbone_path=str(out),
        )
        pipe = SSLFinetunePipeline(config=cfg)
        result = pipe.run_ssl(images_dir=dummy_images)

        assert result == str(out)
        assert out.exists()
        # Dense checkpoint carries the dense_ssl marker
        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert ckpt.get("extra", {}).get("type") == "dense_ssl"

    @pytest.mark.slow
    def test_legacy_real_1epoch(self, dummy_images, tmp_path):
        from yolo_contrastive import SSLFinetunePipeline, PipelineConfig

        out = tmp_path / "legacy_real.pt"
        cfg = PipelineConfig(
            ssl_method="legacy", model="yolov8n.pt", imgsz=64, device="cpu",
            ssl_epochs=1, ssl_batch=2, ssl_warmup_epochs=0,
            ssl_num_workers=0, ssl_print_every=1, ssl_save_every=0,
            backbone_path=str(out),
        )
        pipe = SSLFinetunePipeline(config=cfg)
        result = pipe.run_ssl(images_dir=dummy_images)

        assert result == str(out)
        assert out.exists()
