"""Test CLConfig."""

import os
import pytest
from yolo_contrastive._config import CLConfig
from yolo_contrastive import ConfigError


def test_defaults():
    cfg = CLConfig()
    assert not cfg.enabled
    assert not cfg.pretext_enabled
    assert not cfg.rotation_enabled
    assert not cfg.adapter_enabled


def test_pretext_enabled():
    cfg = CLConfig(pretext_tasks=["freq_band"], lambda_pretext=0.5,
                   pretext_weights=[1.0])
    assert cfg.pretext_enabled
    assert not cfg.rotation_enabled


def test_rotation_enabled():
    cfg = CLConfig(lambda_rot=0.5)
    assert cfg.rotation_enabled
    assert not cfg.pretext_enabled


def test_adapter_enabled():
    cfg = CLConfig(adapter_type="freq_gated", adapter_rank=8)
    assert cfg.adapter_enabled
    assert cfg.adapter_rank == 8


def test_negative_lambda_raises():
    with pytest.raises(ConfigError):
        CLConfig(lambda_cl=-1.0).validate()


def test_negative_temp_raises():
    with pytest.raises(ConfigError):
        CLConfig(temperature=-0.1).validate()


def test_from_env():
    os.environ["YCL_LAMBDA"] = "0.1"
    os.environ["YCL_PRETEXT_TASKS"] = "freq_band,blur"
    os.environ["YCL_LAMBDA_PRETEXT"] = "0.3"
    os.environ["YCL_ADAPTER"] = "task_routed"
    os.environ["YCL_ADAPTER_RANK"] = "16"
    try:
        cfg = CLConfig.from_env()
        assert cfg.lambda_cl == 0.1
        assert cfg.pretext_tasks == ["freq_band", "blur"]
        assert cfg.lambda_pretext == 0.3
        assert cfg.adapter_type == "task_routed"
        assert cfg.adapter_rank == 16
    finally:
        for k in ["YCL_LAMBDA", "YCL_PRETEXT_TASKS", "YCL_LAMBDA_PRETEXT",
                   "YCL_PRETEXT_WEIGHTS", "YCL_LAMBDA_ROT",
                   "YCL_ADAPTER", "YCL_ADAPTER_RANK"]:
            os.environ.pop(k, None)
