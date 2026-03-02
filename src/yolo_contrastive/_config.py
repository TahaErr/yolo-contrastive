"""CL + pretext task configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .exceptions import ConfigError


def _log_warn(msg):
    try:
        from ultralytics.utils import LOGGER
        LOGGER.info(msg)
    except Exception:
        print(msg)


def env_float(key, default, *, min_val=None):
    v = os.getenv(key)
    if not v or not v.strip():
        return float(default)
    try:
        f = float(v)
    except ValueError:
        _log_warn(f"[ycl] WARN: {key}={v!r} invalid float, default={default}")
        return float(default)
    if min_val is not None and f < min_val:
        import warnings
        warnings.warn(
            f"[ycl] {key}={f} is below minimum {min_val}, clamping to {min_val}. "
            f"Set a valid value to suppress this warning.",
            UserWarning, stacklevel=3,
        )
        return float(min_val)
    return f


def env_str(key, default):
    v = os.getenv(key)
    if not v or not v.strip():
        return default
    return v


def env_int(key, default, *, min_val=None):
    v = os.getenv(key)
    if not v or not v.strip():
        return int(default)
    try:
        i = int(v)
    except ValueError:
        _log_warn(f"[ycl] WARN: {key}={v!r} invalid int, default={default}")
        return int(default)
    if min_val is not None and i < min_val:
        import warnings
        warnings.warn(
            f"[ycl] {key}={i} is below minimum {min_val}, clamping to {min_val}. "
            f"Set a valid value to suppress this warning.",
            UserWarning, stacklevel=3,
        )
        return int(min_val)
    return i


def env_bool(key, default):
    v = os.getenv(key)
    if not v or not v.strip():
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class CLConfig:
    lambda_cl: float = 0.0
    loss_name: str = "ntxent"
    temperature: float = 0.2
    print_every: int = 50
    two_view: bool = False
    pseudo_view: bool = True
    noise_std: float = 1e-3
    aug_preset: str = ""
    lambda_rot: float = 0.0
    rot_hidden_dim: int = 256
    flip_p: float = 0.5
    gray_p: float = 0.2
    blur_p: float = 0.5
    blur_k: int = 5
    blur_sigma: float = 1.0
    brightness_lo: float = 0.6
    brightness_hi: float = 1.4
    contrast_lo: float = 0.6
    contrast_hi: float = 1.4

    @property
    def enabled(self):
        return self.lambda_cl > 0

    @property
    def rotation_enabled(self):
        return self.lambda_rot > 0

    def validate(self):
        if self.lambda_cl < 0:
            raise ConfigError(f"lambda_cl must be >= 0, got {self.lambda_cl}")
        if self.lambda_rot < 0:
            raise ConfigError(f"lambda_rot must be >= 0, got {self.lambda_rot}")
        if self.temperature <= 0:
            raise ConfigError(f"temperature must be > 0, got {self.temperature}")

    @classmethod
    def from_env(cls):
        two_view = env_bool("YCL_TWO_VIEW", False)
        pseudo_view = env_bool("YCL_PSEUDO_VIEW", True)
        if two_view:
            pseudo_view = False
        cfg = cls(
            lambda_cl=env_float("YCL_LAMBDA", 0.0, min_val=0.0),
            loss_name=env_str("YCL_LOSS", "ntxent"),
            temperature=env_float("YCL_TEMP", 0.2, min_val=1e-6),
            print_every=env_int("YCL_PRINT_EVERY", 50, min_val=0),
            two_view=two_view,
            pseudo_view=pseudo_view,
            noise_std=env_float("YCL_NOISE_STD", 1e-3, min_val=0.0),
            aug_preset=env_str("YCL_AUG_PRESET", ""),
            lambda_rot=env_float("YCL_LAMBDA_ROT", 0.0, min_val=0.0),
            rot_hidden_dim=env_int("YCL_ROT_HIDDEN", 256, min_val=32),
            flip_p=env_float("YCL_V2_FLIP_P", 0.5, min_val=0.0),
            gray_p=env_float("YCL_V2_GRAY_P", 0.2, min_val=0.0),
            blur_p=env_float("YCL_V2_BLUR_P", 0.5, min_val=0.0),
            blur_k=env_int("YCL_V2_BLUR_K", 5, min_val=3),
            blur_sigma=env_float("YCL_V2_BLUR_SIGMA", 1.0, min_val=0.01),
            brightness_lo=env_float("YCL_V2_BRIGHT_LO", 0.6),
            brightness_hi=env_float("YCL_V2_BRIGHT_HI", 1.4),
            contrast_lo=env_float("YCL_V2_CONT_LO", 0.6),
            contrast_hi=env_float("YCL_V2_CONT_HI", 1.4),
        )
        cfg.validate()
        return cfg
