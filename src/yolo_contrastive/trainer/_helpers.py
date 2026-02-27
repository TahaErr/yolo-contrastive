"""Internal helpers for the trainer module.

Contains: env var readers, safe_scalar, loss extraction, BN preserver.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
try:
    from ultralytics.utils import LOGGER as _LOGGER
except Exception:
    _LOGGER = None

try:
    from ultralytics.utils import RANK as _RANK
except Exception:
    _RANK = 0


def is_main_process(trainer) -> bool:
    r = getattr(trainer, "rank", None)
    if r is None:
        r = _RANK
    try:
        r = int(r)
    except Exception:
        r = 0
    return r in (-1, 0)


def log(msg: str) -> None:
    if _LOGGER is not None:
        _LOGGER.info(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# BatchNorm type tuple — public API only (audit §3.1)
# ---------------------------------------------------------------------------
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


# ---------------------------------------------------------------------------
# Env-var helpers — _config.py'den import (tek kaynak)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Safe scalar (audit §3.8)
# ---------------------------------------------------------------------------
def safe_scalar(x: Any) -> float:
    if x is None:
        return 0.0
    if not torch.is_tensor(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    return float(x.detach().float().mean().item())


# ---------------------------------------------------------------------------
# Loss extraction (audit §3.3)
# ---------------------------------------------------------------------------
def _pick_loss_index_tuple(out: tuple) -> Optional[int]:
    for i, v in enumerate(out):
        if torch.is_tensor(v) and v.numel() == 1 and v.requires_grad:
            return i
    for i, v in enumerate(out):
        if torch.is_tensor(v) and v.numel() == 1:
            return i
    for i, v in enumerate(out):
        if torch.is_tensor(v):
            return i
    return None


def extract_loss_from_out(out: Any) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    if isinstance(out, tuple) and len(out) >= 1:
        idx = _pick_loss_index_tuple(out)
        if idx is None:
            return None, None
        loss_t = out[idx]
        return (loss_t, idx) if torch.is_tensor(loss_t) else (None, None)
    if isinstance(out, dict) and "loss" in out and torch.is_tensor(out["loss"]):
        return out["loss"], None
    if torch.is_tensor(out):
        return out, None
    return None, None


def replace_in_output(out: Any, idx: Optional[int], new_val: torch.Tensor) -> Any:
    if isinstance(out, tuple) and idx is not None:
        if hasattr(out, "_replace") and hasattr(out, "_fields"):
            return out._replace(**{out._fields[idx]: new_val})
        out_list = list(out)
        out_list[idx] = new_val
        return tuple(out_list)
    if isinstance(out, dict) and "loss" in out:
        return {**out, "loss": new_val}
    if torch.is_tensor(out):
        return new_val
    return out


# ---------------------------------------------------------------------------
# BN running-stats preserver (audit §3.1, §3.2)
# ---------------------------------------------------------------------------
def preserve_bn_running_stats(model: nn.Module):
    class _Ctx:
        def __init__(self, m):
            self.m = m
            self.bns: List[nn.Module] = []
            self.saved: list = []

        def __enter__(self):
            for mod in self.m.modules():
                if isinstance(mod, BN_TYPES):
                    self.bns.append(mod)
                    rm = mod.running_mean.detach().clone() if mod.running_mean is not None else None
                    rv = mod.running_var.detach().clone() if mod.running_var is not None else None
                    nbt = mod.num_batches_tracked.detach().clone() if getattr(mod, "num_batches_tracked", None) is not None else None
                    self.saved.append((rm, rv, nbt))
            return self

        def __exit__(self, exc_type, exc, tb):
            for mod, (rm, rv, nbt) in zip(self.bns, self.saved):
                try:
                    if rm is not None and mod.running_mean is not None:
                        mod.running_mean.data.copy_(rm)
                    if rv is not None and mod.running_var is not None:
                        mod.running_var.data.copy_(rv)
                    if nbt is not None and getattr(mod, "num_batches_tracked", None) is not None:
                        mod.num_batches_tracked.data.copy_(nbt)
                except Exception as e:
                    log(f"[yolo-contrastive] WARN: Failed to restore BN stats for {mod.__class__.__name__}: {e}")
            return False

    return _Ctx(model)
