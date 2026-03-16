"""Logging module — pluggable metric logging for training."""

from .base import BaseLogger
from .csv_logger import CSVLogger
from .wandb_logger import WandBLogger
from .tb_logger import TBLogger
from .multi_logger import MultiLogger


def build_logger(backend: str = "csv", **kwargs) -> BaseLogger:
    """Logger factory.

    Args:
        backend: "csv", "wandb", "tensorboard", "tb", "multi", "none"
        **kwargs: Backend-specific arguments

    Returns:
        BaseLogger instance

    Examples:
        logger = build_logger("csv", save_dir="runs/")
        logger = build_logger("wandb", project="yolo-ssl", name="exp_A")
        logger = build_logger("tb", log_dir="runs/tb")
        logger = build_logger("multi", loggers=[
            CSVLogger(save_dir="runs/"),
            WandBLogger(project="yolo-ssl"),
        ])
    """
    if backend == "csv":
        return CSVLogger(**kwargs)
    elif backend == "wandb":
        return WandBLogger(**kwargs)
    elif backend in ("tensorboard", "tb"):
        return TBLogger(**kwargs)
    elif backend == "multi":
        return MultiLogger(**kwargs)
    elif backend == "none":
        return _NullLogger()
    else:
        raise ValueError(f"Unknown logger backend: {backend}. "
                         f"Available: csv, wandb, tb, multi, none")


class _NullLogger(BaseLogger):
    """Hiçbir şey loglamayan dummy logger."""
    def log_scalar(self, key, value, step=None): pass
    def finish(self): pass


__all__ = [
    "BaseLogger", "CSVLogger", "WandBLogger", "TBLogger",
    "MultiLogger", "build_logger",
]
