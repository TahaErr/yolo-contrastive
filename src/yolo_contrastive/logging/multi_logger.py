"""MultiLogger — birden fazla logger'ı aynı anda kullan."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLogger


class MultiLogger(BaseLogger):
    """Birden fazla logger'a aynı anda yazar.

    Kullanım:
        logger = MultiLogger([
            CSVLogger(save_dir="runs/"),
            WandBLogger(project="yolo-ssl"),
            TBLogger(log_dir="runs/tb"),
        ])
        logger.log_scalars({"loss": 0.5}, step=1)
        logger.finish()
    """

    def __init__(self, loggers: List[BaseLogger],
                 project: str = "", name: str = "",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(project=project, name=name, config=config)
        self.loggers = loggers

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_scalar(key, value, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_scalars(metrics, step=step)

    def log_image(self, key: str, image, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_image(key, image, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        super().log_config(config)
        for logger in self.loggers:
            logger.log_config(config)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()

    def __repr__(self) -> str:
        names = [type(l).__name__ for l in self.loggers]
        return f"MultiLogger({names})"
