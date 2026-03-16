"""BaseLogger — abstract logging interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLogger(ABC):
    """Tüm logger'ların base class'ı.

    Her logger şu lifecycle'ı izler:
        1. __init__(project, name, config)
        2. log_scalar / log_scalars (her step)
        3. log_image (opsiyonel)
        4. finish()

    Kullanım:
        logger = WandBLogger(project="yolo-ssl", name="exp_A")
        for step in range(100):
            logger.log_scalars({"loss": 0.5, "acc": 0.8}, step=step)
        logger.finish()
    """

    def __init__(self, project: str = "", name: str = "",
                 config: Optional[Dict[str, Any]] = None):
        self.project = project
        self.name = name
        self.config = config or {}
        self._step = 0

    @abstractmethod
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Tek bir skalar değer logla."""
        ...

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Birden fazla skalar değer logla."""
        for key, value in metrics.items():
            self.log_scalar(key, value, step=step)

    def log_image(self, key: str, image, step: Optional[int] = None) -> None:
        """Görüntü logla (opsiyonel, desteklemeyen logger'lar skip eder)."""
        pass

    def log_config(self, config: Dict[str, Any]) -> None:
        """Hyperparameter config logla."""
        self.config.update(config)

    @abstractmethod
    def finish(self) -> None:
        """Logger'ı kapat, kaynakları serbest bırak."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False

    def _resolve_step(self, step: Optional[int]) -> int:
        if step is not None:
            self._step = step
            return step
        self._step += 1
        return self._step
