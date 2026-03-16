"""WandBLogger — Weights & Biases integration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseLogger


class WandBLogger(BaseLogger):
    """Weights & Biases logger.

    Otomatik olarak:
        - wandb.init() çağırır
        - Her step'te wandb.log() ile metrik gönderir
        - Config'i hyperparameter olarak kaydeder
        - finish()'te wandb.finish() çağırır

    Args:
        project: WandB proje adı
        name: Run adı
        config: Hyperparameter dict
        tags: Run tag'leri
        notes: Run notları
        mode: "online", "offline", "disabled"
    """

    def __init__(self, project: str = "yolo-contrastive",
                 name: str = "", config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None, notes: str = "",
                 mode: str = "online"):
        super().__init__(project=project, name=name, config=config)
        self._run = None
        self._available = False

        try:
            import wandb
            self._wandb = wandb
            self._available = True
        except ImportError:
            print("[ycl-log] WARN: wandb not installed. pip install wandb")
            return

        init_kwargs = {
            "project": project,
            "config": config or {},
            "mode": mode,
        }
        if name:
            init_kwargs["name"] = name
        if tags:
            init_kwargs["tags"] = tags
        if notes:
            init_kwargs["notes"] = notes

        self._run = wandb.init(**init_kwargs)
        print(f"[ycl-log] WandB initialized: {self._run.url or mode}")

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self._available or self._run is None:
            return
        s = self._resolve_step(step)
        self._wandb.log({key: value}, step=s)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._available or self._run is None:
            return
        s = self._resolve_step(step)
        self._wandb.log(metrics, step=s)

    def log_image(self, key: str, image, step: Optional[int] = None) -> None:
        if not self._available or self._run is None:
            return
        s = self._resolve_step(step)
        import numpy as np
        if hasattr(image, "cpu"):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray):
            self._wandb.log({key: self._wandb.Image(image)}, step=s)

    def log_config(self, config: Dict[str, Any]) -> None:
        super().log_config(config)
        if self._available and self._run is not None:
            self._run.config.update(config)

    def finish(self) -> None:
        if self._available and self._run is not None:
            self._run.finish()
            self._run = None
            print("[ycl-log] WandB run finished")

    def __repr__(self) -> str:
        url = self._run.url if self._run else "N/A"
        return f"WandBLogger(project={self.project!r}, url={url})"
