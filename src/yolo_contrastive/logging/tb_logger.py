"""TBLogger — TensorBoard integration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseLogger


class TBLogger(BaseLogger):
    """TensorBoard logger.

    Args:
        log_dir: TensorBoard log klasörü
        project: Proje adı (log_dir prefix)
        name: Run adı (log_dir suffix)
    """

    def __init__(self, log_dir: str = "runs/tb_logs",
                 project: str = "", name: str = "",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(project=project, name=name, config=config)
        self._writer = None
        self._available = False

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._SummaryWriter = SummaryWriter
            self._available = True
        except ImportError:
            print("[ycl-log] WARN: tensorboard not installed. pip install tensorboard")
            return

        import os
        if name:
            log_dir = os.path.join(log_dir, name)
        self._writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        print(f"[ycl-log] TensorBoard: {log_dir}")

        # Config'i text olarak logla
        if config:
            config_str = "\n".join(f"- {k}: {v}" for k, v in config.items())
            self._writer.add_text("config", config_str)

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self._available or self._writer is None:
            return
        s = self._resolve_step(step)
        self._writer.add_scalar(key, value, global_step=s)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._available or self._writer is None:
            return
        s = self._resolve_step(step)
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step=s)

    def log_image(self, key: str, image, step: Optional[int] = None) -> None:
        if not self._available or self._writer is None:
            return
        s = self._resolve_step(step)
        import torch
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # [C, H, W]
                self._writer.add_image(key, image, global_step=s)
            elif image.dim() == 4:  # [B, C, H, W]
                from torchvision.utils import make_grid
                self._writer.add_image(key, make_grid(image), global_step=s)

    def log_config(self, config: Dict[str, Any]) -> None:
        super().log_config(config)
        if self._available and self._writer is not None:
            self._writer.add_hparams(
                {k: str(v) for k, v in config.items()},
                {}
            )

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
            print("[ycl-log] TensorBoard writer closed")

    def __repr__(self) -> str:
        return f"TBLogger(log_dir={getattr(self, 'log_dir', 'N/A')!r})"
