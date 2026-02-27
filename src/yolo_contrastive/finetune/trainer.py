"""FinetuneDetectionTrainer — pretrained backbone ile az etiketli fine-tuning."""

from __future__ import annotations

import os

import torch

from ..pretrain.backbone_utils import load_backbone, unfreeze_all

DetectionTrainer = None
try:
    from ultralytics.models.yolo.detect.train import DetectionTrainer as _DT
    DetectionTrainer = _DT
except Exception:
    pass
if DetectionTrainer is None:
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer as _DT
        DetectionTrainer = _DT
    except Exception:
        pass
if DetectionTrainer is None:
    raise ImportError("Could not import DetectionTrainer from ultralytics")

BACKBONE_LAYERS = 10


def _log(msg: str):
    try:
        from ultralytics.utils import LOGGER
        LOGGER.info(msg)
    except Exception:
        print(msg)


class FinetuneDetectionTrainer(DetectionTrainer):
    """YOLO fine-tuning trainer with pretrained SSL backbone."""

    def setup_model(self):
        super().setup_model()
        pretrained_path = os.environ.get("YCL_PRETRAINED", "")
        if not pretrained_path or not os.path.exists(pretrained_path):
            if pretrained_path:
                _log(f"[ycl-ft] WARN: Pretrained not found: {pretrained_path}")
            return
        n_loaded = load_backbone(self.model, pretrained_path, strict=False, verbose=True, backbone_only=True)
        _log(f"[ycl-ft] Loaded pretrained backbone: {pretrained_path} ({n_loaded} params)")
        self._ycl_freeze_layers = int(os.environ.get("YCL_FREEZE_BACKBONE", "10"))
        self._ycl_unfreeze_epoch = int(os.environ.get("YCL_UNFREEZE_EPOCH", "0"))
        self._ycl_backbone_lr_scale = float(os.environ.get("YCL_BACKBONE_LR_SCALE", "0.5"))
        self._ycl_has_pretrained = True
        self._ycl_frozen = False
        if self._ycl_freeze_layers > 0:
            self.args.freeze = list(range(self._ycl_freeze_layers))
            self._ycl_frozen = True
            _log(f"[ycl-ft] Freeze layers 0-{self._ycl_freeze_layers - 1}, "
                 f"unfreeze at epoch {self._ycl_unfreeze_epoch}")

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=0.0, **kwargs):
        if not getattr(self, "_ycl_has_pretrained", False):
            return super().build_optimizer(model, name, lr, momentum, decay, **kwargs)

        # Önce Ultralytics'in auto optimizer'ını oluştur — doğru LR'yi hesaplasın
        base_opt = super().build_optimizer(model, name, lr, momentum, decay, **kwargs)

        # Auto hesaplanan LR'yi al
        auto_lr = base_opt.param_groups[0]["lr"]
        backbone_lr_scale = self._ycl_backbone_lr_scale
        bb_lr = auto_lr * backbone_lr_scale

        # Parametreleri ayır
        backbone_params = []
        head_params_weight = []
        head_params_bias = []

        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = False
            parts = pname.split(".")
            if len(parts) >= 2 and parts[0] == "model":
                try:
                    layer_idx = int(parts[1])
                    is_backbone = layer_idx < BACKBONE_LAYERS
                except ValueError:
                    pass
            if is_backbone:
                backbone_params.append(param)
            elif "bias" in pname or "bn" in pname:
                head_params_bias.append(param)
            else:
                head_params_weight.append(param)

        if not backbone_params:
            _log("[ycl-ft] No trainable backbone params — using auto optimizer")
            return base_opt

        _log(f"[ycl-ft] Differential LR (from auto_lr={auto_lr:.6f}): "
             f"backbone({len(backbone_params)})={bb_lr:.6f}, "
             f"head({len(head_params_weight) + len(head_params_bias)})={auto_lr:.6f}")

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": bb_lr,
             "initial_lr": bb_lr, "weight_decay": decay},
            {"params": head_params_weight, "lr": auto_lr,
             "initial_lr": auto_lr, "weight_decay": decay},
            {"params": head_params_bias, "lr": auto_lr,
             "initial_lr": auto_lr, "weight_decay": 0.0},
        ])
        return optimizer

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if getattr(self, "_ycl_frozen", False):
            unfreeze_ep = getattr(self, "_ycl_unfreeze_epoch", 0)
            current_ep = getattr(self, "epoch", 0)
            if unfreeze_ep > 0 and current_ep >= unfreeze_ep:
                unfreeze_all(self.model, verbose=True)
                self._ycl_frozen = False
                self.args.freeze = None
                _log(f"[ycl-ft] Backbone unfrozen at epoch {current_ep}")
        return batch
