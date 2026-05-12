"""FinetuneDetectionTrainer — pretrained backbone ile az etiketli fine-tuning."""

from __future__ import annotations

import os

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

        # Önce Ultralytics'in optimizer'ını oluştur — doğru LR'yi hesaplasın
        base_opt = super().build_optimizer(model, name, lr, momentum, decay, **kwargs)

        # Auto hesaplanan LR'yi al
        auto_lr = base_opt.param_groups[0]["lr"]
        backbone_lr_scale = self._ycl_backbone_lr_scale
        bb_lr = auto_lr * backbone_lr_scale

        # Parametreleri ayır
        backbone_params = []
        head_params = []

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
            else:
                head_params.append(param)

        if not backbone_params:
            _log("[ycl-ft] No trainable backbone params — using base optimizer")
            return base_opt

        _log(f"[ycl-ft] Differential LR (from auto_lr={auto_lr:.6f}): "
             f"backbone({len(backbone_params)})={bb_lr:.6f}, "
             f"head({len(head_params)})={auto_lr:.6f}")

        # FIX: Kullanıcının seçtiği optimizer türüne saygı duy
        # base_opt'ın türünü kullanarak aynı tip optimizer oluştur
        opt_cls = type(base_opt)

        # base_opt'tan defaults'ı al (momentum, betas vb.)
        defaults = {k: v for k, v in base_opt.defaults.items()
                    if k not in ("lr", "initial_lr")}

        try:
            optimizer = opt_cls([
                {"params": backbone_params, "lr": bb_lr,
                 "initial_lr": bb_lr, **defaults},
                {"params": head_params, "lr": auto_lr,
                 "initial_lr": auto_lr, **defaults},
            ])
        except Exception as e:
            _log(f"[ycl-ft] WARN: Could not create {opt_cls.__name__} with "
                 f"differential LR: {e}. Falling back to base optimizer.")
            return base_opt

        _log(f"[ycl-ft] Optimizer: {opt_cls.__name__} with differential LR")
        return optimizer

    def _safe_ema_sync(self) -> None:
        """Sync EMA weights to current model state, robust to InferenceMode taint.

        Background (Risk 16, WORK_PLAN §10.22 + §10.23 + §10.25):
            Ultralytics' internal validator runs forward passes under
            ``torch.inference_mode()``. After such a pass, parameters or
            buffers on the EMA copy can carry the InferenceMode flag. A
            subsequent plain ``load_state_dict(state)`` then tries to
            update those tensors in-place and raises::

                RuntimeError: Inplace update to inference tensor outside
                              InferenceMode is not allowed.

        Strategy (v2): rebuild any InferenceMode-tainted parameters and
        buffers on the destination as detached clones (which carry no
        inference flag), then call **plain** ``load_state_dict`` (no
        ``assign=True``). This is critical: ``assign=True`` replaces
        destination tensors by reference, aliasing EMA storage to the
        model. Ultralytics' ``ema.update()`` then evaluates
        ``v *= d; v += (1-d) * msd[k]`` where ``v`` and ``msd[k]`` are the
        same tensor, collapsing weights by ~2d per step (~99.9% per step
        at typical early-training decay), zeroing the model within an
        epoch. See §10.25 for the forensic trail; v1 (assign=True) failed
        production validation despite passing isolated unit tests.

        The cleanup step touches only tainted tensors, so this is a
        no-op when called on a clean module (e.g. during ``_setup_train``
        before any validator has run).
        """
        import torch.nn as _nn

        # 1) Rebuild tainted params on destination as clean clones
        for name, param in list(self.ema.ema.named_parameters()):
            if param.is_inference():
                *path, leaf = name.split(".")
                parent = self.ema.ema
                for p in path:
                    parent = getattr(parent, p)
                parent._parameters[leaf] = _nn.Parameter(
                    param.detach().clone(),
                    requires_grad=param.requires_grad,
                )

        # 2) Rebuild tainted buffers on destination as clean clones
        for name, buf in list(self.ema.ema.named_buffers()):
            if buf.is_inference():
                *path, leaf = name.split(".")
                parent = self.ema.ema
                for p in path:
                    parent = getattr(parent, p)
                parent._buffers[leaf] = buf.detach().clone()

        # 3) Plain load — copies VALUES into destination's existing tensors,
        # preserving the independent-storage invariant that EMA requires.
        self.ema.ema.load_state_dict(self.model.state_dict())

    def _setup_train(self, world_size=1):
        """EMA'yı SSL backbone ile senkronize et."""
        super()._setup_train()
        if getattr(self, "_ycl_has_pretrained", False) and hasattr(self, "ema") and self.ema is not None:
            self._safe_ema_sync()  # Risk 16 fix — WORK_PLAN §10.23
            self.ema.updates = 0
            _log("[ycl-ft] EMA synced with SSL backbone")

    def save_model(self):
        """EMA'yı kaydetmeden önce gerçek modelle senkronize et."""
        if hasattr(self, 'ema') and self.ema is not None:
            self._safe_ema_sync()  # Risk 16 fix — WORK_PLAN §10.23
        super().save_model()

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
