"""ContrastiveDetectionTrainer — CL + rotation pretext + pluggable augmentation."""

from __future__ import annotations

import os
from typing import Tuple

import torch

from ..feature_tap import FeatureTap
from ..contrastive.losses import build_contrastive_loss
from .._config import CLConfig
from ._helpers import log, safe_scalar
from ._augmentation import AugmentationMixin
from ._patching import PatchingMixin
from ._csv_logger import CSVLoggerMixin

# Robust DetectionTrainer import
DetectionTrainer = None
_import_errs: list = []
try:
    from ultralytics.models.yolo.detect.train import DetectionTrainer as _DT
    DetectionTrainer = _DT
except Exception as e:
    _import_errs.append(("ultralytics.models.yolo.detect.train", repr(e)))
if DetectionTrainer is None:
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer as _DT
        DetectionTrainer = _DT
    except Exception as e:
        _import_errs.append(("ultralytics.models.yolo.detect", repr(e)))
if DetectionTrainer is None:
    raise ImportError(
        "Could not import DetectionTrainer from ultralytics. Tried:\n"
        + "\n".join([f"- {p}: {err}" for p, err in _import_errs])
    )

try:
    from ultralytics.utils.torch_utils import unwrap_model as _unwrap_model
except Exception:
    def _unwrap_model(m):
        return m.module if hasattr(m, "module") else m


def _add_params_to_optimizer(optimizer, params, scheduler=None, log_prefix=""):
    """Add param group with initial_lr (Ultralytics LR scheduler compat)."""
    param_list = list(params)
    if not param_list:
        return
    lr = optimizer.defaults.get("lr", 0.01)
    optimizer.add_param_group({
        "params": param_list,
        "lr": lr,
        "initial_lr": lr,
    })
    # Scheduler'ın base_lrs ve lr_lambdas listelerini güncelle — yoksa zip() hata verir
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs.append(lr)
        # LambdaLR ise lr_lambdas da senkron olmalı
        if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
            scheduler.lr_lambdas.append(scheduler.lr_lambdas[-1])
    log(f"{log_prefix}Added {len(param_list)} params to optimizer (lr={lr}).")


class ContrastiveDetectionTrainer(
    AugmentationMixin, PatchingMixin, CSVLoggerMixin, DetectionTrainer,
):
    """YOLO trainer + contrastive loss + rotation pretext task.

    Loss:  total = det_loss + λ_cl × cl_loss + λ_rot × rot_loss
    """

    # ── step dedup ──

    def _trainer_step_id(self) -> Tuple[str, int, int]:
        ni = getattr(self, "ni", None)
        if isinstance(ni, int):
            return ("ni", ni, -1)
        it = getattr(self, "iter", None)
        if isinstance(it, int):
            return ("iter", it, -1)
        ep = getattr(self, "epoch", None)
        bi = getattr(self, "batch_i", None)
        return ("epoch_batch", int(ep) if ep is not None else -1, int(bi) if bi is not None else -1)

    def _batch_fingerprint(self, batch):
        if not isinstance(batch, dict):
            return None
        imf = batch.get("im_file")
        if isinstance(imf, (list, tuple)) and imf:
            return ("im_file", tuple(str(x) for x in imf))
        img = batch.get("img")
        if torch.is_tensor(img):
            try:
                return ("img_ptr", (str(int(img.data_ptr())), str(tuple(img.shape))))
            except Exception:
                return ("img_id", (str(id(img)), str(tuple(img.shape))))
        return None

    def _touch_step_key(self, batch):
        key = (self._trainer_step_id(), self._batch_fingerprint(batch))
        if getattr(self, "_cl_last_key", None) != key:
            self._cl_last_key = key
            self._cl_added_for_key = False

    def _can_add_cl_now(self, batch=None):
        if batch is not None:
            self._touch_step_key(batch)
        return not getattr(self, "_cl_added_for_key", False)

    def _mark_cl_added(self):
        self._cl_added_for_key = True

    # ── init ──

    def _ensure_cl(self, batch: dict) -> None:
        if getattr(self, "_cl_inited", False):
            return

        self.cl_cfg = CLConfig.from_env()
        cfg = self.cl_cfg

        # FIX: CL ve rotation ikisi de kapalıysa hook kurma — gereksiz maliyet
        if not cfg.enabled and not cfg.rotation_enabled:
            self._cl_inited = True
            self._rot_params_pending = False
            self._rot_task = None
            self._feature_tap = None
            self._cl_loss_fn = None
            self._cl_aug_pipeline = None
            self._cl_step = 0
            self._cl_last_key = None
            self._cl_added_for_key = False
            self._cl_grad_warned = False
            self._cl_bn_note_printed = False
            log("[ycl] Init: CL and rotation both disabled (lambda_cl=0, lambda_rot=0). "
                "No hooks installed.")
            return

        self._cl_loss_fn = build_contrastive_loss(cfg.loss_name, temperature=cfg.temperature)

        # Augmentation pipeline
        self._cl_aug_pipeline = None
        if cfg.two_view and cfg.aug_preset:
            try:
                from ..augmentations.presets import build_pipeline
                self._cl_aug_pipeline = build_pipeline(cfg.aug_preset)
                log(f"[ycl] Aug pipeline: {cfg.aug_preset} → {self._cl_aug_pipeline}")
            except Exception as e:
                log(f"[ycl] WARN: preset '{cfg.aug_preset}' failed: {e}. Using legacy.")

        # Feature tap
        self._cl_img_normalized = True
        self._cl_scale_warned = False
        imgsz = int(batch["img"].shape[-1])
        device = batch["img"].device
        base_model = _unwrap_model(self.model)
        self._feature_tap = FeatureTap(
            base_model, min_channels=128,
            store_grad=(cfg.enabled or cfg.rotation_enabled),
        )
        self._feature_tap.setup(device=device, imgsz=imgsz)

        # Rotation pretext
        self._rot_task = None
        self._rot_params_pending = False
        if cfg.rotation_enabled:
            feat_dim = self._get_feat_dim(batch)
            from ..pretext.rotation import RotationTask
            self._rot_task = RotationTask(feat_dim=feat_dim, hidden_dim=cfg.rot_hidden_dim).to(device)
            self._rot_params_pending = True
            log(f"[ycl] Rotation pretext: λ_rot={cfg.lambda_rot}, feat_dim={feat_dim}")

        # State
        self._cl_step = 0
        self._cl_last_key = None
        self._cl_added_for_key = False
        self._touch_step_key(batch)
        self._cl_grad_warned = False
        self._cl_bn_note_printed = False

        # CSV
        save_dir = getattr(self, "save_dir", None) or os.getcwd()
        self._cl_csv_path = os.path.join(str(save_dir), "cl_losses.csv")
        self._csv_init_if_needed()

        self._cl_inited = True

        log(
            f"[ycl] Init: cl={cfg.enabled}(λ={cfg.lambda_cl}) "
            f"rot={cfg.rotation_enabled}(λ={cfg.lambda_rot}) "
            f"loss={cfg.loss_name} temp={cfg.temperature} "
            f"tap={self._feature_tap.layer_name} "
            f"2v={cfg.two_view} preset={cfg.aug_preset or 'legacy'}"
        )
        if cfg.enabled and not cfg.two_view and cfg.pseudo_view:
            log("[ycl] NOTE: pseudo view. Set YCL_TWO_VIEW=1 for real augmentation.")

        self._install_model_patches()

    def _get_feat_dim(self, batch) -> int:
        base_model = _unwrap_model(self.model)
        was_training = base_model.training
        base_model.eval()
        with torch.no_grad():
            _ = base_model(batch["img"][:1])
        if was_training:
            base_model.train()
        emb = self._feature_tap.get_embedding()
        if emb is None:
            raise RuntimeError("[ycl] Could not determine feat_dim.")
        return emb.shape[1]

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._ensure_cl(batch)
        self._touch_step_key(batch)
        self._cl_last_img = batch.get("img", None)

        if self._rot_params_pending and self._rot_task is not None:
            opt = getattr(self, "optimizer", None)
            if opt is not None:
                sched = getattr(self, "scheduler", None)
                _add_params_to_optimizer(opt, self._rot_task.parameters(), scheduler=sched, log_prefix="[ycl] ")
                self._rot_params_pending = False

        return batch

    # ── CL ──

    def _compute_cl(self, z1, z2=None):
        cfg = self.cl_cfg
        if not cfg.enabled or z1 is None:
            return None
        if z2 is None and cfg.pseudo_view:
            z2 = z1 + cfg.noise_std * torch.randn_like(z1)
        elif z2 is None:
            z2 = z1
        return self._cl_loss_fn(z1, z2)

    # ── Rotation ──

    def _compute_rotation(self, img, model_self, orig_forward):
        """Rotate → forward (BN preserved) → predict → CE loss.

        Returns:
            Tuple[Optional[Tensor], float]: (rot_loss, rot_accuracy)
            Always returns a tuple — never bare None.
        """
        from ._helpers import preserve_bn_running_stats

        if self._rot_task is None:
            return None, 0.0
        if not self.cl_cfg.rotation_enabled:
            return None, 0.0
        if not torch.is_tensor(img):
            return None, 0.0

        rotated_img, rot_labels = self._rot_task.rotate_batch(img)

        with preserve_bn_running_stats(model_self):
            _ = orig_forward(rotated_img)

        rot_features = self._feature_tap.get_embedding()
        if rot_features is None:
            return None, 0.0

        rot_loss, rot_acc = self._rot_task(rot_features, rot_labels)
        return rot_loss, rot_acc

    # ── Recording ──

    def _record_step(self, det_loss, cl_loss, rot_loss, total, rot_acc, source):
        self._cl_step += 1
        cfg = self.cl_cfg
        det_f = safe_scalar(det_loss)
        cl_f = safe_scalar(cl_loss)
        rot_f = safe_scalar(rot_loss)
        tot_f = safe_scalar(total)

        if self._cl_step == 1 or (cfg.print_every > 0 and self._cl_step % cfg.print_every == 0):
            parts = [f"step={self._cl_step}", f"det={det_f:.4f}"]
            if cfg.enabled:
                parts.append(f"cl={cl_f:.4f}(λ={cfg.lambda_cl:.3f})")
            if cfg.rotation_enabled:
                parts.append(f"rot={rot_f:.4f}(λ={cfg.lambda_rot:.3f},acc={rot_acc:.1%})")
            parts.append(f"total={tot_f:.4f}")
            log(f"[ycl] {' '.join(parts)}")

        tag, a, b = self._trainer_step_id()
        ep = int(getattr(self, "epoch", -1)) if getattr(self, "epoch", None) is not None else -1
        self._csv_append([
            self._cl_step, ep, tag, a, b, source,
            det_f, cl_f, rot_f, tot_f,
            cfg.lambda_cl, cfg.lambda_rot, cfg.temperature,
            self._feature_tap.layer_name,
            int(cfg.two_view), cfg.aug_preset or "legacy",
            f"{rot_acc:.4f}" if cfg.rotation_enabled else "",
        ])

    def cleanup(self):
        self._uninstall_model_patches()
        if hasattr(self, "_feature_tap"):
            self._feature_tap.close()
