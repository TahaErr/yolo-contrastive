"""Model patching: forward/loss → inject CL + pretext tasks."""

from __future__ import annotations

import types
import weakref

import torch

from ._helpers import extract_loss_from_out, is_main_process, log, preserve_bn_running_stats, replace_in_output


class PatchingMixin:

    def _install_model_patches(self) -> None:
        if getattr(self, "_cl_patched", False):
            return
        try:
            from ultralytics.utils.torch_utils import unwrap_model
        except Exception:
            def unwrap_model(m): return m.module if hasattr(m, "module") else m
        base_model = unwrap_model(self.model)
        self._patch_forward_for_loss_return(base_model)
        self._patch_loss_for_compile(base_model)
        self._cl_patched = True

    def _uninstall_model_patches(self) -> None:
        if not getattr(self, "_cl_patched", False):
            return
        try:
            from ultralytics.utils.torch_utils import unwrap_model
        except Exception:
            def unwrap_model(m): return m.module if hasattr(m, "module") else m
        base_model = unwrap_model(self.model)
        if hasattr(self, "_cl_orig_forward"):
            base_model.forward = self._cl_orig_forward
            del self._cl_orig_forward
            self._cl_forward_patched = False
        if hasattr(self, "_cl_orig_loss"):
            base_model.loss = self._cl_orig_loss
            del self._cl_orig_loss
            self._cl_loss_patched = False
        self._cl_patched = False
        log("[ycl] Model patches removed.")

    def _inject_all(self, model_self, out, orig_forward, batch=None, source="forward"):
        """Shared: det → CL → pretext → total = det + λ_cl*cl + λ_pretext*pretext."""
        cfg = getattr(self, "cl_cfg", None)
        if cfg is None:
            return out

        # Herhangi bir loss aktif mi?
        any_active = cfg.enabled or cfg.pretext_enabled or cfg.rotation_enabled
        if not any_active:
            return out

        # Çift enjeksiyon koruması
        if getattr(self, "_cl_added_for_key", False):
            return out

        det_loss, idx = extract_loss_from_out(out)
        if det_loss is None or not self._can_add_cl_now(batch=batch):
            return out

        z1 = self._feature_tap.get_embedding()
        if z1 is None:
            return out

        # Grad warning (once)
        if is_main_process(self) and not getattr(self, "_cl_grad_warned", False):
            if not bool(getattr(z1, "requires_grad", False)):
                log("[ycl] WARN: z1.requires_grad=False → gradient will not affect backbone.")
                self._cl_grad_warned = True

        # ── Contrastive ──
        cl_loss = None
        if cfg.enabled:
            z2 = None
            if cfg.two_view:
                img = getattr(self, "_cl_last_img", None)
                if img is None and isinstance(batch, dict):
                    img = batch.get("img")
                if torch.is_tensor(img):
                    img2 = self.make_view2(img)
                    if is_main_process(self) and not getattr(self, "_cl_bn_note_printed", False):
                        log("[ycl] NOTE: Preserving BN stats for view2.")
                        self._cl_bn_note_printed = True
                    with preserve_bn_running_stats(model_self):
                        _ = orig_forward(img2)
                    z2 = self._feature_tap.get_embedding()
            cl_loss = self._compute_cl(z1, z2=z2)

        # ── Pretext (composite veya legacy rotation) ──
        pretext_loss = None
        pretext_acc = 0.0
        pretext_detail = ""
        if cfg.pretext_enabled or cfg.rotation_enabled:
            img = getattr(self, "_cl_last_img", None)
            if img is None and isinstance(batch, dict):
                img = batch.get("img")
            if torch.is_tensor(img):
                pretext_loss, pretext_acc, pretext_detail = self._compute_pretext(
                    img, model_self, orig_forward
                )

        # ── Combine ──
        if cl_loss is None and pretext_loss is None:
            return out

        self._mark_cl_added()
        det_s = det_loss if det_loss.numel() == 1 else det_loss.mean()
        total = det_s.clone()

        if cl_loss is not None:
            total = total + cfg.lambda_cl * cl_loss

        if pretext_loss is not None:
            lambda_pt = cfg.lambda_pretext if cfg.pretext_enabled else cfg.lambda_rot
            total = total + lambda_pt * pretext_loss

        self._record_step(det_s, cl_loss, pretext_loss, total, pretext_acc,
                          source=source, pretext_detail=pretext_detail)
        return replace_in_output(out, idx, total)

    def _patch_forward_for_loss_return(self, base_model) -> None:
        if getattr(self, "_cl_forward_patched", False):
            return
        if not hasattr(base_model, "forward"):
            self._cl_forward_patched = True
            return
        orig_forward = base_model.forward
        self._cl_orig_forward = orig_forward
        self._cl_base_forward = orig_forward
        trainer_ref = weakref.ref(self)

        def forward_patched(model_self, *args, **kwargs):
            out = orig_forward(*args, **kwargs)
            trainer = trainer_ref()
            if trainer is None:
                return out
            return trainer._inject_all(model_self, out, orig_forward, source="forward")

        base_model.forward = types.MethodType(forward_patched, base_model)
        self._cl_forward_patched = True
        log("[ycl] Patched base_model.forward.")

    def _patch_loss_for_compile(self, base_model) -> None:
        if getattr(self, "_cl_loss_patched", False):
            return
        if not hasattr(base_model, "loss") or not callable(getattr(base_model, "loss")):
            self._cl_loss_patched = True
            return
        orig_loss = base_model.loss
        self._cl_orig_loss = orig_loss
        trainer_ref = weakref.ref(self)

        def loss_patched(model_self, batch, preds=None, *args, **kwargs):
            out = orig_loss(batch, preds, *args, **kwargs)
            trainer = trainer_ref()
            if trainer is None:
                return out
            if getattr(trainer, "_cl_added_for_key", False):
                return out
            fwd = getattr(trainer, "_cl_base_forward", None)
            if fwd is None:
                fwd = orig_loss
            return trainer._inject_all(model_self, out, fwd, batch=batch, source="loss")

        base_model.loss = types.MethodType(loss_patched, base_model)
        self._cl_loss_patched = True
        log("[ycl] Patched base_model.loss().")
