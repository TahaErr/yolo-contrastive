"""DualTeacherTrainer — DT-SAPS dual-teacher pretraining (Faz 5.3, §10.30).

Composition over inheritance (Karar K5, Risk 19): this trainer does NOT
subclass DenseSSLPretrainer. It holds one as a member (`self.ssl_trainer`),
so the entire SAPS machinery — and its tests — stay untouched. On top of the
SAPS loss it adds a dual-teacher distillation loss.

Per training step:
    1. ssl_trainer._step(imgs)         → SAPS loss; also runs the student
                                          backbone, so online_tap now holds
                                          view1's raw P3/P4/P5 student features.
    2. ssl_trainer.online_tap.get_features()  → student features for distill.
    3. teacher features — from a TeacherCache (offline, §10.27) or computed
       live from a frozen teacher model.
    4. ConsensusLoss(student, t1, t2, disagreement_weight) → distill loss.
    5. total = saps_loss + distill_weight * distill_loss → backward → step
       → ssl_trainer._ema_update().

teacher_combo dispatch:
    none       — distillation off; pure SAPS (the ablation's baseline floor).
    coco_only  — single teacher; ConsensusLoss(student, coco, coco).
    ssl_only   — single teacher; ConsensusLoss(student, ssl, ssl).
    both       — ConsensusLoss(student, coco, ssl).
    For single-teacher combos the two ConsensusLoss teacher slots receive the
    same features: Form B's target collapses to that teacher, Form C becomes
    2x its KL, and the disagreement weight is identically 1 (a teacher never
    disagrees with itself) — so the same code path serves all combos.

Teacher feature source — two modes, both genuinely used:
    cache mode  — a TeacherCache is supplied; features are read from disk.
                  Required for the Faz 5.3 full run (181K imgs x ~60 cells).
    live mode   — no cache; a frozen teacher computes features per batch.
                  Used by smoke tests and unit tests (no cache build needed).
The COCO teacher's trainable adapter (Risk 17) is applied in BOTH modes —
the cache stores raw (un-adapted) teacher features by design.
"""

from __future__ import annotations

import copy
import math
import os
import time
import warnings as _warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..pretrain.dense_trainer import DenseSSLPretrainer
from ..pretrain.dataset import UnlabeledImageDataset
from .consensus_loss import ConsensusLoss
from .disagreement import DisagreementWeighter

_VALID_COMBOS = ("none", "coco_only", "ssl_only", "both")


# ─────────────────────────────────────────────────────────────────────────
# Indexed dataset — wraps UnlabeledImageDataset to also yield an image_id
# ─────────────────────────────────────────────────────────────────────────


class _IndexedImageDataset(Dataset):
    """UnlabeledImageDataset that also yields a stable, path-derived image_id.

    image_id is the dataset-root-relative path without extension — stable
    across runs, so a teacher cache built once can be looked up later.
    """

    def __init__(self, images_dir: str, imgsz: int):
        self.base = UnlabeledImageDataset(images_dir, imgsz=imgsz)
        self.root = Path(images_dir)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        img = self.base[idx]
        path = self.base.files[idx]
        image_id = str(path.relative_to(self.root).with_suffix(""))
        return image_id, img


# ─────────────────────────────────────────────────────────────────────────
# DualTeacherTrainer
# ─────────────────────────────────────────────────────────────────────────


class DualTeacherTrainer:
    """DT-SAPS trainer — SAPS pretraining + dual-teacher distillation.

    Args:
        model: student backbone — Ultralytics spec str or nn.Module.
        teacher_combo: ``none`` | ``coco_only`` | ``ssl_only`` | ``both``.
        coco_teacher: a CocoTeacher (frozen YOLOv8x + trainable adapter).
            Required unless teacher_combo is ``none`` or ``ssl_only``.
        ssl_teacher: a frozen feature teacher for the SSL branch (a CocoTeacher
            built with student_channels=None — the Faz 5.1 SAPS winner).
            Required unless teacher_combo is ``none`` or ``coco_only``.
        coco_cache: optional TeacherCache for the COCO teacher. If given, COCO
            raw features are read from disk instead of computed live.
        ssl_cache: optional TeacherCache for the SSL teacher.
        distill_weight: weight of the distillation loss relative to SAPS.
        distill_form, alpha, beta, w_init, kl_temperature: ConsensusLoss args.
        use_disagreement: enable per-position disagreement weighting.
        disagreement_mode, disagreement_per_scale, disagreement_alpha:
            DisagreementWeighter args.
        ssl_kwargs: dict of DenseSSLPretrainer constructor kwargs (out_dim,
            queue_size, momentum, saps_mode, ...). model/imgsz/device are
            injected automatically.
        imgsz: training image size.
        device: torch device; auto-detected if None.
    """

    def __init__(
        self,
        model,
        *,
        teacher_combo: str = "both",
        coco_teacher=None,
        ssl_teacher=None,
        coco_cache=None,
        ssl_cache=None,
        distill_weight: float = 1.0,
        distill_form: str = "B+C",
        alpha: float = 1.0,
        beta: float = 1.0,
        w_init: float = 0.5,
        kl_temperature: float = 4.0,
        use_disagreement: bool = True,
        disagreement_mode: str = "learnable",
        disagreement_per_scale: bool = True,
        disagreement_alpha: float = 1.0,
        ssl_kwargs: Optional[Dict] = None,
        imgsz: int = 640,
        device=None,
    ):
        if teacher_combo not in _VALID_COMBOS:
            raise ValueError(
                f"teacher_combo must be one of {_VALID_COMBOS}, got {teacher_combo!r}"
            )

        self.teacher_combo = teacher_combo
        self.distill_weight = float(distill_weight)
        self.imgsz = int(imgsz)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── SSL trainer (composition — SAPS machinery untouched) ─────────
        ssl_kwargs = dict(ssl_kwargs or {})
        ssl_kwargs.update(model=model, imgsz=self.imgsz, device=str(self.device))
        self.ssl_trainer = DenseSSLPretrainer(**ssl_kwargs)

        # ── teachers ─────────────────────────────────────────────────────
        needs_coco = teacher_combo in ("coco_only", "both")
        needs_ssl = teacher_combo in ("ssl_only", "both")
        if needs_coco and coco_teacher is None:
            raise ValueError(f"teacher_combo={teacher_combo!r} requires coco_teacher")
        if needs_ssl and ssl_teacher is None:
            raise ValueError(f"teacher_combo={teacher_combo!r} requires ssl_teacher")

        self.coco_teacher = coco_teacher
        self.ssl_teacher = ssl_teacher
        self.coco_cache = coco_cache
        self.ssl_cache = ssl_cache

        # ── distillation modules ─────────────────────────────────────────
        levels = tuple(self.ssl_trainer._in_channels.keys())
        self.consensus_loss = ConsensusLoss(
            levels=levels, distill_form=distill_form,
            alpha=alpha, beta=beta, w_init=w_init,
            kl_temperature=kl_temperature,
        ).to(self.device)

        self.use_disagreement = bool(use_disagreement)
        self.disagreement: Optional[DisagreementWeighter] = None
        if self.use_disagreement:
            self.disagreement = DisagreementWeighter(
                levels=levels, mode=disagreement_mode,
                per_scale=disagreement_per_scale,
                init_alpha=disagreement_alpha,
            ).to(self.device)

    # ── trainable parameter collection ───────────────────────────────────

    def _trainable_parameters(self) -> List[nn.Parameter]:
        """All parameters the optimizer updates.

        Student backbone + projection head + (COCO adapter) + ConsensusLoss
        fusion weight + (learnable disagreement alpha). Teacher backbones and
        the momentum/EMA copies are excluded — they are not optimized.
        """
        params: List[nn.Parameter] = []
        params += list(self.ssl_trainer.model.parameters())
        params += list(self.ssl_trainer.proj_online.parameters())
        if self.coco_teacher is not None and self.coco_teacher.adapter is not None:
            params += list(self.coco_teacher.adapter.parameters())
        params += list(self.consensus_loss.parameters())
        if self.disagreement is not None:
            params += [p for p in self.disagreement.parameters() if p.requires_grad]
        # De-dup while preserving order (a param could appear twice in theory).
        seen = set()
        unique = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    # ── teacher feature retrieval ────────────────────────────────────────

    def _teacher_features_from(
        self,
        teacher,
        cache,
        image_ids: List[str],
        imgs: torch.Tensor,
        apply_adapter: bool,
    ) -> Dict[str, torch.Tensor]:
        """Raw-or-cached teacher features, optionally adapter-mapped.

        cache given  → load per-image raw features from disk, stack.
        cache None   → run the frozen teacher live on `imgs`.
        apply_adapter → run the COCO teacher's trainable adapter (Risk 17).
        """
        if cache is not None:
            per_image = [cache.load(iid) for iid in image_ids]
            raw = {
                lv: torch.stack([d[lv] for d in per_image], dim=0).to(self.device)
                for lv in per_image[0]
            }
        else:
            raw = teacher.extract_features(imgs)

        if apply_adapter:
            return teacher.adapt(raw)
        return {lv: t.to(self.device) for lv, t in raw.items()}

    def _get_distill_inputs(
        self, image_ids: List[str], imgs: torch.Tensor,
    ) -> Tuple[Dict, Dict]:
        """Return the two teacher feature dicts for ConsensusLoss.

        For single-teacher combos both slots hold the same features.
        """
        if self.teacher_combo == "coco_only":
            coco = self._teacher_features_from(
                self.coco_teacher, self.coco_cache, image_ids, imgs,
                apply_adapter=True,
            )
            return coco, coco
        if self.teacher_combo == "ssl_only":
            ssl = self._teacher_features_from(
                self.ssl_teacher, self.ssl_cache, image_ids, imgs,
                apply_adapter=False,
            )
            return ssl, ssl
        # both
        coco = self._teacher_features_from(
            self.coco_teacher, self.coco_cache, image_ids, imgs,
            apply_adapter=True,
        )
        ssl = self._teacher_features_from(
            self.ssl_teacher, self.ssl_cache, image_ids, imgs,
            apply_adapter=False,
        )
        return coco, ssl

    # ── single training step ─────────────────────────────────────────────

    def _step(self, image_ids: List[str], imgs: torch.Tensor) -> Dict:
        """One DT-SAPS step. Returns dict with total loss + components."""
        # 1) SAPS step — also populates ssl_trainer.online_tap with view1's
        #    raw student features.
        saps_out = self.ssl_trainer._step(imgs)
        saps_loss = saps_out["loss"]

        info = {"saps": float(saps_loss.detach()), "distill": 0.0}

        if self.teacher_combo == "none":
            info["total"] = float(saps_loss.detach())
            return {"loss": saps_loss, "info": info, "batch_size": imgs.shape[0]}

        # 2) Student features (raw backbone output, projection-free).
        student = self.ssl_trainer.online_tap.get_features()

        # 3) Teacher features.
        t1, t2 = self._get_distill_inputs(image_ids, imgs)

        # 4) Disagreement weight (identically 1 for single-teacher combos).
        dw = None
        if self.disagreement is not None:
            dw = self.disagreement(t1, t2)

        # 5) Distillation loss.
        distill_loss, distill_info = self.consensus_loss(student, t1, t2, dw)
        total = saps_loss + self.distill_weight * distill_loss

        info["distill"] = float(distill_loss.detach())
        info["distill_detail"] = distill_info
        info["total"] = float(total.detach())
        if self.disagreement is not None:
            info["alpha_d"] = self.disagreement.get_alpha()
        return {"loss": total, "info": info, "batch_size": imgs.shape[0]}

    # ── training loop ────────────────────────────────────────────────────

    def train(
        self,
        images_dir: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        num_workers: int = 4,
        output: str = "dt_saps_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
        resume_from: Optional[str] = None,
    ) -> str:
        """Run DT-SAPS pretraining. Returns the saved backbone path.

        resume_from: path to a ``.resume.pt`` state file (written every
            ``save_every`` epochs). If given and present, training resumes
            from the next epoch. The DT-SAPS resume state carries the full
            SAPS machinery via the held ssl_trainer (student backbone,
            online/momentum projectors, momentum encoder, P3/P4/P5 queues)
            plus the distillation-side trainable modules (COCO adapter,
            ConsensusLoss fusion weight, DisagreementWeighter alpha) and
            the optimizer. Frozen teacher backbones are NOT in the state —
            they are rebuilt from the constructor. Epoch-boundary granularity.
        """
        dataset = _IndexedImageDataset(images_dir, imgsz=self.imgsz)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        steps_per_epoch = len(dataloader)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = warmup_epochs * steps_per_epoch

        optimizer = torch.optim.AdamW(
            self._trainable_parameters(), lr=lr, weight_decay=weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ── resume state (optional) ──────────────────────────────────────
        # The SAPS machinery is held inside ssl_trainer; the distillation
        # side (adapter, consensus, disagreement) is local. Both are saved.
        resume_path = output.replace(".pt", ".resume.pt")
        start_epoch = 1
        global_step = 0
        loss_history: list = []
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        if resume_from is not None and os.path.exists(resume_from):
            rs = torch.load(resume_from, map_location=self.device,
                            weights_only=False)
            # SAPS machinery (via ssl_trainer)
            self.ssl_trainer.model.load_state_dict(rs["model"])
            self.ssl_trainer.proj_online.load_state_dict(rs["proj_online"])
            self.ssl_trainer.momentum.momentum.load_state_dict(
                rs["momentum_encoder"])
            self.ssl_trainer.proj_momentum.load_state_dict(rs["proj_momentum"])
            for lv, q in self.ssl_trainer.queues.items():
                if lv in rs["queues"]:
                    q.load_state_dict(rs["queues"][lv])
            # distillation-side trainable modules
            self.consensus_loss.load_state_dict(rs["consensus_loss"])
            if self.coco_teacher is not None and \
                    self.coco_teacher.adapter is not None and \
                    rs.get("coco_adapter") is not None:
                self.coco_teacher.adapter.load_state_dict(rs["coco_adapter"])
            if self.disagreement is not None and \
                    rs.get("disagreement") is not None:
                self.disagreement.load_state_dict(rs["disagreement"])
            optimizer.load_state_dict(rs["optimizer"])
            start_epoch = int(rs["epoch"]) + 1
            global_step = int(rs["global_step"])
            loss_history = list(rs.get("loss_history", []))
            _b = rs.get("best", {})
            best_loss = _b.get("loss", float("inf"))
            best_epoch = _b.get("epoch", 0)
            best_state = _b.get("state", None)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                for _ in range(global_step):
                    scheduler.step()
            print(f"[dt-saps] RESUMED from epoch {rs['epoch']} "
                  f"→ continuing at epoch {start_epoch}/{epochs}")

        self.ssl_trainer.model.train()
        self.ssl_trainer.proj_online.train()

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            ep_loss = ep_saps = ep_distill = 0.0
            n_batches = 0

            for image_ids, imgs in dataloader:
                optimizer.zero_grad(set_to_none=True)
                out = self._step(list(image_ids), imgs)
                loss = out["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.ssl_trainer._ema_update()

                ep_loss += float(loss.detach())
                ep_saps += out["info"]["saps"]
                ep_distill += out["info"]["distill"]
                n_batches += 1
                global_step += 1

            n_batches = max(1, n_batches)
            avg_loss = ep_loss / n_batches
            loss_history.append({
                "epoch": epoch, "loss": avg_loss,
                "saps": ep_saps / n_batches,
                "distill": ep_distill / n_batches,
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
            if print_every and (epoch % print_every == 0 or epoch == 1):
                dt = time.time() - t0
                print(
                    f"[dt-saps] epoch {epoch}/{epochs} "
                    f"loss={avg_loss:.4f} "
                    f"saps={ep_saps / n_batches:.4f} "
                    f"distill={ep_distill / n_batches:.4f} "
                    f"({dt:.1f}s)"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.ssl_trainer.model.state_dict())

            if save_every and epoch % save_every == 0:
                # resume state FIRST — survives a failure in _save() below
                _state = {
                    "model": self.ssl_trainer.model.state_dict(),
                    "proj_online": self.ssl_trainer.proj_online.state_dict(),
                    "momentum_encoder":
                        self.ssl_trainer.momentum.momentum.state_dict(),
                    "proj_momentum":
                        self.ssl_trainer.proj_momentum.state_dict(),
                    "queues": {lv: q.state_dict()
                               for lv, q in self.ssl_trainer.queues.items()},
                    "consensus_loss": self.consensus_loss.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                    "loss_history": loss_history,
                    "best": {"loss": best_loss, "epoch": best_epoch,
                             "state": best_state},
                }
                if self.coco_teacher is not None and \
                        self.coco_teacher.adapter is not None:
                    _state["coco_adapter"] = \
                        self.coco_teacher.adapter.state_dict()
                if self.disagreement is not None:
                    _state["disagreement"] = self.disagreement.state_dict()
                torch.save(_state, resume_path)
                self._save(output.replace(".pt", f"_ep{epoch}.pt"), epoch)

        # final save — best-epoch student weights, full loss_history
        if best_state is not None:
            self.ssl_trainer.model.load_state_dict(best_state)
        self._save(output, best_epoch or epochs,
                   loss_history=loss_history, best_epoch=best_epoch or epochs)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        return output

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save(self, output: str, epoch: int,
              loss_history: Optional[list] = None,
              best_epoch: Optional[int] = None) -> None:
        """Save the student backbone with a DT-SAPS marker.

        loss_history / best_epoch are passed only by the final save; when
        present they go into ``extra`` so the learning curve survives.
        """
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        extra = {
            "type": "dt_saps",
            "teacher_combo": self.teacher_combo,
            "distill_form": self.consensus_loss.distill_form,
            "w_coco": self.consensus_loss.get_w(),
        }
        if self.disagreement is not None:
            extra["alpha_d"] = self.disagreement.get_alpha()
        if loss_history is not None:
            extra["loss_history"] = loss_history
        if best_epoch is not None:
            extra["best_epoch"] = best_epoch
        ckpt = {
            "model_state_dict": self.ssl_trainer.model.state_dict(),
            "epoch": epoch,
            "type": "ssl_pretrained",
            "extra": extra,
        }
        torch.save(ckpt, output)

    # ── lifecycle ────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        try:
            self.ssl_trainer.cleanup()
        except Exception:
            pass
        for teacher in (self.coco_teacher, self.ssl_teacher):
            if teacher is not None:
                try:
                    teacher.cleanup()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"DualTeacherTrainer(teacher_combo={self.teacher_combo}, "
            f"distill_form={self.consensus_loss.distill_form}, "
            f"distill_weight={self.distill_weight})"
        )
