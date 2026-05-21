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

import math
import time
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
    ) -> str:
        """Run DT-SAPS pretraining. Returns the saved backbone path."""
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

        self.ssl_trainer.model.train()
        self.ssl_trainer.proj_online.train()

        global_step = 0
        for epoch in range(1, epochs + 1):
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
            if print_every and (epoch % print_every == 0 or epoch == 1):
                dt = time.time() - t0
                print(
                    f"[dt-saps] epoch {epoch}/{epochs} "
                    f"loss={ep_loss / n_batches:.4f} "
                    f"saps={ep_saps / n_batches:.4f} "
                    f"distill={ep_distill / n_batches:.4f} "
                    f"({dt:.1f}s)"
                )

            if save_every and epoch % save_every == 0:
                self._save(output, epoch)

        self._save(output, epochs)
        return output

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save(self, output: str, epoch: int) -> None:
        """Save the student backbone with a DT-SAPS marker."""
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state_dict": self.ssl_trainer.model.state_dict(),
            "epoch": epoch,
            "type": "ssl_pretrained",
            "extra": {
                "type": "dt_saps",
                "teacher_combo": self.teacher_combo,
                "distill_form": self.consensus_loss.distill_form,
                "w_coco": self.consensus_loss.get_w(),
            },
        }
        if self.disagreement is not None:
            ckpt["extra"]["alpha_d"] = self.disagreement.get_alpha()
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
