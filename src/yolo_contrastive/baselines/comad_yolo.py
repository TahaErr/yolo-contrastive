"""CoMAD-YOLO — external SSL baseline (Faz 5.4, WORK_PLAN_v9 §5.4).

A fair-comparison baseline: CoMAD's core mechanism on a YOLOv8n backbone,
same pool and protocol as DT-SAPS (paper Table 4 "DT-SAPS vs SOTA").

CoMAD (Consensus-oriented Masked Distillation, AAAI 2026) distils three
diverse self-supervised teachers into one compact student. Its defining
mechanisms, ported to our CNN/detection setting:

  - Three SSL teachers. CoMAD uses MAE + MoCo-v3 + iBOT (diverse ViT SSL).
    Our analogue: three diverse SSL backbones produced by this repo —
    SimCLR-YOLO, MoCo-v3-YOLO, and a dense-SAPS backbone — each pretrained
    on the same pool. All frozen.

  - Asymmetric masking. The student sees a heavily masked input; each
    teacher sees a progressively lighter, distinct mask, so the student
    must interpolate missing features under richer teacher context. On a
    CNN we mask by zeroing input image patches.

  - Joint consensus gating (CoMAD's signature, parameter-free). Teacher
    features are fused NOT by naive averaging but by a per-position gate
    that combines each teacher's cosine affinity to the student with the
    inter-teacher agreement. Consensus regions get more weight.

  - Feature distillation via channel-wise KL (CWD-style).

This is the paper's key comparison: CoMAD's consensus gating *filters*
disagreement with a fixed scheme, whereas our DT-SAPS leaves the direction
to an ablatable signed alpha_d (§10.29, §10.31). CoMAD-YOLO is deliberately
single-scale (ViT-like) — multi-scale is one of our method's own axes and
is not given to the baseline.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..augmentations.presets import build_pipeline
from ..dense import MultiScaleFeatureTap, infer_in_channels
from ..pretrain.backbone_utils import load_backbone
from ..pretrain.dataset import UnlabeledImageDataset

_VALID_LEVELS = ("P3", "P4", "P5")


class CoMADYOLOTrainer:
    """CoMAD-style multi-teacher consensus distillation on a YOLOv8n backbone.

    Args:
        model: student backbone — Ultralytics spec str or nn.Module.
        teachers: list of frozen SSL teachers — each an SSL checkpoint path
            (loaded into a YOLOv8n skeleton) or an nn.Module. At least 2.
        feat_level: FPN level distilled. Default ``P5`` (single-scale —
            multi-scale is reserved as one of DT-SAPS's own axes).
        mask_ratio_student: input patch-mask ratio for the student (high).
        mask_ratio_teachers: per-teacher mask ratios (light, progressively
            increasing); length must match ``teachers``.
        patch_size: side length of square mask patches.
        kl_temperature: channel-wise KL softmax temperature.
        imgsz: training image size.
        device: torch device; auto-detected if None.
    """

    def __init__(
        self,
        model,
        teachers: Sequence,
        *,
        feat_level: str = "P5",
        mask_ratio_student: float = 0.75,
        mask_ratio_teachers: Sequence[float] = (0.1, 0.25, 0.4),
        patch_size: int = 16,
        kl_temperature: float = 4.0,
        imgsz: int = 640,
        device=None,
    ):
        if feat_level not in _VALID_LEVELS:
            raise ValueError(f"feat_level must be one of {_VALID_LEVELS}, got {feat_level!r}")
        if len(teachers) < 2:
            raise ValueError(f"CoMAD needs >= 2 teachers for consensus, got {len(teachers)}")
        if len(mask_ratio_teachers) != len(teachers):
            raise ValueError(
                f"mask_ratio_teachers length ({len(mask_ratio_teachers)}) "
                f"must match teachers ({len(teachers)})"
            )

        self.feat_level = feat_level
        self.n_teachers = len(teachers)
        self.mask_ratio_student = float(mask_ratio_student)
        self.mask_ratio_teachers = [float(r) for r in mask_ratio_teachers]
        self.patch_size = int(patch_size)
        self.kl_temperature = float(kl_temperature)
        self.imgsz = int(imgsz)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── student backbone ─────────────────────────────────────────────
        if isinstance(model, str):
            from ultralytics import YOLO
            self.model = YOLO(model).model.to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        self.student_tap = MultiScaleFeatureTap(self.model, levels=(feat_level,))
        self.student_tap.setup()
        student_ch = infer_in_channels(
            self.model, self.student_tap,
            imgsz=min(self.imgsz, 64), device=self.device,
        )[feat_level]
        self.student_tap.clear()
        self.student_ch = student_ch

        # ── frozen teachers + taps + trainable adapters ──────────────────
        self.teachers: List[nn.Module] = []
        self.teacher_taps: List[MultiScaleFeatureTap] = []
        self.adapters = nn.ModuleList()
        for spec in teachers:
            teacher = self._load_teacher(spec)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            tap = MultiScaleFeatureTap(teacher, levels=(feat_level,))
            tap.setup()
            teacher_ch = infer_in_channels(
                teacher, tap, imgsz=min(self.imgsz, 64), device=self.device,
            )[feat_level]
            tap.clear()
            # Linear adapter + layer norm — aligns teacher features to the
            # student's space (CoMAD). GroupNorm(1, C) = channel-wise LayerNorm.
            adapter = nn.Sequential(
                nn.Conv2d(teacher_ch, student_ch, kernel_size=1),
                nn.GroupNorm(1, student_ch),
            ).to(self.device)
            self.teachers.append(teacher)
            self.teacher_taps.append(tap)
            self.adapters.append(adapter)

        self.augmentation = build_pipeline("simclr_v2", imgsz=self.imgsz)

    # ── teacher loading ──────────────────────────────────────────────────

    def _load_teacher(self, spec) -> nn.Module:
        """An SSL checkpoint path → YOLOv8n skeleton with loaded weights;
        an nn.Module → used directly."""
        if isinstance(spec, str):
            from ultralytics import YOLO
            m = YOLO("yolov8n.pt").model.to(self.device)
            load_backbone(m, spec, strict=False, verbose=False)
            return m
        return spec.to(self.device)

    # ── asymmetric masking ───────────────────────────────────────────────

    def _apply_mask(self, imgs: torch.Tensor, ratio: float) -> torch.Tensor:
        """Zero out a `ratio` fraction of square input patches (per image)."""
        if ratio <= 0:
            return imgs
        B, _, H, W = imgs.shape
        ps = self.patch_size
        gh, gw = H // ps, W // ps
        n_patches = gh * gw
        n_mask = int(ratio * n_patches)
        if n_mask == 0:
            return imgs
        masked = imgs.clone()
        for b in range(B):
            idx = torch.randperm(n_patches, device=imgs.device)[:n_mask]
            for p in idx.tolist():
                r, c = p // gw, p % gw
                masked[b, :, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps] = 0.0
        return masked

    # ── feature extraction ───────────────────────────────────────────────

    def _extract(self, model: nn.Module, tap: MultiScaleFeatureTap,
                 x: torch.Tensor) -> torch.Tensor:
        tap.clear()
        _ = model(x)
        return tap.get_features()[self.feat_level]   # [B, C, H, W]

    # ── consensus gating ─────────────────────────────────────────────────

    @staticmethod
    def _cosine_affinity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-position cosine similarity over the channel axis → [B, H, W]."""
        a = F.normalize(a.float(), dim=1)
        b = F.normalize(b.float(), dim=1)
        return (a * b).sum(dim=1)

    def _consensus_gate(
        self, student: torch.Tensor, teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse teacher features by joint consensus gating.

        Each teacher's per-position gate combines its cosine affinity to the
        student with its inter-teacher agreement; gates are softmax-normalized
        across teachers. Parameter-free — gates are detached (no gradient).
        """
        n = len(teacher_feats)
        affinity = [self._cosine_affinity(t, student) for t in teacher_feats]
        agreement = []
        for i in range(n):
            others = [
                self._cosine_affinity(teacher_feats[i], teacher_feats[j])
                for j in range(n) if j != i
            ]
            agreement.append(torch.stack(others, dim=0).mean(dim=0))  # [B, H, W]

        raw = torch.stack(
            [a * g for a, g in zip(affinity, agreement)], dim=0,
        )  # [n, B, H, W]
        gates = torch.softmax(raw, dim=0).detach()
        fused = sum(
            gates[i].unsqueeze(1) * teacher_feats[i] for i in range(n)
        )  # [B, C, H, W]
        return fused

    # ── channel-wise KL ──────────────────────────────────────────────────

    def _cwd_kl(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """CWD-style channel-wise KL divergence (scalar)."""
        T = self.kl_temperature
        B, C, H, W = student.shape
        s = (student.float() / T).reshape(B, C, H * W)
        t = (teacher.float() / T).reshape(B, C, H * W)
        s_logp = F.log_softmax(s, dim=2)
        t_logp = F.log_softmax(t, dim=2)
        t_p = t_logp.exp()
        kl = (t_p * (t_logp - s_logp)).sum(dim=2)   # [B, C]
        return kl.mean() * (T * T)

    # ── single training step ─────────────────────────────────────────────

    def _step(self, imgs: torch.Tensor) -> Dict:
        """One CoMAD step — asymmetric masking, consensus gating, KL distill."""
        imgs = imgs.to(self.device, non_blocking=True)
        with torch.no_grad():
            imgs = self.augmentation(imgs)

        # Student sees a heavily masked input.
        student_in = self._apply_mask(imgs, self.mask_ratio_student)
        student_feat = self._extract(self.model, self.student_tap, student_in)

        # Each teacher sees a lighter, distinct mask; frozen backbone, but
        # the adapter is trainable so gradient still flows into it.
        teacher_feats = []
        for i in range(self.n_teachers):
            t_in = self._apply_mask(imgs, self.mask_ratio_teachers[i])
            with torch.no_grad():
                raw = self._extract(self.teachers[i], self.teacher_taps[i], t_in)
            teacher_feats.append(self.adapters[i](raw))

        fused = self._consensus_gate(student_feat, teacher_feats)
        loss = self._cwd_kl(student_feat, fused)
        return {"loss": loss, "info": {"loss": float(loss.detach())},
                "batch_size": imgs.shape[0]}

    # ── trainable parameters ─────────────────────────────────────────────

    def _trainable_parameters(self) -> List[nn.Parameter]:
        """Student backbone + teacher adapters. Teacher backbones excluded."""
        return list(self.model.parameters()) + list(self.adapters.parameters())

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
        output: str = "comad_yolo_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
    ) -> str:
        """Run CoMAD-style pretraining. Returns the saved backbone path."""
        dataset = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
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

        self.model.train()
        self.adapters.train()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            ep_loss = 0.0
            n_batches = 0
            for imgs in dataloader:
                optimizer.zero_grad(set_to_none=True)
                out = self._step(imgs)
                out["loss"].backward()
                optimizer.step()
                scheduler.step()
                ep_loss += float(out["loss"].detach())
                n_batches += 1

            n_batches = max(1, n_batches)
            if print_every and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"[comad-yolo] epoch {epoch}/{epochs} "
                    f"loss={ep_loss / n_batches:.4f} ({time.time() - t0:.1f}s)"
                )
            if save_every and epoch % save_every == 0:
                self._save(output, epoch)

        self._save(output, epochs)
        return output

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save(self, output: str, epoch: int) -> None:
        """Save the student backbone with a CoMAD-YOLO marker."""
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "type": "ssl_pretrained",
            "extra": {
                "type": "comad_yolo",
                "feat_level": self.feat_level,
                "n_teachers": self.n_teachers,
            },
        }, output)

    # ── lifecycle ────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        taps = [getattr(self, "student_tap", None)] + list(
            getattr(self, "teacher_taps", []),
        )
        for tap in taps:
            if tap is not None:
                try:
                    tap.close()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"CoMADYOLOTrainer(n_teachers={self.n_teachers}, "
            f"feat_level={self.feat_level}, "
            f"mask_student={self.mask_ratio_student})"
        )
