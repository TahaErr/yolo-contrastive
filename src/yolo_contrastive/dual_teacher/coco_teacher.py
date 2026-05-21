"""CocoTeacher — frozen COCO-pretrained YOLOv8x feature teacher.

Faz 5.3 — DT-SAPS dual-teacher framework (WORK_PLAN_v9 §2.4, Risk 17).

The COCO teacher is one of the two teachers in DT-SAPS. It supplies
multi-scale P3/P4/P5 feature maps from a frozen, COCO-pretrained YOLOv8x
backbone. These features are distilled into the student during dense SSL
pretraining (consensus_loss.py).

Two responsibilities:
    1. extract_features() — raw teacher feature maps, used to BUILD the
       teacher cache (§2.4 — features cached once on un-augmented images).
    2. adapt() — per-scale linear adapter mapping teacher channel counts
       to the student's (Risk 17 — YOLOv8x P3/P4/P5 widths differ from
       YOLOv8n). The adapter is the ONLY trainable part; the teacher
       backbone is hard-frozen.

Design decisions (architectural — recorded for the paper):
  - Trainable adapter, not a frozen random projection. Teacher and student
    live in different feature spaces; the mapping between them is something
    to be LEARNED. A frozen random projection would corrupt the distillation
    signal. This mirrors the linear-probe protocol (frozen backbone +
    trainable head).
  - The cache stores RAW (un-adapted) features. The adapter changes during
    training; caching adapted features would make the cache stale after the
    first optimizer step. extract_features() returns raw teacher output;
    adapt() is applied at train time.
  - Spatial sizes already match across YOLOv8x/YOLOv8n at a given imgsz
    (same FPN strides 8/16/32); only channel counts differ. The adapter is
    therefore a pointwise 1x1 conv — channel remap, no spatial change.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from ..dense import MultiScaleFeatureTap, infer_in_channels


class CocoTeacher(nn.Module):
    """Frozen COCO YOLOv8x feature teacher with a trainable per-scale adapter.

    Args:
        weights: an Ultralytics model spec (e.g. ``"yolov8x.pt"``) or a
            pre-built ``nn.Module`` backbone. A string triggers ultralytics
            import + ``YOLO(weights).model`` extraction.
        levels: FPN levels to tap. Default ``("P3", "P4", "P5")``. Pass a
            subset (e.g. ``("P5",)``) for the P5-only cache strategy (§2.4).
        student_channels: ``{level: channel_count}`` of the student backbone.
            If given, a per-scale 1x1-conv adapter is built mapping teacher
            channels to these. If None, ``adapt()`` is unavailable.
        device: where to place the teacher. Auto-detected if None.

    Attributes:
        teacher_channels: ``{level: channel_count}`` probed from the backbone.
    """

    def __init__(
        self,
        weights: Union[str, nn.Module] = "yolov8x.pt",
        levels: tuple = ("P3", "P4", "P5"),
        student_channels: Optional[Dict[str, int]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.levels = tuple(levels)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── load backbone ────────────────────────────────────────────────
        if isinstance(weights, str):
            from ultralytics import YOLO
            self.backbone: nn.Module = YOLO(weights).model
        else:
            self.backbone = weights
        self.backbone = self.backbone.to(self.device)

        # ── hard-freeze the teacher ──────────────────────────────────────
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── tap + channel probe ──────────────────────────────────────────
        self.tap = MultiScaleFeatureTap(self.backbone, levels=self.levels)
        self.tap.setup()
        self.teacher_channels: Dict[str, int] = infer_in_channels(
            self.backbone, self.tap, imgsz=64, device=self.device
        )

        # ── per-scale adapter (Risk 17) — trainable ─────────────────────
        self.student_channels: Optional[Dict[str, int]] = (
            dict(student_channels) if student_channels is not None else None
        )
        self.adapter: Optional[nn.ModuleDict] = None
        if self.student_channels is not None:
            missing = [lv for lv in self.levels if lv not in self.student_channels]
            if missing:
                raise ValueError(
                    f"student_channels missing levels {missing}; "
                    f"got {sorted(self.student_channels.keys())}"
                )
            adapter = {}
            for lv in self.levels:
                adapter[lv] = nn.Conv2d(
                    self.teacher_channels[lv],
                    self.student_channels[lv],
                    kernel_size=1,
                )
            self.adapter = nn.ModuleDict(adapter).to(self.device)

    # ── feature extraction (cache build path) ────────────────────────────

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Raw teacher feature maps — un-adapted, for building the cache.

        Args:
            images: ``[B, 3, H, W]`` un-augmented images.

        Returns:
            ``{level: [B, C_teacher, H/s, W/s]}`` detached feature maps.
        """
        images = images.to(self.device)
        self.tap.clear()
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            _ = self.backbone(images)
        finally:
            if was_training:
                self.backbone.train()
        feats = self.tap.get_features()
        # Detach + clone so downstream mutation / cache I/O is safe.
        return {lv: t.detach().clone() for lv, t in feats.items()}

    # ── adapter (train-time path) ────────────────────────────────────────

    def adapt(self, raw_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map raw teacher features to the student's channel counts.

        Args:
            raw_features: ``{level: [B, C_teacher, H, W]}`` — from
                extract_features() or the teacher cache.

        Returns:
            ``{level: [B, C_student, H, W]}`` — adapter output (gradients flow
            through the adapter; the teacher backbone stays frozen).
        """
        if self.adapter is None:
            raise ValueError(
                "adapt() requires student_channels at construction time."
            )
        missing = [lv for lv in self.levels if lv not in raw_features]
        if missing:
            raise ValueError(f"raw_features missing levels {missing}")
        out: Dict[str, torch.Tensor] = {}
        for lv in self.levels:
            t = raw_features[lv]
            if t.shape[1] != self.teacher_channels[lv]:
                raise ValueError(
                    f"Feature {lv!r} channel mismatch: got {t.shape[1]}, "
                    f"expected teacher channels {self.teacher_channels[lv]}"
                )
            out[lv] = self.adapter[lv](t.to(self.device))
        return out

    # ── convenience: extract + adapt in one call ─────────────────────────

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """extract_features() followed by adapt(). Requires student_channels.

        Note: this re-runs the teacher backbone every call. In training, the
        teacher cache (teacher_cache.py) should be used instead — call
        adapt() directly on cached raw features.
        """
        return self.adapt(self.extract_features(images))

    # ── lifecycle ────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        try:
            self.tap.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        has_adapter = "yes" if self.adapter is not None else "no"
        return (
            f"CocoTeacher(levels={self.levels}, "
            f"teacher_channels={self.teacher_channels}, "
            f"adapter={has_adapter})"
        )
