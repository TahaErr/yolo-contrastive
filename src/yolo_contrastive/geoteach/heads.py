"""TERRA trainable heads: dense ordinal head (P3) + fresh 2-class Detect head.

Both heads are CHANNEL heads in the anchored-trainer sense: returned from
``TerraChannel.attach`` as an ``nn.ModuleList``, trained at head LR (1e-3),
NEVER registered on the detector itself, and discarded at export — inference
cost stays exactly YOLOv8n (R8). The COCO 80-class Detect head is untouched
and used only by replay batches (no label-space interference).

DenseOrdinalHead: 2-conv ``C -> 64 -> 6`` (~50K params at C=64) producing
per-position logits over the 6 ordinal bins at P3 / stride 8.

GeoDetectHead: a fresh ultralytics ``Detect(nc=2)`` head consuming the SAME
P3/P4/P5 neck taps, so the mined polarity boxes train the backbone+neck
through the real TAL assigner + CIoU + DFL + BCE machinery (the actual
v8DetectionLoss, not an approximation). ultralytics is imported lazily inside
``__init__`` (E2). The DFL integral conv inside the fresh head is a FIXED
arange operator — it ships frozen and :meth:`GeoDetectHead.freeze_dfl`
re-freezes it after the trainer's blanket ``requires_grad_(True)`` on channel
heads (E5; it receives no gradient in training mode either way).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from ..dense.multi_scale_tap import YOLOV8_FPN_STRIDES


class DenseOrdinalHead(nn.Module):
    """2-conv dense classification head: ``C -> hidden -> num_classes``.

    Applied to the P3 tap (stride 8). ~50K params at the yolov8n P3 width
    (C=64, hidden=64, 6 classes).
    """

    def __init__(self, in_channels: int, num_classes: int = 6, hidden: int = 64):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] features → [B, num_classes, H, W] logits."""
        return self.conv2(self.act(self.conv1(x)))


class GeoDetectHead(nn.Module):
    """Fresh 2-class ultralytics Detect head on the P3/P4/P5 taps.

    Args:
        ch: per-level channel widths in (P3, P4, P5) order — from
            ``probe_tap_channels`` (yolov8n: (64, 128, 256)).
        nc: number of geometric classes (2: depression / elevation).
        reg_max: DFL bins (ultralytics default 16).
        strides: feature strides, (8, 16, 32) for P3/P4/P5.

    Training-mode forward returns the ultralytics preds dict
    ``{"boxes", "scores", "feats"}`` consumed by ``v8DetectionLoss`` —
    see :meth:`build_criterion`.
    """

    def __init__(
        self,
        ch: Sequence[int],
        nc: int = 2,
        reg_max: int = 16,
        strides: Sequence[int] = (
            YOLOV8_FPN_STRIDES["P3"], YOLOV8_FPN_STRIDES["P4"], YOLOV8_FPN_STRIDES["P5"],
        ),
    ):
        from ultralytics.nn.modules.head import Detect  # lazy: [yolo] extra

        super().__init__()
        ch = tuple(int(c) for c in ch)
        if len(ch) != len(strides):
            raise ValueError(f"ch {ch} and strides {tuple(strides)} length mismatch")
        try:
            self.detect = Detect(nc=nc, reg_max=reg_max, ch=ch)
        except TypeError:
            # Older ultralytics releases (Detect(nc, ch) only) hardwire
            # reg_max at the module default — construct without the kwarg
            # and verify the result below instead of silently training a
            # head with a different DFL bin count than requested.
            self.detect = Detect(nc=nc, ch=ch)
        actual_reg_max = int(getattr(self.detect, "reg_max", -1))
        if actual_reg_max != int(reg_max):
            raise ValueError(
                f"installed ultralytics Detect built with reg_max={actual_reg_max}, "
                f"requested {reg_max} — this ultralytics version does not accept "
                f"Detect(reg_max=...); upgrade ultralytics or use the default reg_max."
            )
        # Detect.stride is normally computed during DetectionModel build; this
        # head lives outside any DetectionModel, so set it explicitly. The
        # taps are verified P3/P4/P5 by the shared MultiScaleFeatureTap.
        self.detect.stride = torch.tensor([float(s) for s in strides])
        self.detect.bias_init()  # stable cls/box bias start (needs stride)
        self.freeze_dfl()
        self.nc = nc
        self._criterion = None

    def freeze_dfl(self) -> None:
        """Keep the fixed DFL integral conv frozen (E5).

        ``Detect`` ships it with ``requires_grad=False``, but the anchored
        trainer blanket-enables grad on channel heads after ``attach`` —
        ``TerraChannel.loss`` calls this again on the first step. (The param
        receives no gradient in training mode regardless; this guards the
        invariant explicitly.)
        """
        dfl = getattr(self.detect, "dfl", None)
        if dfl is not None and hasattr(dfl, "conv"):
            dfl.conv.weight.requires_grad_(False)

    def forward(self, feats: List[torch.Tensor]):
        """[P3, P4, P5] feature list → ultralytics preds (dict in train mode)."""
        # Detect.forward mutates nothing on the inputs; taps stay intact.
        return self.detect(list(feats))

    def build_criterion(self) -> "object":
        """Build (once) the real ultralytics ``v8DetectionLoss`` for this head.

        ``v8DetectionLoss`` reads ``model.model[-1]`` (the Detect module),
        ``model.args`` (hyp gains) and the device from ``model.parameters()``
        — a minimal shim provides exactly that. Called lazily AFTER the
        trainer has moved the head to the training device (the criterion
        captures the device at construction).
        """
        if self._criterion is None:
            from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
            from ultralytics.utils.loss import v8DetectionLoss  # lazy

            shim = _CriterionShim(self.detect, IterableSimpleNamespace(**DEFAULT_CFG_DICT))
            self._criterion = v8DetectionLoss(shim)
        return self._criterion


class _CriterionShim(nn.Module):
    """Just enough model surface for ``v8DetectionLoss.__init__``:
    ``.model[-1]`` → Detect, ``.args`` → hyp namespace, ``.parameters()`` →
    device. Never trained, never exported; holds NO extra parameters (R6) —
    ``self.model[-1]`` references the live GeoDetectHead Detect module."""

    def __init__(self, detect: nn.Module, args) -> None:
        super().__init__()
        self.model = nn.ModuleList([detect])
        self.args = args


def head_channels(tap_channels: Dict[str, int]) -> List[int]:
    """(P3, P4, P5) width list from a ``probe_tap_channels`` dict."""
    try:
        return [int(tap_channels[k]) for k in ("P3", "P4", "P5")]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"tap_channels missing level {exc}; got {sorted(tap_channels)}")
