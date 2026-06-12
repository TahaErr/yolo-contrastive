"""AuxChannel — pluggable teacher-channel interface for the anchored joint trainer.

A *channel* is one source of auxiliary, label-like supervision (TERRA geometry,
REVISIT persistence, GASP-Real natural scale, ...) that plugs into
:class:`~yolo_contrastive.anchored.trainer.AnchoredJointTrainer`. The trainer
owns the COCO replay anchor (R3); each channel owns

    1. its trainable heads        — registered via :meth:`AuxChannel.attach`,
    2. its loss terms             — computed in :meth:`AuxChannel.loss`,
    3. its dataloader factory     — :meth:`AuxChannel.build_loader`.

Trainer <-> channel contract (read carefully — three methods, three guarantees):

``attach(model, taps) -> nn.ModuleList``
    Called once at trainer construction, after the P3/P4/P5 taps are set up.
    The channel builds its trainable head modules here and returns them as an
    ``nn.ModuleList``. The trainer moves them to the training device, enables
    ``requires_grad`` (E5) and puts their parameters into the optimizer's
    head-LR param group. Channels MUST NOT register heads as submodules of
    ``model`` (they would leak into the exported detector and the EMA) and
    MUST NOT add trainable modules on any teacher side (R6).

``loss(batch, taps) -> dict[str, Tensor]``
    Called once per optimizer step with one batch from the channel's loader.
    The trainer guarantees, immediately before this call:
        * every tensor in ``batch`` has been moved to the training device and
          ``batch["img"]`` is float in [0, 1];
        * ``taps.clear()`` was called and exactly ONE forward pass
          ``model(batch["img"])`` ran under autocast — so
          ``taps.get_features()`` returns this batch's {"P3", "P4", "P5"}
          feature maps with grad attached (after warmup).
    The channel computes its named scalar loss terms from those features (and
    any label tensors it packed into ``batch``) through its own heads. The
    trainer sums the dict values, multiplies by ``lambda_aux`` and backprops
    in the SAME optimizer step as the replay detection loss (R3).
    Channels must NOT run extra ``model(...)`` forwards inside ``loss()`` —
    the taps would be overwritten. Multi-view channels concatenate views
    along the batch dimension in their collate function and split the tap
    features by batch index instead.

``build_loader(cfg) -> iterable``
    Returns the channel's dataloader: any (re-)iterable yielding ``dict``
    batches with at least ``"img"``: float [B, 3, H, W] in [0, 1] (uint8 is
    also accepted and normalized by the trainer). All other keys are
    channel-specific (dense label maps, boxes, coords, pair indices, ...).
    HARD RULE (R5): every spatial label in the batch must be produced by the
    SAME geometric transform as the image, inside the loader/collate — the
    trainer never re-augments, so image/label misalignment must be impossible
    by construction. ``cfg`` is a plain dict with keys
    ``{"imgsz", "batch", "workers", "device"}``.

Design rules inherited from the measured failure history (see wf_readings.md):
    R1/R2 — supervision must be label-like and vary per image; never a
            pointwise regression onto a frozen-teacher feature.
    R4    — no cross-scale contrastive negatives; prefer classification /
            regression on maps and boxes, or positive-only consistency.
    R6    — zero trainable teacher-side modules.
    R7    — no COCO-class detector pseudo-labeling of the pool.

Minimal example (also used by tests/test_anchored.py)::

    class DummyChannel(AuxChannel):
        name = "dummy"

        def attach(self, model, taps):
            c5 = probe_tap_channels(model, taps)["P5"]
            self.head = nn.Conv2d(c5, 4, 1)
            return nn.ModuleList([self.head])

        def loss(self, batch, taps):
            f5 = taps.get_features()["P5"]
            return {"mse": self.head(f5).pow(2).mean()}

        def build_loader(self, cfg):
            b, s = cfg["batch"], cfg["imgsz"]
            return [{"img": torch.rand(b, 3, s, s)} for _ in range(8)]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


class AuxChannel(ABC):
    """Abstract base for an auxiliary teacher channel.

    Subclasses set ``name`` (unique within a trainer — it keys the channel's
    loader, its loss terms in the metrics dict as ``"{name}/{term}"``, and its
    head group in the sentinel log).
    """

    #: Unique channel identifier. Override in subclasses.
    name: str = "aux"

    @abstractmethod
    def attach(self, model: nn.Module, taps: Any) -> nn.ModuleList:
        """Build and return this channel's trainable heads.

        Args:
            model: the (already device-placed) detector being trained.
            taps: the shared ``MultiScaleFeatureTap`` (already ``setup()``)
                capturing P3/P4/P5. Use :func:`probe_tap_channels` to read
                channel widths.

        Returns:
            ``nn.ModuleList`` of head modules. The trainer adds their
            parameters to the optimizer at head LR; do not add them to
            ``model`` or to any optimizer yourself.
        """

    @abstractmethod
    def loss(self, batch: Dict[str, Any], taps: Any) -> Dict[str, torch.Tensor]:
        """Compute named scalar loss terms for one channel batch.

        ``taps.get_features()`` holds the features of ``batch["img"]``
        (the trainer ran the forward). Return ``{term_name: scalar_tensor}``;
        the trainer sums terms, scales by ``lambda_aux`` and backprops.
        """

    @abstractmethod
    def build_loader(self, cfg: Dict[str, Any]) -> Iterable:
        """Build this channel's dataloader.

        Args:
            cfg: ``{"imgsz": int, "batch": int, "workers": int, "device": str}``.

        Returns:
            A re-iterable yielding dict batches with at least ``"img"``.
            The trainer cycles it for ``steps_per_epoch`` steps per epoch.
        """

    def on_epoch_end(self, epoch: int) -> Dict[str, float]:
        """Optional per-epoch CHANNEL sentinel hook (R9, structural).

        ``AnchoredJointTrainer.train`` calls this once per epoch, right after
        its own sentinel suite, and logs the returned flat metric dict into
        the epoch history as ``sentinel/{channel.name}/{key}`` — so channel
        sentinels run by construction, not by run-loop convention. Default:
        no channel sentinels (empty dict).
        """
        return {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


def probe_tap_features(
    model: nn.Module, taps: Any, imgsz: int = 64
) -> Dict[str, torch.Tensor]:
    """One dummy forward through ``model``; returns the detached tap features.

    Probe hygiene, both halves of the documented silent-bug class:
        * ``model.eval()`` is set (and the previous mode restored) around the
          forward, so BatchNorm running statistics are never polluted by the
          all-zeros probe input (mirrors ``PersistenceChannel._probe``);
        * the taps are cleared afterwards, so no stale probe features leak
          into the first real training step.

    Channels use the returned feature maps for width AND stride checks
    (spatial size of level L must be ``imgsz / stride(L)``).
    """
    try:
        p = next(model.parameters())
        device, dtype = p.device, p.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)
    was_training = model.training
    model.eval()
    taps.clear()
    try:
        with torch.no_grad():
            _ = model(dummy)
        feats = {k: v.detach() for k, v in taps.get_features().items()}
    finally:
        taps.clear()
        if was_training:
            model.train()
    return feats


def probe_tap_channels(model: nn.Module, taps: Any, imgsz: int = 64) -> Dict[str, int]:
    """Read per-level channel widths {"P3": C3, ...} via a dummy forward.

    Thin wrapper over :func:`probe_tap_features` (eval-mode probe, taps
    cleared afterwards) keeping only the channel counts.
    """
    feats = probe_tap_features(model, taps, imgsz=imgsz)
    return {lv: int(t.shape[1]) for lv, t in feats.items()}
