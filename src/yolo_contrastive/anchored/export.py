"""Export / reload for anchored joint training — whole-detector transplant (R8).

The measured failure history is unambiguous: transplanting a pretrained
BACKBONE under an untouched COCO neck/head destroyed downstream mAP, while the
only historical win moved the whole detector. ``save_checkpoint`` therefore
defaults to ``transplant="full"`` (backbone + neck + head in one state dict)
and ``load_for_finetune`` copies it whole into a fresh ultralytics ``YOLO``.

Risk-16 discipline: every tensor written to disk is a detached CPU CLONE of
the live weight, and reloads go through plain ``load_state_dict`` value copies
— ``assign=True`` is never used anywhere on this path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# NOTE: ..pretrain.backbone_utils is imported lazily inside functions —
# importing the pretrain package pulls its dataset module (opencv), and
# "import yolo_contrastive.anchored" must stay lightweight (E2).

#: YOLOv8 layout: layers 0-9 backbone, 10-(n-2) neck, last layer Detect head.
BACKBONE_LAYERS = 10

_TRANSPLANT_MODES = ("full", "backbone")


def _is_backbone_key(key: str, backbone_layers: int = BACKBONE_LAYERS) -> bool:
    """True for state-dict keys of the form ``model.<idx>.*`` with idx < cutoff."""
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "model":
        try:
            return int(parts[1]) < backbone_layers
        except ValueError:
            return False
    return False


def save_checkpoint(
    model: nn.Module,
    path: str,
    transplant: str = "full",
    epoch: int = -1,
    extra: Optional[Dict[str, Any]] = None,
    backbone_layers: int = BACKBONE_LAYERS,
) -> str:
    """Save a detector checkpoint for downstream fine-tuning.

    Args:
        model: detector to save (typically the trainer's EMA model).
        path: output ``.pt`` path (parent dirs created).
        transplant: ``"full"`` (default, R8 — whole detector) or
            ``"backbone"`` (layers 0..backbone_layers-1 only; kept for
            ablation arms that deliberately reproduce the historical failure).
        epoch: epoch stamp stored in the checkpoint.
        extra: optional metadata dict (must be torch.save-serializable).
        backbone_layers: backbone cutoff used by ``transplant="backbone"``.

    Returns:
        The saved path (str).
    """
    if transplant not in _TRANSPLANT_MODES:
        raise ValueError(f"transplant must be one of {_TRANSPLANT_MODES}, got {transplant!r}")

    state = model.state_dict()
    if transplant == "backbone":
        state = {k: v for k, v in state.items() if _is_backbone_key(k, backbone_layers)}
        if not state:
            raise ValueError(
                "transplant='backbone' produced an empty state dict — the model's layer "
                "naming does not match the expected 'model.<idx>.*' scheme."
            )
    # Detached CPU clones: the file must never alias live training tensors.
    state = {k: v.detach().cpu().clone() for k, v in state.items()}

    payload: Dict[str, Any] = {
        "model_state_dict": state,
        "epoch": int(epoch),
        "type": "anchored_joint",
        "transplant": transplant,
    }
    yaml_cfg = getattr(model, "yaml", None)
    if isinstance(yaml_cfg, dict):
        payload["model_yaml"] = yaml_cfg
    if extra:
        payload["extra"] = dict(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return str(out)


def load_for_finetune(path: str, base: str = "yolov8n.pt", device: str = "cpu",
                      verbose: bool = True):
    """Rebuild an ultralytics ``YOLO`` and transplant an anchored checkpoint into it.

    Args:
        path: checkpoint written by :func:`save_checkpoint` (or any checkpoint
            with a ``model_state_dict`` / ``model`` / flat state dict).
        base: ultralytics model spec used to build the architecture —
            ``"yolov8n.pt"`` (downloads/caches the official weights; every
            non-transplanted tensor keeps its COCO value) or
            ``"yolov8n.yaml"`` (offline random init; tests use this).
        device: where to place the rebuilt model.
        verbose: print transplant summary.

    Returns:
        ``ultralytics.YOLO`` with the checkpoint weights copied in (plain
        value copies — storage independent of the checkpoint tensors), ready
        for ``.train(...)`` fine-tuning.
    """
    from ultralytics import YOLO  # lazy: optional [yolo] extra

    from ..pretrain.backbone_utils import _safe_torch_load, transplant_full  # lazy (E2)

    ckpt = _safe_torch_load(path)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path!r} is not a dict (got {type(ckpt).__name__})")

    yolo = YOLO(base, task="detect")
    yolo.model.to(device)
    n_loaded = transplant_full(yolo.model, ckpt, verbose=verbose)
    if n_loaded == 0:
        raise RuntimeError(
            f"load_for_finetune: zero tensors transplanted from {path!r} into {base!r} — "
            f"architecture mismatch?"
        )

    # ultralytics Model.train() only carries self.model weights into its
    # internal trainer when self.ckpt is truthy (yaml-built YOLOs have
    # ckpt=None and would silently re-randomize). epoch=-1 and no
    # "optimizer" key keep the auto-resume path disabled.
    if not getattr(yolo, "ckpt", None):
        yolo.ckpt = {"epoch": -1}

    if verbose:
        mode = ckpt.get("transplant", "full") if isinstance(ckpt, dict) else "full"
        print(f"[ycl-anchored] load_for_finetune: {n_loaded} tensors ({mode}) "
              f"from {path} into {base}")
    return yolo
