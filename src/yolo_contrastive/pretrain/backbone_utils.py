"""Backbone save / load / freeze utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_backbone(
    model: nn.Module,
    path: str,
    epoch: int = -1,
    extra: Optional[dict] = None,
) -> str:
    """Backbone ağırlıklarını kaydet.

    Args:
        model: YOLO model (DetectionModel veya nn.Module)
        path: Kayıt yolu (.pt)
        epoch: Pretraining epoch sayısı
        extra: Ek metadata (config vs.)

    Returns:
        Kaydedilen dosya yolu
    """
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "type": "ssl_pretrained",
    }
    if extra:
        state["extra"] = extra

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_backbone(
    model: nn.Module,
    path: str,
    strict: bool = False,
    verbose: bool = True,
    backbone_only: bool = False,
    backbone_layers: int = 10,
) -> int:
    """Pretrained backbone ağırlıklarını yükle.

    Args:
        model: Hedef YOLO model
        path: Pretrained checkpoint yolu
        strict: True ise tüm key'ler eşleşmeli
        verbose: Loglama
        backbone_only: True ise sadece backbone katmanlarını yükle (layer 0..backbone_layers-1)
        backbone_layers: Backbone katman sınırı (default 10, YOLOv8 backbone=0-9)

    Returns:
        Yüklenen parametre sayısı
    """
    # FIX: Önce güvenli yüklemeyi dene (weights_only=True), başarısız olursa fallback
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        import warnings
        warnings.warn(
            f"torch.load weights_only=True failed for '{path}'. "
            f"Falling back to weights_only=False — only load checkpoints you trust.",
            UserWarning, stacklevel=2,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Checkpoint formatını tespit et
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    filtered = 0

    for key, value in state.items():
        # backbone_only filtresi: sadece "model.X." formatında X < backbone_layers
        if backbone_only:
            parts = key.split(".")
            is_backbone = False
            if len(parts) >= 2 and parts[0] == "model":
                try:
                    layer_idx = int(parts[1])
                    is_backbone = layer_idx < backbone_layers
                except ValueError:
                    pass
            if not is_backbone:
                filtered += 1
                continue

        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state)

    if verbose:
        total = len(state)
        print(f"[ycl] Backbone loaded: {loaded}/{total} params "
              f"({skipped} skipped, {filtered} filtered by backbone_only)")
        if "epoch" in checkpoint:
            print(f"[ycl] Pretrained for {checkpoint['epoch']} epochs")

    return loaded


def freeze_backbone(model: nn.Module, num_layers: int = 10, verbose: bool = True) -> int:
    """İlk N katmanı dondur (gradient hesaplanmaz).

    YOLOv8 yapısı:
        - Layers 0-9: backbone
        - Layers 10-21: neck
        - Layer 22: detect head

    Args:
        model: YOLO model
        num_layers: Dondurulacak katman sayısı
        verbose: Loglama

    Returns:
        Dondurulan parametre sayısı
    """
    frozen = 0
    for name, param in model.named_parameters():
        # "model.X." formatında — X katman numarası
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model":
            try:
                layer_idx = int(parts[1])
                if layer_idx < num_layers:
                    param.requires_grad = False
                    frozen += 1
            except ValueError:
                pass

    if verbose:
        total = sum(1 for _ in model.parameters())
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[ycl] Frozen: {frozen}/{total} params, trainable: {trainable}")

    return frozen


def unfreeze_all(model: nn.Module, verbose: bool = True) -> None:
    """Tüm parametreleri aç."""
    for param in model.parameters():
        param.requires_grad = True
    if verbose:
        total = sum(1 for _ in model.parameters())
        print(f"[ycl] Unfrozen: all {total} params trainable")


# ── whole-detector transplant (R8) ──────────────────────────────────────────


def _safe_torch_load(path) -> dict:
    """torch.load with weights_only=True first, trusted fallback second."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        import warnings

        warnings.warn(
            f"torch.load weights_only=True failed for '{path}'. "
            f"Falling back to weights_only=False — only load checkpoints you trust.",
            UserWarning, stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(source) -> dict:
    """Normalize a checkpoint path / checkpoint dict / module / raw state dict
    into a flat ``{key: Tensor}`` state dict."""
    if isinstance(source, (str, Path)):
        source = _safe_torch_load(source)
    if isinstance(source, nn.Module):
        return source.state_dict()
    if not isinstance(source, dict):
        raise TypeError(
            f"source must be a path, checkpoint dict, state dict or nn.Module, "
            f"got {type(source).__name__}"
        )
    for key in ("model_state_dict", "model"):
        inner = source.get(key)
        if isinstance(inner, nn.Module):
            return inner.state_dict()
        if isinstance(inner, dict):
            return inner
    return source


def transplant_full(model: nn.Module, source, strict: bool = False, verbose: bool = True) -> int:
    """Whole-detector transplant (design rule R8): copy EVERY shape-matching
    tensor of a full state dict (backbone + neck + head) into ``model``.

    This is the export-side counterpart of ``load_backbone(backbone_only=True)``:
    the measured failure history showed that transplanting a pretrained
    backbone under an untouched COCO neck/head is what destroyed downstream
    mAP, so anchored-joint checkpoints are moved whole.

    Values are copied via plain ``load_state_dict`` — NEVER
    ``load_state_dict(assign=True)``, which would alias checkpoint storage
    into the live module and reproduce the Risk-16 EMA-collapse catastrophe.

    Args:
        model: target detector (e.g. ``YOLO(...).model``).
        source: checkpoint path, checkpoint dict (``model_state_dict`` /
            ``model`` key, or flat), raw state dict, or nn.Module.
        strict: if True, raise when any source key is missing/shape-mismatched.
        verbose: print a one-line summary.

    Returns:
        Number of tensors copied.
    """
    state = _extract_state_dict(source)
    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for key, value in state.items():
        if torch.is_tensor(value) and key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped.append(key)

    if strict and skipped:
        raise RuntimeError(
            f"transplant_full(strict=True): {len(skipped)} source keys not transplanted "
            f"(first few: {skipped[:5]})"
        )

    model.load_state_dict(model_state)  # plain load — copies values, no aliasing

    if verbose:
        print(f"[ycl] Full transplant: {loaded}/{len(state)} tensors copied "
              f"({len(skipped)} skipped)")
    return loaded
