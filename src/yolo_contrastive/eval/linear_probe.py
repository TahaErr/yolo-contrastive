"""Linear probe — frozen backbone + linear head, multi-label classification.

Faz 4.5 — Eval infrastructure (WORK_PLAN_v5 §5).

What is linear probe?
    The classic SSL evaluation protocol: freeze the pretrained backbone,
    pool one FPN level's features to a vector, fit a linear classifier,
    measure how well it predicts image-level labels. If SSL learned
    useful features, this works; if not, it doesn't. (DINOv2, MoCo-v3,
    SimCLR all report linear probe numbers.)

Why multi-label, not single-label?
    Detection datasets have multiple classes per image (a typical BDD
    scene has cars + persons + traffic signs). Picking a single
    "dominant class" throws away most of the signal. Multi-label probe
    uses BCE loss on a multi-hot target and reports mean Average
    Precision (mAP), the standard for multi-label classification.

What gets trained?
    Only the linear head (D → num_classes). The backbone is hard-frozen:
    `requires_grad=False`, `model.eval()`, and forward inside `torch.no_grad()`.
    Tests verify this: backbone params receive zero gradient.

Inputs/outputs:
    DataLoader yields (image_tensor, multi_hot_label) where
        image_tensor: [B, 3, H, W]
        multi_hot_label: [B, num_classes] float in {0, 1}
    Trainer reports val_mAP per epoch and tracks best.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..dense import MultiScaleFeatureTap, infer_in_channels
from ..pretrain.backbone_utils import load_backbone


# ─────────────────────────────────────────────────────────────────────────
# Linear head
# ─────────────────────────────────────────────────────────────────────────


class LinearProbeHead(nn.Module):
    """Single-layer Linear(in_dim → num_classes) probe head.

    Optionally normalizes input features before projection (useful when
    backbone features have very large magnitudes, e.g. unnormalized
    raw FPN outputs).

    Args:
        in_dim: pooled feature dimension from backbone.
        num_classes: number of output classes (multi-label).
        normalize: if True, L2-normalize input features before linear.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.normalize = bool(normalize)
        self.fc = nn.Linear(self.in_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"LinearProbeHead expects [B, D], got {tuple(x.shape)}")
        if self.normalize:
            x = F.normalize(x, dim=1)
        return self.fc(x)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, num_classes={self.num_classes}, "
            f"normalize={self.normalize}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Multi-label mAP
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def multilabel_average_precision(
    logits: torch.Tensor,         # [N, C]
    targets: torch.Tensor,        # [N, C] in {0, 1}
) -> Dict[str, float]:
    """Compute per-class AP and mean AP for multi-label classification.

    Returns dict with:
        per_class_ap: [C] tensor of AP per class
        mAP: mean over classes that have at least one positive sample
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} != targets {tuple(targets.shape)}"
        )
    if logits.dim() != 2:
        raise ValueError(f"expected [N, C], got {tuple(logits.shape)}")

    N, C = logits.shape
    aps: list = []
    valid_mask: list = []

    # Per-class AP via standard precision-recall integration
    for c in range(C):
        scores_c = logits[:, c].float()
        targets_c = targets[:, c].float()
        n_pos = int(targets_c.sum().item())
        if n_pos == 0:
            aps.append(0.0)
            valid_mask.append(False)
            continue
        # Sort by score descending
        order = torch.argsort(scores_c, descending=True)
        targets_sorted = targets_c[order]
        # Cumulative TP count
        tp_cum = torch.cumsum(targets_sorted, dim=0)
        # Precision at each rank: TP_cum / (rank+1)
        ranks = torch.arange(1, N + 1, device=logits.device, dtype=torch.float32)
        precision = tp_cum / ranks
        # AP = mean precision at positive ranks
        ap = (precision * targets_sorted).sum() / n_pos
        aps.append(float(ap.item()))
        valid_mask.append(True)

    aps_tensor = torch.tensor(aps, dtype=torch.float32)
    if any(valid_mask):
        valid_aps = [aps[i] for i, ok in enumerate(valid_mask) if ok]
        mean_ap = float(sum(valid_aps) / len(valid_aps))
    else:
        mean_ap = 0.0

    return {
        "per_class_ap": aps_tensor,
        "mAP": mean_ap,
        "n_valid_classes": int(sum(valid_mask)),
    }


# ─────────────────────────────────────────────────────────────────────────
# Linear probe trainer
# ─────────────────────────────────────────────────────────────────────────


class LinearProbeTrainer:
    """Frozen-backbone linear classification probe with multi-label BCE.

    Pipeline per step:
        with no_grad: features = backbone_tap(image) → pool([B, D, H, W]) → [B, D]
        logits = head(features)              # gradient flows here only
        loss = BCEWithLogits(logits, target)
        backward + optimizer step on head only

    Args:
        backbone: pre-built nn.Module (e.g. YOLO model.model). String path
            also accepted — passed to ultralytics YOLO and `.model` taken.
            If a checkpoint path is also given via `backbone_ckpt`, those
            weights are loaded (backbone-only) into the model.
        num_classes: number of output classes for multi-label head.
        backbone_ckpt: optional path to SSL backbone checkpoint. Loaded via
            backbone_utils.load_backbone(strict=False, backbone_only=True).
        feat_level: which FPN level to probe ("P3", "P4", or "P5"). Default "P5".
        normalize_features: L2-normalize pooled features before head. Default False.
        device: "cuda", "cpu", or specific device. Auto-detected if None.

    Usage:
        probe = LinearProbeTrainer(
            backbone="yolov8n.pt",
            backbone_ckpt="my_ssl_backbone.pt",
            num_classes=80,
        )
        result = probe.fit(train_loader, val_loader, epochs=10)
        # result["val_mAP"] is the metric of interest
    """

    def __init__(
        self,
        backbone: Any,
        num_classes: int,
        backbone_ckpt: Optional[str] = None,
        feat_level: str = "P5",
        normalize_features: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if feat_level not in ("P3", "P4", "P5"):
            raise ValueError(
                f"feat_level must be P3/P4/P5, got {feat_level!r}"
            )

        self.num_classes = int(num_classes)
        self.feat_level = feat_level
        self.normalize_features = bool(normalize_features)

        # ── device ──────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, (int, float)) or (
            isinstance(device, str) and device.isdigit()
        ):
            self.device = torch.device(f"cuda:{int(device)}")
        else:
            self.device = torch.device(device)

        # ── backbone ────────────────────────────────────────────────────
        if isinstance(backbone, str):
            from ultralytics import YOLO
            yolo = YOLO(backbone)
            self.backbone = yolo.model.to(self.device)
        else:
            self.backbone = backbone.to(self.device)

        if backbone_ckpt is not None:
            n_loaded = load_backbone(
                self.backbone, backbone_ckpt,
                strict=False, verbose=True, backbone_only=True,
            )
            if n_loaded == 0:
                import warnings
                warnings.warn(
                    f"load_backbone loaded 0 params from {backbone_ckpt} — "
                    f"checkpoint may not match this backbone.",
                    UserWarning, stacklevel=2,
                )

        # Hard-freeze backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── tap, only on the requested level ────────────────────────────
        self.tap = MultiScaleFeatureTap(self.backbone, levels=(self.feat_level,))
        self.tap.setup()

        # Probe channel widths via tiny dummy forward
        ch = infer_in_channels(self.backbone, self.tap, imgsz=64, device=self.device)
        in_dim = ch[self.feat_level]

        # ── head (only trainable component) ─────────────────────────────
        self.head = LinearProbeHead(
            in_dim=in_dim,
            num_classes=self.num_classes,
            normalize=self.normalize_features,
        ).to(self.device)

        self._in_dim = in_dim

    # ── lifecycle ─────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        try:
            self.tap.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    # ── feature extraction ───────────────────────────────────────────────

    @torch.no_grad()
    def _extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run backbone → tap → global avg pool → [B, D]."""
        self.tap.clear()
        _ = self.backbone(imgs.to(self.device))
        feats = self.tap.get_features()
        x = feats[self.feat_level]            # [B, C, H, W]
        x = x.mean(dim=(2, 3))                # GAP → [B, C]
        return x

    # ── single epoch ─────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch on the head. Returns mean loss."""
        self.head.train()
        total_loss = 0.0
        n_batches = 0
        for imgs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = targets.to(self.device).float()

            with torch.no_grad():
                feats = self._extract_features(imgs)

            logits = self.head(feats)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    # ── evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Run head on a full loader, compute multi-label mAP."""
        self.head.eval()
        all_logits = []
        all_targets = []
        for imgs, targets in loader:
            targets = targets.to(self.device).float()
            feats = self._extract_features(imgs)
            logits = self.head(feats)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
        if not all_logits:
            return {"mAP": 0.0, "n_valid_classes": 0,
                    "per_class_ap": torch.zeros(self.num_classes)}
        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return multilabel_average_precision(logits, targets)

    # ── public training API ─────────────────────────────────────────────

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for :meth:`fit` — library-wide consistent entry point.

        Every trainer in yolo-contrastive exposes ``train(...)``; this alias
        lets ``LinearProbeTrainer`` be driven the same way. ``fit`` is kept
        as the primary name (scikit-learn convention for a classification
        probe); both share one implementation.
        """
        return self.fit(*args, **kwargs)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the linear head; return best val_mAP and history.

        Args:
            train_loader, val_loader: DataLoaders.
            epochs: max epoch count.
            lr, weight_decay: AdamW hyperparameters for the head.
            verbose: print per-epoch progress.
            early_stopping_patience: if set, stop training when val_mAP has
                not improved for this many consecutive epochs since the best
                so far. None (default) → no early stopping (full epochs run).

        Returns:
            {
              "best_val_mAP": float,
              "best_epoch": int,
              "final_val_mAP": float,
              "history": [{"epoch":, "train_loss":, "val_mAP":}, ...],
              "early_stopped": bool,        # True if stopped before `epochs`
              "epochs_run": int,            # actual count of completed epochs
            }
        """
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1 or None, "
                f"got {early_stopping_patience}"
            )

        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=lr, weight_decay=weight_decay,
        )
        history: list = []
        best_map = -1.0
        best_epoch = 0
        epochs_since_improvement = 0
        early_stopped = False
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)
            val_map = val_metrics["mAP"]
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mAP": val_map,
                "n_valid_classes": val_metrics["n_valid_classes"],
            })
            improved = val_map > best_map
            if improved:
                best_map = val_map
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if verbose:
                stale = (
                    f" stale={epochs_since_improvement}/{early_stopping_patience}"
                    if early_stopping_patience is not None and not improved
                    else ""
                )
                self._print(
                    f"[ycl-probe] epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | val_mAP={val_map:.4f}{stale}"
                )

            if (
                early_stopping_patience is not None
                and epochs_since_improvement >= early_stopping_patience
            ):
                early_stopped = True
                if verbose:
                    self._print(
                        f"[ycl-probe] early stop at epoch {epoch} "
                        f"(no improvement for {epochs_since_improvement} epochs)"
                    )
                break

        elapsed = time.time() - t0
        epochs_run = len(history)
        if verbose:
            self._print(
                f"[ycl-probe] === Done in {elapsed:.1f}s | "
                f"best mAP={best_map:.4f} @ epoch {best_epoch}"
                f"{' (early stopped)' if early_stopped else ''} ==="
            )

        return {
            "best_val_mAP": best_map,
            "best_epoch": best_epoch,
            "final_val_mAP": history[-1]["val_mAP"] if history else 0.0,
            "history": history,
            "early_stopped": early_stopped,
            "epochs_run": epochs_run,
        }

    # ── helpers ─────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(msg)
        except Exception:
            print(msg)

    def __repr__(self) -> str:
        return (
            f"LinearProbeTrainer(num_classes={self.num_classes}, "
            f"feat_level={self.feat_level!r}, in_dim={self._in_dim}, "
            f"normalize={self.normalize_features})"
        )
