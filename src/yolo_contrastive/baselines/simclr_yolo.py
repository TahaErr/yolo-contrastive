"""SimCLR-YOLO — external SSL baseline (Faz 5.4, WORK_PLAN_v9 §5.4).

A fair-comparison baseline: SimCLR's core mechanism on a YOLOv8n backbone,
trained on the same pool with the same protocol as DT-SAPS. This is what the
paper's Table 4 compares against — not a line-for-line port of the original
SimCLR codebase, but its defining SSL principle applied to our setting.

SimCLR's core mechanism (Chen et al. 2020):
    - one encoder, NO momentum encoder, NO memory queue
    - two augmented views per image
    - global-pooled embedding → projection head
    - NT-Xent contrastive loss with in-batch negatives

Contrast with our DT-SAPS / dense SAPS: SimCLR is global-pooled (one vector
per image) and single-teacher (in-batch negatives only). It has no dense
per-position matching and no multi-scale structure — exactly the axes our
method adds, which is why it makes a clean baseline.

Reuses existing library components: MultiScaleFeatureTap (P5 tap),
ProjectionHead, NTXentLoss, and the simclr_v2 augmentation preset.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..augmentations.presets import build_pipeline
from ..contrastive.losses import NTXentLoss
from ..dense import MultiScaleFeatureTap, infer_in_channels
from ..pretext.heads import ProjectionHead
from ..pretrain.dataset import UnlabeledImageDataset

_VALID_LEVELS = ("P3", "P4", "P5")


class SimCLRYOLOTrainer:
    """SimCLR pretraining on a YOLOv8n backbone (global-pooled, in-batch NT-Xent).

    Args:
        model: backbone — Ultralytics spec str or nn.Module.
        feat_level: FPN level to global-pool for the embedding. Default ``P5``
            (deepest, most semantic — SimCLR's usual single-feature choice).
        out_dim: projection head output dimension. Default 128 (SimCLR paper).
        proj_hidden: projection head hidden width.
        temperature: NT-Xent temperature.
        aug_preset: augmentation preset name (a ``presets.py`` key).
        imgsz: training image size.
        device: torch device; auto-detected if None.
    """

    def __init__(
        self,
        model,
        feat_level: str = "P5",
        out_dim: int = 128,
        proj_hidden: int = 256,
        temperature: float = 0.2,
        aug_preset: str = "simclr_v2",
        imgsz: int = 640,
        device=None,
    ):
        if feat_level not in _VALID_LEVELS:
            raise ValueError(f"feat_level must be one of {_VALID_LEVELS}, got {feat_level!r}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.feat_level = feat_level
        self.out_dim = int(out_dim)
        self.imgsz = int(imgsz)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── backbone ─────────────────────────────────────────────────────
        if isinstance(model, str):
            from ultralytics import YOLO
            self.model = YOLO(model).model.to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        # ── feature tap (single level — SimCLR is global-pooled) ─────────
        self.tap = MultiScaleFeatureTap(self.model, levels=(feat_level,))
        self.tap.setup()
        in_channels = infer_in_channels(
            self.model, self.tap,
            imgsz=min(self.imgsz, 64), device=self.device,
        )
        self.tap.clear()
        self.feat_dim = in_channels[feat_level]

        # ── projection head ──────────────────────────────────────────────
        self.projection_head = ProjectionHead(
            feat_dim=self.feat_dim, out_dim=out_dim, hidden_dim=proj_hidden,
        ).to(self.device)

        # ── loss + augmentation ──────────────────────────────────────────
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.augmentation = build_pipeline(aug_preset, imgsz=self.imgsz)

    # ── embedding ────────────────────────────────────────────────────────

    def _embed(self, view: torch.Tensor) -> torch.Tensor:
        """Backbone → tapped feature → global avg pool → projection → [B, out_dim]."""
        self.tap.clear()
        _ = self.model(view)
        feat = self.tap.get_features()[self.feat_level]   # [B, C, H, W]
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]
        return self.projection_head(pooled)               # [B, out_dim]

    # ── single training step ─────────────────────────────────────────────

    def _step(self, imgs: torch.Tensor) -> Dict:
        """One SimCLR step — two views, in-batch NT-Xent."""
        imgs = imgs.to(self.device, non_blocking=True)
        with torch.no_grad():
            view1 = self.augmentation(imgs)
            view2 = self.augmentation(imgs)
        z1 = self._embed(view1)
        z2 = self._embed(view2)
        loss = self.loss_fn(z1, z2)
        return {"loss": loss, "info": {"loss": float(loss.detach())},
                "batch_size": imgs.shape[0]}

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
        output: str = "simclr_yolo_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
    ) -> str:
        """Run SimCLR pretraining. Returns the saved backbone path."""
        dataset = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        steps_per_epoch = len(dataloader)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = warmup_epochs * steps_per_epoch

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.model.train()
        self.projection_head.train()

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
                    f"[simclr-yolo] epoch {epoch}/{epochs} "
                    f"loss={ep_loss / n_batches:.4f} ({time.time() - t0:.1f}s)"
                )
            if save_every and epoch % save_every == 0:
                self._save(output, epoch)

        self._save(output, epochs)
        return output

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save(self, output: str, epoch: int) -> None:
        """Save the backbone with a SimCLR-YOLO marker."""
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "type": "ssl_pretrained",
            "extra": {"type": "simclr_yolo", "feat_level": self.feat_level},
        }, output)

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
        return (
            f"SimCLRYOLOTrainer(feat_level={self.feat_level}, "
            f"out_dim={self.out_dim}, feat_dim={self.feat_dim})"
        )
