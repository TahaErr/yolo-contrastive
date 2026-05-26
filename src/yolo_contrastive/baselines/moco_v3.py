"""MoCo-v3-YOLO — external SSL baseline (Faz 5.4, WORK_PLAN_v9 §5.4).

A fair-comparison baseline: MoCo-v3's core mechanism on a YOLOv8n backbone,
same pool and protocol as DT-SAPS (paper Table 4 "DT-SAPS vs SOTA").

MoCo-v3's core mechanism (Chen et al. 2021), and how it differs from MoCo-v2:
    - momentum (EMA) key encoder — like MoCo-v2
    - NO memory queue — MoCo-v3 dropped it, uses in-batch negatives only
    - prediction head on the query branch (asymmetric, BYOL-style) —
      key branch has projection only, no predictor
    - symmetric InfoNCE: ctr(q1, k2) + ctr(q2, k1)
    - global-pooled (one vector per image), not dense

Contrast with our DT-SAPS / dense SAPS: MoCo-v3 is global-pooled and
single-teacher (its own momentum encoder). It has no dense per-position
matching, no multi-scale structure, and no second supervised teacher —
the axes our method adds.

Reuses existing components: MultiScaleFeatureTap, MomentumEncoder,
ProjectionHead (used for both the projector and the predictor — a 2-layer
MLP with BN), and the simclr_v2 augmentation preset.
"""

from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..augmentations.presets import build_pipeline
from ..dense import MultiScaleFeatureTap, MomentumEncoder, infer_in_channels
from ..pretext.heads import ProjectionHead  # noqa: F401 (kept for API parity)
import torch.nn as nn


class _LNProjectionHead(nn.Module):
    """MoCo-v3 MLP head with LayerNorm instead of BatchNorm.

    The shared ProjectionHead (pretext/heads.py) uses BatchNorm1d. The
    ICLR-2023 MoCo-v3 improvement study found BatchNorm in projection /
    prediction heads introduces representation instability — correlated
    with "random failing in training". MoCo v2 reproduced exactly this:
    a clean ep1-10 descent, then a chaotic ep24-34 rise-and-recover.
    LayerNorm — batch-statistics-free — removes that failure mode.

    This is a MoCo-specific head: ProjectionHead stays BatchNorm so SimCLR
    and the other consumers (which trained stably) are untouched.
    Structure mirrors ProjectionHead: Linear-Norm-ReLU-Linear.
    """

    def __init__(self, feat_dim: int, out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
from ..pretrain.dataset import UnlabeledImageDataset

_VALID_LEVELS = ("P3", "P4", "P5")


def _moco_v3_infonce(q: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:
    """MoCo-v3 InfoNCE — query vs key, in-batch negatives.

    q, k: ``[B, D]``. The positive for query i is key i; all other keys in
    the batch are negatives. k must already be detached.

    The ``2 * temperature`` factor follows the official MoCo-v3 loss scaling.
    """
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    logits = (q @ k.T) / temperature             # [B, B]
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels) * (2.0 * temperature)


class MoCoV3YOLOTrainer:
    """MoCo-v3 pretraining on a YOLOv8n backbone.

    Args:
        model: backbone — Ultralytics spec str or nn.Module.
        feat_level: FPN level to global-pool. Default ``P5``.
        out_dim: projection/prediction output dimension.
        proj_hidden: projector hidden width.
        pred_hidden: predictor hidden width.
        temperature: InfoNCE temperature.
        momentum: EMA coefficient for the key encoder.
        aug_preset: augmentation preset name.
        imgsz: training image size.
        device: torch device; auto-detected if None.
    """

    def __init__(
        self,
        model,
        feat_level: str = "P5",
        out_dim: int = 128,
        proj_hidden: int = 256,
        pred_hidden: int = 256,
        temperature: float = 0.2,
        momentum: float = 0.99,
        aug_preset: str = "simclr_v2",
        imgsz: int = 640,
        device=None,
    ):
        if feat_level not in _VALID_LEVELS:
            raise ValueError(f"feat_level must be one of {_VALID_LEVELS}, got {feat_level!r}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")

        self.feat_level = feat_level
        self.out_dim = int(out_dim)
        self.temperature = float(temperature)
        self.momentum_coef = float(momentum)
        self.imgsz = int(imgsz)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── online backbone ──────────────────────────────────────────────
        if isinstance(model, str):
            from ultralytics import YOLO
            self.model = YOLO(model).model.to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        # ── online tap + channel probe ───────────────────────────────────
        self.online_tap = MultiScaleFeatureTap(self.model, levels=(feat_level,))
        self.online_tap.setup()
        in_channels = infer_in_channels(
            self.model, self.online_tap,
            imgsz=min(self.imgsz, 64), device=self.device,
        )
        self.online_tap.clear()
        self.feat_dim = in_channels[feat_level]

        # ── momentum key encoder + its own tap ───────────────────────────
        # MomentumEncoder strips forward hooks on its deep copy, so the
        # momentum encoder needs its own tap.
        self.momentum = MomentumEncoder(
            self.model, m=self.momentum_coef, force_fp32=True,
        ).to(self.device)
        self.momentum_tap = MultiScaleFeatureTap(
            self.momentum.momentum, levels=(feat_level,),
        )
        self.momentum_tap.setup()

        # ── projection heads — online (trainable) + momentum (EMA copy) ──
        self.proj_online = _LNProjectionHead(
            feat_dim=self.feat_dim, out_dim=out_dim, hidden_dim=proj_hidden,
        ).to(self.device)
        self.proj_momentum = copy.deepcopy(self.proj_online).to(self.device)
        for p in self.proj_momentum.parameters():
            p.requires_grad = False
        self.proj_momentum.eval()

        # ── prediction head — query branch only (asymmetric) ─────────────
        # A 2-layer MLP out_dim→hidden→out_dim; ProjectionHead's structure
        # (Linear-BN-ReLU-Linear) matches the MoCo-v3 predictor.
        self.predictor = _LNProjectionHead(
            feat_dim=out_dim, out_dim=out_dim, hidden_dim=pred_hidden,
        ).to(self.device)

        # ── augmentation ─────────────────────────────────────────────────
        self.augmentation = build_pipeline(aug_preset, imgsz=self.imgsz)

    # ── embeddings ───────────────────────────────────────────────────────

    def _pool(self, tap: MultiScaleFeatureTap) -> torch.Tensor:
        feat = tap.get_features()[self.feat_level]      # [B, C, H, W]
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]

    def _embed_query(self, view: torch.Tensor) -> torch.Tensor:
        """Online branch: backbone → projector → predictor → [B, out_dim]."""
        self.online_tap.clear()
        _ = self.model(view)
        z = self.proj_online(self._pool(self.online_tap))
        return self.predictor(z)

    def _embed_key(self, view: torch.Tensor) -> torch.Tensor:
        """Momentum branch: momentum encoder → momentum projector → detached key."""
        self.momentum_tap.clear()
        with torch.no_grad():
            _ = self.momentum(view)
            k = self.proj_momentum(self._pool(self.momentum_tap))
        return k.detach()

    # ── single training step ─────────────────────────────────────────────

    def _step(self, imgs: torch.Tensor) -> Dict:
        """One MoCo-v3 step — symmetric InfoNCE over two views."""
        imgs = imgs.to(self.device, non_blocking=True)
        with torch.no_grad():
            view1 = self.augmentation(imgs)
            view2 = self.augmentation(imgs)

        q1 = self._embed_query(view1)
        q2 = self._embed_query(view2)
        k1 = self._embed_key(view1)
        k2 = self._embed_key(view2)

        loss = (
            _moco_v3_infonce(q1, k2, self.temperature)
            + _moco_v3_infonce(q2, k1, self.temperature)
        )
        return {"loss": loss, "info": {"loss": float(loss.detach())},
                "batch_size": imgs.shape[0]}

    # ── EMA update ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _ema_update(self) -> None:
        """Update the momentum encoder + momentum projection head."""
        self.momentum.update(self.model)
        m = self.momentum_coef
        for p_o, p_m in zip(self.proj_online.parameters(),
                            self.proj_momentum.parameters()):
            p_m.data.mul_(m).add_(p_o.data, alpha=1.0 - m)
        for b_o, b_m in zip(self.proj_online.buffers(),
                            self.proj_momentum.buffers()):
            if b_m.dtype.is_floating_point:
                b_m.data.mul_(m).add_(b_o.data.to(b_m.dtype), alpha=1.0 - m)
            else:
                b_m.data.copy_(b_o.data)

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
        output: str = "moco_v3_yolo_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
        resume_from: Optional[str] = None,
        gradient_clip: Optional[float] = None,
    ) -> str:
        """Run MoCo-v3 pretraining. Returns the saved backbone path.

        gradient_clip: if set, clip the global grad norm to this value
            after backward, before the optimizer step (torch
            clip_grad_norm_). MoCo-v3 is prone to sudden gradient
            spikes / divergence; clipping bounds them. None = no clipping.

        resume_from: path to a ``.resume.pt`` state file (written every
            ``save_every`` epochs). If given and present, training resumes
            from the next epoch — online backbone, online/momentum
            projectors, predictor, momentum encoder, optimizer and
            loss_history are all restored. Epoch-boundary granularity.
        """
        dataset = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        steps_per_epoch = len(dataloader)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = warmup_epochs * steps_per_epoch

        optimizer = torch.optim.AdamW(
            list(self.model.parameters())
            + list(self.proj_online.parameters())
            + list(self.predictor.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ── resume state (optional) ──────────────────────────────────────
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
            self.model.load_state_dict(rs["model"])
            self.proj_online.load_state_dict(rs["proj_online"])
            self.predictor.load_state_dict(rs["predictor"])
            self.momentum.momentum.load_state_dict(rs["momentum_encoder"])
            self.proj_momentum.load_state_dict(rs["proj_momentum"])
            optimizer.load_state_dict(rs["optimizer"])
            start_epoch = int(rs["epoch"]) + 1
            global_step = int(rs["global_step"])
            loss_history = list(rs.get("loss_history", []))
            _b = rs.get("best", {})
            best_loss = _b.get("loss", float("inf"))
            best_epoch = _b.get("epoch", 0)
            best_state = _b.get("state", None)
            with __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                for _ in range(global_step):
                    scheduler.step()
            print(f"[moco-v3-yolo] RESUMED from epoch {rs['epoch']} "
                  f"→ continuing at epoch {start_epoch}/{epochs}")

        self.model.train()
        self.proj_online.train()
        self.predictor.train()

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            ep_loss = 0.0
            n_batches = 0
            for imgs in dataloader:
                optimizer.zero_grad(set_to_none=True)
                out = self._step(imgs)
                out["loss"].backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in (self.model.parameters(),
                                     self.proj_online.parameters(),
                                     self.predictor.parameters())
                         for p in g],
                        max_norm=gradient_clip,
                    )
                optimizer.step()
                scheduler.step()
                self._ema_update()
                ep_loss += float(out["loss"].detach())
                n_batches += 1
                global_step += 1

            n_batches = max(1, n_batches)
            avg_loss = ep_loss / n_batches
            loss_history.append({
                "epoch": epoch, "loss": avg_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
            if print_every and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"[moco-v3-yolo] epoch {epoch}/{epochs} "
                    f"loss={avg_loss:.4f} ({time.time() - t0:.1f}s)"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

            if save_every and epoch % save_every == 0:
                # resume state FIRST — survives a failure in _save() below
                torch.save({
                    "model": self.model.state_dict(),
                    "proj_online": self.proj_online.state_dict(),
                    "predictor": self.predictor.state_dict(),
                    "momentum_encoder": self.momentum.momentum.state_dict(),
                    "proj_momentum": self.proj_momentum.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                    "loss_history": loss_history,
                    "best": {"loss": best_loss, "epoch": best_epoch,
                             "state": best_state},
                }, resume_path)
                self._save(output.replace(".pt", f"_ep{epoch}.pt"), epoch)

        # final save — best-epoch weights, full loss_history
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._save(output, best_epoch or epochs,
                   loss_history=loss_history, best_epoch=best_epoch or epochs)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        return output

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save(self, output: str, epoch: int,
              loss_history: Optional[list] = None,
              best_epoch: Optional[int] = None) -> None:
        """Save the online backbone with a MoCo-v3-YOLO marker.

        loss_history / best_epoch are passed only by the final save; when
        present they go into ``extra`` so the learning curve survives.
        """
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        extra = {"type": "moco_v3_yolo", "feat_level": self.feat_level}
        if loss_history is not None:
            extra["loss_history"] = loss_history
        if best_epoch is not None:
            extra["best_epoch"] = best_epoch
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "type": "ssl_pretrained",
            "extra": extra,
        }, output)

    # ── lifecycle ────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release tap hooks. Idempotent."""
        for tap in (getattr(self, "online_tap", None),
                    getattr(self, "momentum_tap", None)):
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
            f"MoCoV3YOLOTrainer(feat_level={self.feat_level}, "
            f"out_dim={self.out_dim}, momentum={self.momentum_coef})"
        )
