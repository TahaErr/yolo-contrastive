"""RoadReconstructor — standalone from-scratch trainer for the B2 reconstruction net.

Trains :class:`~yolo_contrastive.roadrecon.recon_net.RoadReconNet` by **denoising /
inpainting the road region**: random square patches inside a depth-free road-region
prior (``geoteach.plane_fit.trapezoid_mask``) are masked in the input, and the net
must reconstruct the clean image. Because the masked patches can only be recovered
from surrounding context, the encoder is forced to model normal-road appearance —
so at mining time a *real* anomaly (which never matched the learned normal manifold)
reconstructs poorly and stands out in the error map.

Standalone (not an ultralytics ``BaseTrainer``) for the same reason as
``pretrain.dense_trainer.DenseSSLPretrainer``, whose loop structure (warmup+cosine
LR, AMP, best-epoch save) this mirrors. The trained encoder is saved via
``save_backbone`` → the **M2** representation init.

``ultralytics`` is imported lazily inside :class:`RoadReconNet`; ``geoteach``'s pure
numpy ``trapezoid_mask`` is imported lazily inside :meth:`_road_mask`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .recon_net import RoadReconNet


class _PathListDataset(Dataset):
    """Load RGB images from an explicit list of paths (for --limit trial subsets).

    Mirrors ``pretrain.dataset.UnlabeledImageDataset``'s loading (cv2, resize to a
    square ``imgsz``, RGB float in [0, 1]) but over a caller-supplied path list, so B2
    training and mining can share the exact same seeded subset. cv2 is lazy (E2).
    """

    def __init__(self, paths, imgsz: int) -> None:
        self.paths = [str(p) for p in paths]
        self.imgsz = int(imgsz)
        if not self.paths:
            raise ValueError("_PathListDataset got an empty path list")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import cv2  # lazy
        import numpy as np
        img = cv2.imread(self.paths[idx])
        if img is None:
            return torch.zeros(3, self.imgsz, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


class RoadReconstructor:
    """From-scratch road-reconstruction pretrainer (building block B2).

    Args:
        model: ultralytics spec for the encoder — ``"yolov8n.yaml"`` (scratch; the
            pure setting) or a pre-built module.
        imgsz: square training image size.
        tap_level / decoder_base: forwarded to :class:`RoadReconNet`.
        n_mask_patches: number of square patches masked per image.
        mask_patch_frac: patch side as a fraction of ``min(H, W)``.
        mask_value: fill value for masked pixels (``0.5`` = neutral grey).
        road_region_only: compute the reconstruction loss only over the road-region
            prior (recommended — focuses learning on the drivable surface).
        device: ``"cuda"``, ``"cpu"``, int, or ``None`` (auto).
    """

    def __init__(
        self,
        model: Any = "yolov8n.yaml",
        imgsz: int = 640,
        tap_level: str = "P3",
        decoder_base: int = 128,
        n_mask_patches: int = 4,
        mask_patch_frac: float = 0.12,
        mask_value: float = 0.5,
        road_region_only: bool = True,
        device: Optional[Any] = None,
    ) -> None:
        if n_mask_patches < 0:
            raise ValueError(f"n_mask_patches must be >= 0, got {n_mask_patches}")
        if not 0.0 < mask_patch_frac < 1.0:
            raise ValueError(f"mask_patch_frac must be in (0, 1), got {mask_patch_frac}")
        self.n_mask_patches = int(n_mask_patches)
        self.mask_patch_frac = float(mask_patch_frac)
        self.mask_value = float(mask_value)
        self.road_region_only = bool(road_region_only)
        # config needed to rebuild for a full-net reload (mining / kill-gate)
        self._model_spec = model if isinstance(model, str) else "yolov8n.yaml"
        self._tap_level = tap_level
        self._decoder_base = int(decoder_base)

        self.net = RoadReconNet(
            model=model, imgsz=imgsz, tap_level=tap_level,
            decoder_base=decoder_base, device=device,
        )
        self.imgsz = self.net.imgsz
        self.device = self.net.device

        # Lazily-built road-region prior (bool [H, W]) + its pixel coords.
        self._road_mask_t: Optional[torch.Tensor] = None
        self._road_coords: Optional[torch.Tensor] = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release the net's tap hook. Idempotent."""
        self.net.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    # ── road-region prior ─────────────────────────────────────────────────────

    def _road_mask(self) -> torch.Tensor:
        """Boolean road-region prior ``[H, W]`` on the training device (cached)."""
        if self._road_mask_t is None:
            from ..geoteach.plane_fit import trapezoid_mask  # lazy; pure numpy
            m = trapezoid_mask((self.imgsz, self.imgsz))          # numpy bool [H, W]
            self._road_mask_t = torch.from_numpy(m).to(self.device)
            ys, xs = torch.nonzero(self._road_mask_t, as_tuple=True)
            self._road_coords = torch.stack([ys, xs], dim=1)      # [N, 2] (y, x)
        return self._road_mask_t

    def _corrupt(self, imgs: torch.Tensor) -> torch.Tensor:
        """Mask ``n_mask_patches`` random road-region squares → corrupted copy."""
        self._road_mask()                       # ensure coords built
        corrupted = imgs.clone()
        coords = self._road_coords
        if self.n_mask_patches == 0 or coords is None or coords.shape[0] == 0:
            return corrupted
        B, _, H, W = imgs.shape
        s = max(2, int(self.mask_patch_frac * min(H, W)))
        half = s // 2
        for b in range(B):
            idx = torch.randint(0, coords.shape[0], (self.n_mask_patches,))
            for cy, cx in coords[idx].tolist():
                y0, y1 = max(0, cy - half), min(H, cy + half)
                x0, x1 = max(0, cx - half), min(W, cx + half)
                corrupted[b, :, y0:y1, x0:x1] = self.mask_value
        return corrupted

    # ── one training step (used by tests + train loop) ───────────────────────

    def _step(self, imgs: torch.Tensor) -> dict:
        """One step: mask → reconstruct → masked/road MSE. Returns loss + info."""
        imgs = imgs.to(self.device, non_blocking=True)
        corrupted = self._corrupt(imgs)
        recon = self.net(corrupted)                          # [B, 3, H, W] in [0,1]
        diff = (recon - imgs).pow(2).mean(dim=1)             # [B, H, W]
        if self.road_region_only:
            m = self._road_mask()                            # [H, W] bool
            loss = diff[:, m].mean()
        else:
            loss = diff.mean()
        return {"loss": loss, "batch_size": imgs.shape[0]}

    def error_map(self, imgs: torch.Tensor) -> torch.Tensor:
        """Per-pixel reconstruction error ``[B, H, W]`` (delegates to the net)."""
        return self.net.error_map(imgs)

    # ── full-net save / load (encoder + decoder, for mining / kill-gate) ──────

    def save(self, path: str) -> str:
        """Save the FULL net (encoder + decoder) so mining/kill-gate can reload it.

        Distinct from ``train()``'s ``save_backbone`` (encoder-only, the M2 init):
        the error map needs the trained decoder too. Load with
        :func:`load_reconstructor`.
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net_state": self.net.state_dict(),
            "type": "roadrecon_full",
            "config": {
                "model": self._model_spec, "imgsz": self.imgsz,
                "tap_level": self._tap_level, "decoder_base": self._decoder_base,
                "n_mask_patches": self.n_mask_patches,
                "mask_patch_frac": self.mask_patch_frac,
                "mask_value": self.mask_value,
                "road_region_only": self.road_region_only,
            },
        }, path)
        return path

    # ── public training API ──────────────────────────────────────────────────

    def train(
        self,
        images_dir: Optional[str] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        num_workers: int = 4,
        output: str = "roadrecon_backbone.pt",
        save_every: int = 25,
        print_every: int = 10,
        image_list: Optional[list] = None,
    ) -> str:
        """Run road-reconstruction pretraining; save the best-epoch encoder.

        Args:
            images_dir: pool image directory (recursively globbed). Ignored if
                ``image_list`` is given.
            image_list: explicit list of image paths (e.g. a seeded --limit subset for
                trial runs). Takes precedence over ``images_dir``.

        Returns:
            Path to the saved backbone checkpoint (the **M2** init).
        """
        import copy as _copy

        from ..pretrain.backbone_utils import save_backbone  # lazy (pulls cv2 via pretrain/__init__)
        if image_list is not None:
            dataset = _PathListDataset(image_list, self.imgsz)
        elif images_dir is not None:
            from ..pretrain.dataset import UnlabeledImageDataset  # lazy
            dataset = UnlabeledImageDataset(images_dir, imgsz=self.imgsz)
        else:
            raise ValueError("train() needs either images_dir or image_list")
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"), drop_last=True,
        )
        steps_per_epoch = max(1, len(dataloader))
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        use_amp = self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:  # older torch
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self._print(
            f"[ycl-recon] === Road Reconstruction Pretraining ===\n"
            f"[ycl-recon] epochs={epochs}, batch={batch_size}, lr={lr}, "
            f"tap={self.net.tap_level}, imgsz={self.imgsz}\n"
            f"[ycl-recon] mask: {self.n_mask_patches} patches @ "
            f"{self.mask_patch_frac:.2f} side, road_only={self.road_region_only}\n"
            f"[ycl-recon] dataset: {len(dataset)} imgs, {steps_per_epoch} batches/epoch\n"
            f"[ycl-recon] output: {output}\n"
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        global_step = 0
        t0_total = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.net.train()
            ep_loss, n_batches = 0.0, 0

            for imgs in dataloader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    out = self._step(imgs)
                    loss = out["loss"]
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                ep_loss += float(loss.detach().item())
                n_batches += 1
                global_step += 1

            avg_loss = ep_loss / max(1, n_batches)
            if epoch % print_every == 0 or epoch in (1, epochs):
                self._print(
                    f"[ycl-recon] epoch {epoch:3d}/{epochs} | loss={avg_loss:.5f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s"
                )

            if save_every > 0 and epoch % save_every == 0:
                chkpt = output.replace(".pt", f"_ep{epoch}.pt")
                save_backbone(self.net.encoder, chkpt, epoch=epoch,
                              extra={"loss": avg_loss, "type": "roadrecon"})

            if avg_loss < best_loss:
                best_loss, best_epoch = avg_loss, epoch
                best_state = _copy.deepcopy(self.net.encoder.state_dict())

        if best_state is not None:
            self.net.encoder.load_state_dict(best_state)
        save_backbone(
            self.net.encoder, output, epoch=best_epoch or epochs,
            extra={"loss": best_loss, "type": "roadrecon",
                   "best_epoch": best_epoch or epochs,
                   "tap_level": self.net.tap_level},
        )
        self._print(
            f"[ycl-recon] === Done in {time.time() - t0_total:.1f}s | "
            f"best_loss={best_loss:.5f} @ epoch {best_epoch or epochs} → {output} ==="
        )
        return output

    # ── helpers ──────────────────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(msg)
        except Exception:
            print(msg)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RoadReconstructor(tap={self.net.tap_level}, imgsz={self.imgsz}, "
            f"mask={self.n_mask_patches}@{self.mask_patch_frac:.2f}, device={self.device})"
        )


def load_reconstructor(path: str, device: Optional[Any] = None) -> "RoadReconstructor":
    """Rebuild a :class:`RoadReconstructor` (encoder + decoder) from a ``save()`` file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("type") != "roadrecon_full":
        raise ValueError(f"{path!r} is not a roadrecon_full checkpoint (type={ckpt.get('type')!r})")
    cfg = ckpt["config"]
    r = RoadReconstructor(
        model=cfg["model"], imgsz=cfg["imgsz"], tap_level=cfg["tap_level"],
        decoder_base=cfg["decoder_base"], n_mask_patches=cfg["n_mask_patches"],
        mask_patch_frac=cfg["mask_patch_frac"], mask_value=cfg["mask_value"],
        road_region_only=cfg["road_region_only"], device=device,
    )
    r.net.load_state_dict(ckpt["net_state"])
    return r
