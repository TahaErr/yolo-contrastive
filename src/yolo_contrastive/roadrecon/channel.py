"""RoadReconChannel — reconstruction aux channel for the anchored joint trainer.

During M3 detection-pretraining the main signal is the mined-pothole detection loss
carried by the trainer's replay slot. This channel adds a **content-pressuring**
auxiliary objective in the same optimizer step: reconstruct the input image from the
detector's own tapped FPN feature map. Recovering pixels from a lossy feature map
forces the backbone to keep road content in its features (anti-collapse), the same
structural force the standalone B2 reconstructor uses — here kept alive *while* the
detector learns to localize.

Contract (see ``anchored.channel.AuxChannel``): ``loss()`` reads the shared tap the
trainer already populated with ONE forward of ``batch["img"]`` — the channel never
forwards the model itself. Its only trainable module is the decoder head, discarded
at export (inference cost stays exactly YOLOv8n).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..anchored.channel import AuxChannel, probe_tap_channels
from ..dense.multi_scale_tap import YOLOV8_FPN_STRIDES
from .recon_net import ReconDecoder


def _recon_collate(batch) -> Dict[str, torch.Tensor]:
    """Stack ``UnlabeledImageDataset`` tensors into the ``{"img": [B,3,H,W]}`` dict
    the anchored trainer expects."""
    return {"img": torch.stack(list(batch), dim=0)}


class RoadReconChannel(AuxChannel):
    """Reconstruction aux channel (content-pressuring regularizer for M3).

    Args:
        images_dir: pool image directory for the channel loader.
        tap_level: FPN level to reconstruct from (must be tapped by the trainer;
            ``"P3"`` / ``"P4"`` / ``"P5"``). Default ``"P3"``.
        imgsz: image size the channel reconstructs at (should match the trainer's
            ``imgsz``).
        decoder_base: first-block width of the decoder head.
        road_region_only: restrict the reconstruction loss to the road-region prior.
    """

    name = "roadrecon"

    def __init__(
        self,
        images_dir: str,
        tap_level: str = "P3",
        imgsz: int = 640,
        decoder_base: int = 128,
        road_region_only: bool = True,
    ) -> None:
        if tap_level not in YOLOV8_FPN_STRIDES:
            raise ValueError(
                f"tap_level must be one of {sorted(YOLOV8_FPN_STRIDES)}, got {tap_level!r}"
            )
        self.images_dir = images_dir
        self.tap_level = tap_level
        self.imgsz = int(imgsz)
        self.decoder_base = int(decoder_base)
        self.road_region_only = bool(road_region_only)
        self.decoder: Optional[ReconDecoder] = None
        self._road_masks: Dict[tuple, torch.Tensor] = {}

    # ── AuxChannel contract ───────────────────────────────────────────────────

    def attach(self, model: nn.Module, taps: Any) -> nn.ModuleList:
        c = probe_tap_channels(model, taps)[self.tap_level]
        up_steps = int(round(math.log2(YOLOV8_FPN_STRIDES[self.tap_level])))
        self.decoder = ReconDecoder(
            in_channels=c, out_size=self.imgsz, base=self.decoder_base, up_steps=up_steps,
        )
        return nn.ModuleList([self.decoder])

    def loss(self, batch: Dict[str, Any], taps: Any) -> Dict[str, torch.Tensor]:
        if self.decoder is None:
            raise RuntimeError("RoadReconChannel.attach must be called before loss().")
        feat = taps.get_features()[self.tap_level]
        img = batch["img"]
        recon = self.decoder(feat)
        if recon.shape[-2:] != img.shape[-2:]:
            recon = F.interpolate(recon, size=img.shape[-2:], mode="bilinear", align_corners=False)
        diff = (recon - img).pow(2).mean(dim=1)          # [B, H, W]
        if self.road_region_only:
            m = self._road_mask(img.shape[-2], img.shape[-1], img.device)
            loss = diff[:, m].mean()
        else:
            loss = diff.mean()
        return {"recon": loss}

    def build_loader(self, cfg: Dict[str, Any]) -> Iterable:
        from ..pretrain.dataset import UnlabeledImageDataset  # lazy (pulls cv2)
        imgsz = int(cfg.get("imgsz", self.imgsz))
        batch = int(cfg["batch"])
        workers = int(cfg.get("workers", 0))
        dataset = UnlabeledImageDataset(self.images_dir, imgsz=imgsz)
        return DataLoader(
            dataset, batch_size=batch, shuffle=True, num_workers=workers,
            drop_last=True, collate_fn=_recon_collate,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _road_mask(self, h: int, w: int, device) -> torch.Tensor:
        """Boolean road-region prior ``[H, W]`` (cached per size, on ``device``)."""
        key = (int(h), int(w), str(device))
        m = self._road_masks.get(key)
        if m is None:
            from ..geoteach.plane_fit import trapezoid_mask  # lazy; pure numpy
            arr = trapezoid_mask((int(h), int(w)))
            m = torch.from_numpy(arr).to(device)
            self._road_masks[key] = m
        return m
