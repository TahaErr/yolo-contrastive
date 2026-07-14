"""RoadReconNet — own, from-scratch road-reconstruction network (building block B2).

Pure and self-contained: a scratch YOLOv8n backbone (encoder) + a small conv
decoder that reconstructs the input image from a tapped FPN feature map. No
COCO, no Depth-Anything, no external pretrained weights anywhere.

Two roles downstream:
    * the trained encoder is the **M2** representation init (``save_backbone``);
    * the per-pixel reconstruction error localizes road-surface anomalies for
      the **M3** anomaly-mining factory (:mod:`~yolo_contrastive.roadrecon.mining`).

Why reconstruction. Recovering pixels from a lossy, compressed feature map forces
the encoder to *write content into its features* — a structural, label-free
anti-collapse force (the same one GASP's cross-scale reconstruction uses; closest
prior Scale-MAE, Reed 2023). The decoder is deliberately tiny so reconstruction
quality comes from the backbone, not decoder capacity (easy to defend in an
ablation, mirrors ``gasp.reconstruction.ScaleConditionedDecoder``).

The module top imports only torch/numpy; ``ultralytics`` is imported lazily
inside ``__init__`` (E2).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dense import MultiScaleFeatureTap, infer_in_channels
from ..dense.multi_scale_tap import YOLOV8_FPN_STRIDES


def _resolve_device(device: Any) -> torch.device:
    """Mirror the repo's device resolution (DenseSSLPretrainer)."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, (int, float)) or (isinstance(device, str) and str(device).isdigit()):
        return torch.device(f"cuda:{int(device)}")
    return torch.device(device)


def build_scratch_detector(nc: int = 1, cfg: str = "yolov8n.yaml", device: Optional[Any] = None):
    """Build a from-scratch (random-init) ultralytics ``DetectionModel`` with ``nc`` classes.

    The M3 detector: pretrained on ``nc``-class mined boxes (via the anchored replay
    slot) and fine-tuned on the ``nc``-class downstream set — so the head, the mined
    ``data.yaml`` and the downstream ``data.yaml`` must all agree on ``nc``. No COCO
    weights are ever loaded (pure). Pass the returned module to
    ``AnchoredJointTrainer(model=...)``.
    """
    from ultralytics.nn.tasks import DetectionModel  # lazy (E2)
    m = DetectionModel(cfg, nc=int(nc), verbose=False)
    return m.to(_resolve_device(device))


class ReconDecoder(nn.Module):
    """Light conv decoder: a feature map ``[N, C, h, w]`` → an image ``[N, 3, H, W]``.

    Successive ``interpolate(2x) → Conv-BN-ReLU`` blocks (Scale-MAE style, no FiLM
    conditioning), then a final bilinear resize to ``out_size`` and a sigmoid so
    the output lives in ``[0, 1]``. Channels halve each block (floored at 16).
    ``up_steps`` should equal ``log2(stride)`` of the tapped level so the blocks
    roughly restore full resolution before the final resize.

    Args:
        in_channels: channels of the tapped feature map.
        out_size: reconstructed image side (= training image size).
        base: channel width of the first upsample block.
        up_steps: number of 2x upsample blocks.
    """

    def __init__(self, in_channels: int, out_size: int, base: int = 128, up_steps: int = 3):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if up_steps <= 0:
            raise ValueError(f"up_steps must be positive, got {up_steps}")
        self.out_size = int(out_size)
        chs = [int(in_channels)]
        c = int(base)
        for _ in range(up_steps):
            chs.append(max(c, 16))
            c = max(c // 2, 16)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(up_steps)
        ])
        self.head = nn.Conv2d(chs[-1], 3, 3, padding=1)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """[N, C, h, w] → [N, 3, out_size, out_size] in [0, 1]."""
        x = feat_map
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = blk(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


class RoadReconNet(nn.Module):
    """Scratch YOLOv8n encoder + :class:`ReconDecoder` reconstructing the input.

    Args:
        model: ultralytics spec — ``"yolov8n.yaml"`` (scratch / random init; the
            pure production setting), ``"yolov8n.pt"`` (only for quick tests where
            COCO weights are acceptable), or a pre-built ``nn.Module``.
        imgsz: training/inference image size (square).
        tap_level: which FPN level to reconstruct from — ``"P3"`` (stride 8, the
            default; finest, best anomaly localization), ``"P4"`` or ``"P5"``.
        decoder_base: first-block channel width of the decoder.
        device: ``"cuda"``, ``"cpu"``, int, or ``None`` (auto).

    The encoder is exposed as :attr:`encoder` so ``save_backbone(net.encoder, ...)``
    writes a checkpoint whose keys (``model.<idx>...``) match ``load_backbone``.
    """

    def __init__(
        self,
        model: Any = "yolov8n.yaml",
        imgsz: int = 640,
        tap_level: str = "P3",
        decoder_base: int = 128,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if tap_level not in YOLOV8_FPN_STRIDES:
            raise ValueError(
                f"tap_level must be one of {sorted(YOLOV8_FPN_STRIDES)}, got {tap_level!r}"
            )
        self.imgsz = int(imgsz)
        self.tap_level = tap_level
        self.device = _resolve_device(device)

        # ── encoder (scratch YOLO backbone+neck) ────────────────────────────
        if isinstance(model, str):
            from ultralytics import YOLO  # lazy (E2)
            self.encoder = YOLO(model).model.to(self.device)
        else:
            self.encoder = model.to(self.device)
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad = True

        # ── tap the FPN levels ──────────────────────────────────────────────
        # Tap all three P3/P4/P5 (the detect-head auto-map only resolves a single
        # requested level to the DEEPEST index, so ("P3",) alone would silently
        # capture P5). We select ``tap_level`` from the full dict.
        self.tap = MultiScaleFeatureTap(self.encoder)
        self.tap.setup()

        # probe channel width (eval-mode dummy forward; taps cleared after)
        in_ch = infer_in_channels(
            self.encoder, self.tap, imgsz=min(self.imgsz, 64), device=self.device,
        )[tap_level]
        self.tap.clear()

        # ── decoder ─────────────────────────────────────────────────────────
        up_steps = int(round(math.log2(YOLOV8_FPN_STRIDES[tap_level])))
        self.decoder = ReconDecoder(
            in_channels=in_ch, out_size=self.imgsz, base=decoder_base, up_steps=up_steps,
        ).to(self.device)
        self._in_channels = in_ch

    # ── lifecycle ───────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release the tap hook. Idempotent."""
        try:
            self.tap.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    # ── forward paths ─────────────────────────────────────────────────────────

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return the tapped ``tap_level`` feature map (grad-attached)."""
        self.tap.clear()
        _ = self.encoder(imgs)
        return self.tap.get_features()[self.tap_level]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``imgs`` → ``[B, 3, imgsz, imgsz]`` in ``[0, 1]``."""
        return self.decoder(self.encode(imgs))

    @torch.no_grad()
    def error_map(self, imgs: torch.Tensor) -> torch.Tensor:
        """Per-pixel reconstruction error ``[B, H, W]`` (mean squared error over channels).

        Feeds the *clean* image and compares its reconstruction to it: normal road
        reconstructs well (low error); rare surface anomalies reconstruct poorly
        (high error). Used by the mining factory to localize pseudo-anomalies.
        """
        was_training = self.training
        self.eval()
        try:
            imgs = imgs.to(self.device, non_blocking=True)
            recon = self.forward(imgs)
            err = (recon - imgs).pow(2).mean(dim=1)  # [B, H, W]
        finally:
            if was_training:
                self.train()
        return err

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RoadReconNet(tap={self.tap_level}, in_ch={self._in_channels}, "
            f"imgsz={self.imgsz}, device={self.device})"
        )
