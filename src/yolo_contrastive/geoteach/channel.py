"""TerraChannel — geometry-as-teacher AuxChannel for the anchored joint trainer.

Plugs TERRA Stage 1 into :class:`~yolo_contrastive.anchored.AnchoredJointTrainer`:

    L_terra = L_res + beta * L_geobox

    L_res    — dense 6-way ordinal classification at P3 (stride 8) through a
               2-conv head; pixel labels majority-pooled to stride 8; ordinal
               label smoothing (0.8 true bin, 0.1 each chain neighbor);
               balanced cell sampling (all anomaly cells + 3x F + 1x X per
               anomaly cell, cap 1024 cells/img) so "predict F everywhere" is
               never a shortcut (R2: per-image label maps vary like real
               labels — no static target to exhaust).
    L_geobox — the REAL ultralytics v8DetectionLoss (TAL assigner + CIoU +
               DFL + BCE) on the mined 2-class polarity boxes through a
               SEPARATE fresh Detect head on the same P3/P4/P5 taps; the
               80-class COCO head is untouched (replay-only).

The trainer multiplies the summed terms by ``lambda_aux`` (= lambda_g) and
backprops them in the SAME optimizer step as the COCO replay loss (R3).

R5 — joint augmentation, the hard rule: the pool loader applies ONE sampled
RandomResizedCrop + horizontal flip to the image, the dense label map
(nearest) and the boxes (coordinate math) inside ``__getitem__``. Image and
labels can never desynchronize because a single normalized crop rectangle
parameterizes all three (the documented teacher-cache misalignment bug class
is impossible by construction). Pool batches use RRC + flip + photometric
color jitter only (jitter is applied after geometry and touches pixels
exclusively) — no mosaic.

R6 — zero teacher-side trainables: the depth model never appears here; it ran
offline in Stage 0 (depth_cache.py) and only its LABELS reach this channel.

Heavy deps: ultralytics is imported lazily via heads.py; cv2 only in the
directory-backed dataset path. Synthetic in-memory samples need torch only.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..anchored.channel import AuxChannel, probe_tap_features
from ..dense.multi_scale_tap import YOLOV8_FPN_STRIDES
from ..exceptions import FeatureTapError
from .heads import DenseOrdinalHead, GeoDetectHead, head_channels
from .residual_labels import (
    ANOMALY_CLASSES,
    CLS_F,
    CLS_X,
    IGNORE_LABEL,
    NUM_CLASSES,
    ORDINAL_CHAIN,
)

# ── configs ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class TerraAugConfig:
    """Joint RRC + flip + photometric parameters for the pool loader (R5).

    ``photometric`` is the brightness/contrast/saturation jitter strength
    (the spec's "RRC + flip + color"); it is applied AFTER geometry and
    touches pixels only, so labels/boxes are untouched by construction.
    """

    scale: Tuple[float, float] = (0.33, 1.0)   # crop area fraction
    ratio: Tuple[float, float] = (0.75, 4.0 / 3.0)
    hflip_prob: float = 0.5
    photometric: float = 0.2           # color jitter strength (0 disables)
    min_box_visibility: float = 0.25   # surviving area fraction of a box
    min_box_size: float = 0.01         # min normalized box side after crop


@dataclasses.dataclass
class OrdinalLossConfig:
    """Balanced sampling + ordinal smoothing knobs for L_res."""

    smooth_true: float = 0.8      # mass on the true bin (interior chain bins)
    smooth_neighbor: float = 0.1  # mass on each ordinal chain neighbor
    flat_per_anomaly: int = 3     # F cells sampled per anomaly cell
    x_per_anomaly: int = 1        # X cells sampled per anomaly cell
    max_cells_per_image: int = 1024
    fallback_cells: int = 128     # cells sampled when an image has no anomaly


# ── ordinal smoothing ─────────────────────────────────────────────────────────


def ordinal_smoothing_matrix(cfg: Optional[OrdinalLossConfig] = None) -> torch.Tensor:
    """[NUM_CLASSES, NUM_CLASSES] soft-target rows with ordinal smoothing.

    Chain bins (D2-D1-F-E1-E2) put ``smooth_true`` on the true bin and
    ``smooth_neighbor`` on each existing chain neighbor; mass for a missing
    neighbor (chain endpoints) goes back to the true bin so every row sums to
    1. X is off-chain → one-hot.
    """
    cfg = cfg or OrdinalLossConfig()
    t = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    chain = list(ORDINAL_CHAIN)
    for pos, cls in enumerate(chain):
        row = torch.zeros(NUM_CLASSES)
        row[cls] = cfg.smooth_true
        for npos in (pos - 1, pos + 1):
            if 0 <= npos < len(chain):
                row[chain[npos]] = cfg.smooth_neighbor
            else:
                row[cls] += cfg.smooth_neighbor  # endpoint: mass back to true bin
        t[cls] = row
    t[CLS_X, CLS_X] = 1.0
    return t


# ── label pooling + balanced sampling ─────────────────────────────────────────


def majority_pool_labels(labels: torch.Tensor, out_hw: Tuple[int, int],
                         ignore: int = IGNORE_LABEL) -> torch.Tensor:
    """Majority-pool a [B, H, W] pixel label map to [B, h, w] cells.

    Each stride-8 cell takes the plurality pixel label; cells where the
    plurality is the ignore label stay ignored. Requires H % h == 0.
    """
    if labels.dim() != 3:
        raise ValueError(f"labels must be [B, H, W], got shape {tuple(labels.shape)}")
    b, h_in, w_in = labels.shape
    h, w = int(out_hw[0]), int(out_hw[1])
    if h_in % h or w_in % w:
        raise ValueError(
            f"label map {h_in}x{w_in} not divisible by feature grid {h}x{w} — "
            f"the loader must emit labels at the (32-multiple) train imgsz"
        )
    lab = labels.long().clone()
    lab[lab == ignore] = NUM_CLASSES  # ignore channel
    onehot = F.one_hot(lab, NUM_CLASSES + 1).permute(0, 3, 1, 2).float()
    pooled = F.avg_pool2d(onehot, (h_in // h, w_in // w))
    maj = pooled.argmax(dim=1)
    out = maj.clone()
    out[maj == NUM_CLASSES] = ignore
    return out


def sample_balanced_cells(cell_labels: torch.Tensor,
                          cfg: Optional[OrdinalLossConfig] = None) -> torch.Tensor:
    """Balanced cell indices (flat, over h*w) for ONE image's [h, w] cell map.

    All anomaly cells + ``flat_per_anomaly`` x F + ``x_per_anomaly`` x X per
    anomaly cell, capped at ``max_cells_per_image`` (anomaly cells kept
    preferentially). Images without anomaly cells contribute up to
    ``fallback_cells`` split between F and X, so flat-road geometry still
    teaches road delineation. Ignore cells are never sampled.
    """
    cfg = cfg or OrdinalLossConfig()
    flat = cell_labels.reshape(-1)
    device = flat.device
    anom_mask = torch.zeros_like(flat, dtype=torch.bool)
    for c in ANOMALY_CLASSES:
        anom_mask |= flat == c
    anom_idx = anom_mask.nonzero(as_tuple=True)[0]
    f_idx = (flat == CLS_F).nonzero(as_tuple=True)[0]
    x_idx = (flat == CLS_X).nonzero(as_tuple=True)[0]

    def _take(idx: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or idx.numel() == 0:
            return idx.new_empty(0)
        if k >= idx.numel():
            return idx
        return idx[torch.randperm(idx.numel(), device=device)[:k]]

    n_a = int(anom_idx.numel())
    if n_a > 0:
        anom = _take(anom_idx, cfg.max_cells_per_image)
        budget = cfg.max_cells_per_image - int(anom.numel())
        n_f = min(cfg.flat_per_anomaly * n_a, budget)
        flat_cells = _take(f_idx, n_f)
        budget -= int(flat_cells.numel())
        x_cells = _take(x_idx, min(cfg.x_per_anomaly * n_a, budget))
        return torch.cat([anom, flat_cells, x_cells])
    half = cfg.fallback_cells // 2
    return torch.cat([_take(f_idx, half), _take(x_idx, cfg.fallback_cells - half)])


# ── joint geometric transform (R5) ────────────────────────────────────────────


def joint_crop_flip(
    img: torch.Tensor,
    labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor],
    crop: Tuple[float, float, float, float],
    flip: bool,
    out_size: int,
    min_box_visibility: float = 0.25,
    min_box_size: float = 0.01,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """ONE crop rectangle + flip applied jointly to image, label map and boxes.

    Args:
        img: [3, H, W] float image.
        labels: [Hl, Wl] integer label map (any resolution — e.g. the
            half-res cache) or None.
        boxes: [N, 5] ``(cls, cx, cy, w, h)`` normalized, or None.
        crop: normalized ``(x0, y0, w, h)`` rectangle in [0, 1].
        flip: horizontal flip.
        out_size: square output side (the train imgsz).
        min_box_visibility: drop boxes whose surviving area fraction is lower.
        min_box_size: drop boxes whose post-crop normalized w or h is lower.

    Returns:
        ``(img [3, s, s], labels [s, s] | None, boxes [M, 5] | None)``.

    The image rectangle is snapped to image pixels and the EFFECTIVE
    normalized rectangle is re-derived from those pixels, then applied to the
    label grid and the box coordinates — all three views come from the same
    rectangle, so misalignment is impossible by construction (R5); residual
    error is sub-pixel grid rounding only.
    """
    if img.dim() != 3:
        raise ValueError(f"img must be [3, H, W], got shape {tuple(img.shape)}")
    s = int(out_size)
    _, h_img, w_img = img.shape
    x0, y0, cw, ch = (float(c) for c in crop)

    # Snap to image pixels; re-derive the effective normalized rectangle.
    px0 = min(max(int(round(x0 * w_img)), 0), w_img - 1)
    py0 = min(max(int(round(y0 * h_img)), 0), h_img - 1)
    px1 = min(max(int(round((x0 + cw) * w_img)), px0 + 1), w_img)
    py1 = min(max(int(round((y0 + ch) * h_img)), py0 + 1), h_img)
    ex0, ey0 = px0 / w_img, py0 / h_img
    ew, eh = (px1 - px0) / w_img, (py1 - py0) / h_img

    out_img = F.interpolate(
        img[None, :, py0:py1, px0:px1], size=(s, s),
        mode="bilinear", align_corners=False,
    )[0]
    if flip:
        out_img = torch.flip(out_img, dims=[-1])

    out_labels = None
    if labels is not None:
        if labels.dim() != 2:
            raise ValueError(f"labels must be [H, W], got shape {tuple(labels.shape)}")
        hl, wl = labels.shape
        lx0 = min(max(int(round(ex0 * wl)), 0), wl - 1)
        ly0 = min(max(int(round(ey0 * hl)), 0), hl - 1)
        lx1 = min(max(int(round((ex0 + ew) * wl)), lx0 + 1), wl)
        ly1 = min(max(int(round((ey0 + eh) * hl)), ly0 + 1), hl)
        out_labels = F.interpolate(
            labels[None, None, ly0:ly1, lx0:lx1].float(), size=(s, s), mode="nearest",
        )[0, 0].to(labels.dtype)
        if flip:
            out_labels = torch.flip(out_labels, dims=[-1])

    out_boxes = None
    if boxes is not None:
        if boxes.numel() == 0:
            out_boxes = boxes.reshape(0, 5)
        else:
            cls = boxes[:, 0]
            bx1 = boxes[:, 1] - boxes[:, 3] / 2
            by1 = boxes[:, 2] - boxes[:, 4] / 2
            bx2 = boxes[:, 1] + boxes[:, 3] / 2
            by2 = boxes[:, 2] + boxes[:, 4] / 2
            # into crop coords
            nx1 = ((bx1 - ex0) / ew).clamp(0.0, 1.0)
            ny1 = ((by1 - ey0) / eh).clamp(0.0, 1.0)
            nx2 = ((bx2 - ex0) / ew).clamp(0.0, 1.0)
            ny2 = ((by2 - ey0) / eh).clamp(0.0, 1.0)
            nw, nh = (nx2 - nx1).clamp(min=0), (ny2 - ny1).clamp(min=0)
            orig_area = (boxes[:, 3] * boxes[:, 4]).clamp(min=1e-12)
            vis = (nw * ew) * (nh * eh) / orig_area
            keep = (vis >= min_box_visibility) & (nw >= min_box_size) & (nh >= min_box_size)
            ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
            if flip:
                ncx = 1.0 - ncx
            out_boxes = torch.stack([cls, ncx, ncy, nw, nh], dim=1)[keep]
    return out_img, out_labels, out_boxes


# ── pool dataset + collate ────────────────────────────────────────────────────


class TerraPoolDataset(Dataset):
    """Pool dataset yielding jointly-augmented (image, label map, boxes).

    Two backing modes:
        * ``samples``: in-memory list of dicts with keys ``img`` ([3, H, W]
          float in [0, 1] or uint8), ``labels`` ([Hl, Wl] integer map) and
          optional ``boxes`` ([N, 5] normalized ``cls cx cy w h``) — used by
          tests and small probes.
        * ``root``: the Stage-0 label-factory directory layout::

              root/images/{stem}.(jpg|png)   # pool image
              root/labels/{stem}.png         # uint8 ordinal label map
              root/boxes/{stem}.txt          # mined YOLO txt (optional)

          Images are loaded with cv2 (lazy import) and indexed by the stems
          present in ``labels/`` (only geometry-valid images get label maps).
    """

    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        samples: Optional[Sequence[Dict[str, Any]]] = None,
        root: Optional[str] = None,
        imgsz: int = 512,
        aug: Optional[TerraAugConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        if (samples is None) == (root is None):
            raise ValueError("provide exactly one of samples= or root=")
        if imgsz % 32:
            raise ValueError(f"imgsz must be a multiple of 32, got {imgsz}")
        self.samples = list(samples) if samples is not None else None
        self.root = Path(root) if root is not None else None
        self.imgsz = int(imgsz)
        self.aug = aug or TerraAugConfig()
        self.seed = seed
        self._gen: Optional[torch.Generator] = None  # lazy (not picklable)
        self._index: List[str] = []
        if self.root is not None:
            labels_dir = self.root / "labels"
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"label-factory dir not found: {labels_dir}")
            self._index = sorted(p.stem for p in labels_dir.glob("*.png"))
            if not self._index:
                raise ValueError(f"no label maps under {labels_dir}")

    # pickling across DataLoader workers: drop the live generator
    def __getstate__(self):
        state = dict(self.__dict__)
        state["_gen"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples) if self.samples is not None else len(self._index)

    # ── raw sample loading ────────────────────────────────────────────────

    def _generator(self) -> torch.Generator:
        if self._gen is None:
            self._gen = torch.Generator()
            if self.seed is not None:
                self._gen.manual_seed(int(self.seed))
        return self._gen

    def _load_raw(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.samples is not None:
            rec = self.samples[i]
            img = torch.as_tensor(rec["img"])
            labels = torch.as_tensor(rec["labels"])
            boxes = torch.as_tensor(
                rec.get("boxes") if rec.get("boxes") is not None else
                torch.zeros(0, 5), dtype=torch.float32,
            ).reshape(-1, 5)
        else:
            import cv2  # lazy: optional [pretrain] extra

            stem = self._index[i]
            img_path = next(
                (p for ext in self._IMG_EXTS
                 if (p := self.root / "images" / f"{stem}{ext}").exists()), None,
            )
            if img_path is None:
                raise FileNotFoundError(f"no image for stem {stem!r} under {self.root}/images")
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"cv2.imread failed for {img_path}")
            img = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)
            lab = cv2.imread(str(self.root / "labels" / f"{stem}.png"),
                             cv2.IMREAD_UNCHANGED)
            if lab is None:
                raise IOError(f"cv2.imread failed for labels/{stem}.png")
            if lab.ndim == 3:
                # ultralytics monkeypatches cv2.imread process-wide to always
                # return 3 dims ([H, W, 1] for grayscale) — undo for label maps.
                lab = lab[..., 0]
            labels = torch.from_numpy(lab.astype("uint8"))
            box_path = self.root / "boxes" / f"{stem}.txt"
            rows = []
            if box_path.exists():
                for line in box_path.read_text(encoding="utf-8").splitlines():
                    vals = line.split()
                    if len(vals) >= 5:
                        rows.append([float(v) for v in vals[:5]])
            boxes = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img.float(), labels, boxes

    # ── one jointly-augmented sample ──────────────────────────────────────

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        img, labels, boxes = self._load_raw(i)
        gen = self._generator()
        a = self.aug

        def _u(lo: float, hi: float) -> float:
            return float(torch.empty((), dtype=torch.float64).uniform_(lo, hi, generator=gen))

        area = _u(a.scale[0], a.scale[1])
        aspect = math.exp(_u(math.log(a.ratio[0]), math.log(a.ratio[1])))
        cw = min(math.sqrt(area * aspect), 1.0)
        ch = min(math.sqrt(area / aspect), 1.0)
        x0 = _u(0.0, 1.0 - cw)
        y0 = _u(0.0, 1.0 - ch)
        flip = _u(0.0, 1.0) < a.hflip_prob

        out_img, out_labels, out_boxes = joint_crop_flip(
            img, labels, boxes, (x0, y0, cw, ch), flip, self.imgsz,
            min_box_visibility=a.min_box_visibility, min_box_size=a.min_box_size,
        )
        if a.photometric > 0:
            # brightness/contrast/saturation jitter AFTER geometry — pixels
            # only, labels/boxes untouched (spec: "RRC + flip + color"; R5).
            js = float(a.photometric)
            fb, fc, fs = (_u(1.0 - js, 1.0 + js) for _ in range(3))
            out_img = out_img * fb
            mean = out_img.mean(dim=(-2, -1), keepdim=True)
            out_img = (out_img - mean) * fc + mean
            gray = out_img.mean(dim=-3, keepdim=True)
            out_img = ((out_img - gray) * fs + gray).clamp(0.0, 1.0)
        return {"img": out_img, "labels": out_labels.long(), "boxes": out_boxes}


def terra_collate(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack images/labels; concatenate boxes ultralytics-style.

    Output keys: ``img`` [B, 3, s, s], ``labels`` [B, s, s] long, and the
    v8DetectionLoss trio ``batch_idx`` [N], ``cls`` [N, 1], ``bboxes`` [N, 4].
    """
    imgs = torch.stack([it["img"] for it in items])
    labels = torch.stack([it["labels"] for it in items])
    idx_parts, cls_parts, box_parts = [], [], []
    for bi, it in enumerate(items):
        b = it["boxes"]
        idx_parts.append(torch.full((b.shape[0],), float(bi)))
        cls_parts.append(b[:, 0:1])
        box_parts.append(b[:, 1:5])
    return {
        "img": imgs,
        "labels": labels,
        "batch_idx": torch.cat(idx_parts) if idx_parts else torch.zeros(0),
        "cls": torch.cat(cls_parts) if cls_parts else torch.zeros(0, 1),
        "bboxes": torch.cat(box_parts) if box_parts else torch.zeros(0, 4),
    }


# ── the channel ───────────────────────────────────────────────────────────────


class TerraChannel(AuxChannel):
    """TERRA geometry channel: dense ordinal CE (P3) + geo-box v8DetectionLoss.

    Args:
        samples / root: pool data — exactly one of an in-memory sample list
            or a Stage-0 label-factory directory (see TerraPoolDataset).
        beta: weight on the geo-box term inside this channel
            (``L = ordinal + beta * geobox``); the trainer's ``lambda_aux``
            multiplies the sum.
        ordinal_cfg: balanced-sampling + smoothing knobs.
        aug_cfg: joint RRC/flip + photometric-jitter knobs (R5).
        seed: loader rng seed (tests).

    Loss terms returned per step: ``{"ordinal": ..., "geobox": ...}`` →
    metrics ``terra/ordinal``, ``terra/geobox``, ``terra/total``.
    """

    name = "terra"

    def __init__(
        self,
        samples: Optional[Sequence[Dict[str, Any]]] = None,
        root: Optional[str] = None,
        beta: float = 1.0,
        ordinal_cfg: Optional[OrdinalLossConfig] = None,
        aug_cfg: Optional[TerraAugConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self._samples = samples
        self._root = root
        self.beta = float(beta)
        self.ordinal_cfg = ordinal_cfg or OrdinalLossConfig()
        self.aug_cfg = aug_cfg or TerraAugConfig()
        self.seed = seed
        self.dense_head: Optional[DenseOrdinalHead] = None
        self.geo_head: Optional[GeoDetectHead] = None
        self._smooth = ordinal_smoothing_matrix(self.ordinal_cfg)
        self._dfl_guarded = False

    # ── AuxChannel API ───────────────────────────────────────────────────

    def attach(self, model: nn.Module, taps: Any) -> nn.ModuleList:
        """Build the dense ordinal head (P3) and the fresh 2-class Detect head.

        E5 wrong-layer guard (the same 64 px probe persistence runs): a tap
        hooked on a wrong-stride layer would silently misalign BOTH the dense
        ordinal cells and the geo-box assigner, so the P3/P4/P5 spatial sizes
        are verified against the strides (8/4/2 cells at 64 px) here, at
        construction — :class:`FeatureTapError`, not corrupted supervision.
        """
        probe_px = 64
        feats = probe_tap_features(model, taps, imgsz=probe_px)
        channels: Dict[str, int] = {}
        for level in ("P3", "P4", "P5"):
            if level not in feats:
                raise FeatureTapError(
                    f"terra channel requires a {level!r} tap; got {sorted(feats)}"
                )
            f = feats[level]
            stride = YOLOV8_FPN_STRIDES[level]
            expected = probe_px // stride
            if f.dim() != 4 or f.shape[-2] != expected or f.shape[-1] != expected:
                raise FeatureTapError(
                    f"{level} tap stride check failed: expected [B, C, {expected}, "
                    f"{expected}] for a {probe_px} px probe (stride {stride}), got "
                    f"{tuple(f.shape)} — the tap is hooked on the wrong layer."
                )
            channels[level] = int(f.shape[1])
        self.dense_head = DenseOrdinalHead(channels["P3"], num_classes=NUM_CLASSES)
        self.geo_head = GeoDetectHead(head_channels(channels))
        return nn.ModuleList([self.dense_head, self.geo_head])

    def loss(self, batch: Dict[str, Any], taps: Any) -> Dict[str, torch.Tensor]:
        if self.dense_head is None or self.geo_head is None:
            raise RuntimeError("TerraChannel.loss called before attach()")
        if not self._dfl_guarded:
            # The trainer blanket-enables requires_grad on channel heads after
            # attach; re-freeze the fixed DFL integral conv (E5).
            self.geo_head.freeze_dfl()
            self._dfl_guarded = True

        feats = taps.get_features()
        terms: Dict[str, torch.Tensor] = {}

        labels = batch.get("labels")
        if labels is not None:
            terms["ordinal"] = self._ordinal_loss(feats["P3"], labels)
        if all(k in batch for k in ("batch_idx", "cls", "bboxes")):
            terms["geobox"] = self.beta * self._geobox_loss(feats, batch)
        return terms

    def build_loader(self, cfg: Dict[str, Any]) -> Iterable:
        dataset = TerraPoolDataset(
            samples=self._samples, root=self._root, imgsz=int(cfg["imgsz"]),
            aug=self.aug_cfg, seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_size=min(int(cfg["batch"]), len(dataset)),
            shuffle=True,
            num_workers=int(cfg.get("workers", 0)),
            collate_fn=terra_collate,
            drop_last=False,
        )

    # ── loss terms ───────────────────────────────────────────────────────

    def _ordinal_loss(self, p3: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Balanced-sampled soft-target CE over majority-pooled stride-8 cells."""
        logits = self.dense_head(p3)                       # [B, 6, h, w]
        b, c, h, w = logits.shape
        cells = majority_pool_labels(labels, (h, w))       # [B, h, w]
        idx_parts = []
        for bi in range(b):
            sel = sample_balanced_cells(cells[bi], self.ordinal_cfg)
            if sel.numel():
                idx_parts.append(sel + bi * h * w)
        if not idx_parts:
            return logits.sum() * 0.0  # keep the graph; nothing to supervise
        idx = torch.cat(idx_parts)
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, c)[idx].float()
        flat_labels = cells.reshape(-1)[idx]
        smooth = self._smooth.to(flat_logits.device)
        targets = smooth[flat_labels]
        return -(targets * F.log_softmax(flat_logits, dim=1)).sum(dim=1).mean()

    def _geobox_loss(self, feats: Dict[str, torch.Tensor],
                     batch: Dict[str, Any]) -> torch.Tensor:
        """Real v8DetectionLoss on mined boxes through the fresh 2-class head.

        Same scaling as the replay anchor (the criterion multiplies by batch
        size) so the two detection terms are magnitude-comparable. Batches
        with zero mined boxes still contribute the background cls term.
        """
        preds = self.geo_head([feats["P3"], feats["P4"], feats["P5"]])
        criterion = self.geo_head.build_criterion()
        loss_vec, _items = criterion(preds, batch)
        return loss_vec.sum()
