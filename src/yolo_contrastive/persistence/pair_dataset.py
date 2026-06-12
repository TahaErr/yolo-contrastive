"""Manifest-driven torch dataset for REVISIT cross-traversal pairs.

Consumes ONLY parquet manifests + local JPEGs (offline-testable). Each item
is one aligned pair (A, B):

    * both images get ONE independent geometric view each (RRC scale
      (0.5, 1.0) + hflip), expressed as an affine theta;
    * the correspondence grid (Signal A) is re-drawn EVERY epoch from the
      overlap region and pushed through H_norm, then through each view's
      theta — supervision varies per batch like real labels (R2);
    * persistence label maps (Signal B) are rasterized at the P3 grid AFTER
      augmentation from normalized boxes transformed by the SAME theta as the
      image (R5 by construction — no cached label rasters);
    * mild photometric jitter is applied per view after geometry (pixels
      only, coords untouched).

The collate stacks A-views then B-views into one ``img`` tensor so each pair
costs exactly ONE student forward (shared BN statistics, half the activation
memory of two forwards) — the trainer runs that forward and the channel
splits tap features by batch index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .align import h_from_row, points_in_quad, roi_rect, warp_points_h
from .pair_aug import (
    ViewAugConfig,
    apply_theta_to_images,
    photometric_jitter,
    rasterize_label_map,
    sample_view_theta,
    transform_boxes,
    transform_points,
    transform_quad,
    valid_points_mask,
)
from .persistence_labels import (
    LABEL_PERSISTENT,
    LABEL_TRANSIENT,
    PersistenceLabelConfig,
    overlap_quad_for_side,
)

__all__ = ["PairDataset", "collate_pairs"]

_LABEL_CODE = {"persistent": LABEL_PERSISTENT, "transient": LABEL_TRANSIENT}


def _load_image(path) -> torch.Tensor:
    """Load a JPEG as float [3, H, W] in [0, 1] (lazy PIL import, E2)."""
    from PIL import Image  # lazy

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PairDataset(Dataset):
    """Aligned-pair dataset producing jointly-augmented images, points, labels.

    Args:
        pairs: pairs.parquet path or a pre-loaded DataFrame. Only rows with
            ``align_ok == True`` are used (the homography trust gates are the
            actual data filter).
        labels: persistence_labels.parquet path / DataFrame / None (None ->
            Signal B label maps are all-ignore; Signal A still trains).
        imgsz: square view size (multiple of 32; P3 grid = imgsz // 8).
        label_cfg: frozen threshold object shared with the label factory.
        aug: per-view geometric/photometric knobs.
        seed: optional base seed. When set, item i at epoch e is
            deterministic in (seed, i) — pass None for production training
            (fresh randomness, R2).

    Item keys: ``img_a``/``img_b`` [3, S, S]; ``pts`` [2, K, 2] view-normalized
    correspondence coords (side 0 = A, side 1 = B); ``valid`` [K] bool;
    ``labels_a``/``labels_b`` [g, g] int64 (0 bg / 1 persistent / 2 transient
    / 255 ignore).
    """

    def __init__(
        self,
        pairs: Union[str, Path, "Any"],
        labels: Union[str, Path, "Any", None] = None,
        imgsz: int = 512,
        label_cfg: Optional[PersistenceLabelConfig] = None,
        aug: Optional[ViewAugConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        import pandas as pd  # lazy (pretrain extra)

        from . import pair_manifest as pm

        if int(imgsz) % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32, got {imgsz}")
        self.imgsz = int(imgsz)
        self.grid = self.imgsz // 8  # P3 stride
        self.cfg = label_cfg or PersistenceLabelConfig()
        self.aug = aug or ViewAugConfig()
        self.seed = seed

        pairs_df = pairs if isinstance(pairs, pd.DataFrame) else pm.read_pairs(pairs)
        pairs_df = pairs_df[pairs_df["align_ok"] == True]  # noqa: E712
        if len(pairs_df) == 0:
            raise ValueError("PairDataset: no aligned pairs (align_ok) in the manifest")
        self.pairs = pairs_df.reset_index(drop=True)

        if labels is None:
            labels_df = pd.DataFrame(columns=pm.LABELS_COLUMNS)
        elif isinstance(labels, pd.DataFrame):
            labels_df = labels
        else:
            labels_df = pm.read_labels(labels)
        # {(pair_id, side): (boxes [N,4], classes [N])}
        self._labels: Dict[tuple, tuple] = {}
        if len(labels_df):
            for (pid, side), sub in labels_df.groupby(["pair_id", "side"]):
                boxes = sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
                classes = np.array([_LABEL_CODE[v] for v in sub["label"]], dtype=np.int64)
                self._labels[(pid, side)] = (boxes, classes)

    def __len__(self) -> int:
        return len(self.pairs)

    # ── correspondence sampling (original-frame, then per-view transform) ──

    def _sample_correspondences(self, h_norm: np.ndarray, quad_a: np.ndarray,
                                rng: np.random.Generator):
        """Sample oversampled (x_a, x_b) candidates in A's ROI ∩ overlap."""
        cfg = self.cfg
        m = cfg.corr_k * cfg.corr_oversample
        rx1, ry1, rx2, ry2 = roi_rect(cfg.roi_bottom_frac)
        x_a = np.stack(
            [rng.uniform(rx1, rx2, size=m), rng.uniform(ry1, ry2, size=m)], axis=1
        )
        keep = points_in_quad(x_a, quad_a)
        x_a = x_a[keep]
        x_b = warp_points_h(h_norm, x_a)
        lo, hi = cfg.coord_margin, 1.0 - cfg.coord_margin
        ok = (
            np.all((x_a >= lo) & (x_a <= hi), axis=1)
            & np.all(np.isfinite(x_b), axis=1)
            & np.all((x_b >= lo) & (x_b <= hi), axis=1)
        )
        return x_a[ok], x_b[ok]

    # ── one item ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.pairs.iloc[int(idx)]
        rng = (np.random.default_rng() if self.seed is None
               else np.random.default_rng((self.seed, int(idx))))
        cfg = self.cfg
        h_norm = h_from_row(row)
        quad_a = overlap_quad_for_side(h_norm, "a", cfg)
        quad_b = overlap_quad_for_side(h_norm, "b", cfg)

        img_a = _load_image(row["path_a"])
        img_b = _load_image(row["path_b"])

        # original-frame correspondence candidates (re-drawn every epoch, R2)
        x_a, x_b = self._sample_correspondences(h_norm, quad_a, rng)

        # ONE independent geometric view per image (R5: same theta for
        # image, points, boxes, quads below)
        theta_a = sample_view_theta(rng, self.aug)
        theta_b = sample_view_theta(rng, self.aug)
        s = self.imgsz
        view_a = apply_theta_to_images(
            img_a[None], torch.from_numpy(theta_a)[None], (s, s))[0]
        view_b = apply_theta_to_images(
            img_b[None], torch.from_numpy(theta_b)[None], (s, s))[0]
        view_a = photometric_jitter(view_a, self.aug.photometric, rng)
        view_b = photometric_jitter(view_b, self.aug.photometric, rng)

        # points -> view coords; re-validate AFTER augmentation
        k = cfg.corr_k
        pts = np.full((2, k, 2), 0.5, dtype=np.float32)
        valid = np.zeros(k, dtype=bool)
        if x_a.shape[0] > 0:
            pa = transform_points(theta_a, x_a)
            pb = transform_points(theta_b, x_b)
            ok = (valid_points_mask(pa, cfg.coord_margin)
                  & valid_points_mask(pb, cfg.coord_margin))
            sel = np.flatnonzero(ok)[:k]
            n = len(sel)
            pts[0, :n] = pa[sel]
            pts[1, :n] = pb[sel]
            valid[:n] = True

        # dense persistence label maps at the P3 grid (rasterized AFTER aug)
        label_maps = []
        for side, theta, quad in (("a", theta_a, quad_a), ("b", theta_b, quad_b)):
            boxes, classes = self._labels.get(
                (row["pair_id"], side), (np.zeros((0, 4)), np.zeros(0, dtype=np.int64)))
            roi_view = transform_boxes(
                theta, np.array([roi_rect(cfg.roi_bottom_frac)]))[0]
            lm = rasterize_label_map(
                self.grid,
                transform_boxes(theta, boxes),
                classes,
                transform_quad(theta, quad),
                roi_view,
                rng,
                bg_cap_ratio=cfg.bg_cap_ratio,
                ignore_index=cfg.ignore_index,
                erode_cells=cfg.erode_cells,
                dilate_cells=cfg.dilate_cells,
                bg_floor=cfg.bg_floor,
            )
            label_maps.append(torch.from_numpy(lm))

        return {
            "img_a": view_a,
            "img_b": view_b,
            "pts": torch.from_numpy(pts),
            "valid": torch.from_numpy(valid),
            "labels_a": label_maps[0],
            "labels_b": label_maps[1],
        }


def collate_pairs(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack pair items into one channel batch.

    ``img`` is [2B, 3, S, S] with ALL A-views first, then all B-views, so the
    trainer's single forward serves both branches; ``labels`` follows the same
    order. ``pts`` [B, 2, K, 2] and ``valid`` [B, K] stay per-pair.
    """
    img = torch.stack([it["img_a"] for it in items] + [it["img_b"] for it in items])
    labels = torch.stack(
        [it["labels_a"] for it in items] + [it["labels_b"] for it in items]
    ).long()
    pts = torch.stack([it["pts"] for it in items]).float()
    valid = torch.stack([it["valid"] for it in items]).bool()
    return {"img": img, "pts": pts, "valid": valid, "labels": labels}
