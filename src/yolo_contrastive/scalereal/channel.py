"""ScaleRealChannel — GASP-Real as an AuxChannel on the AnchoredJointTrainer.

ONLINE COMPUTATION: one full-image student forward per pool batch (run by the
trainer; the channel never forwards the model inside ``loss()``). Patch
descriptors are RoIAlign-pooled from the P4 tap; a per-patch scalar
"log apparent-scale" head supervises pair DIFFERENCES with smooth-L1
(scale-EQUIVARIANCE) while a positive-only SimSiam predictor/stop-grad cosine
term pulls matched patches' content projections together (scale-INVARIANCE).

Channel batch contract (produced by :func:`scalereal_collate` / the loader,
R5: all spatial labels were pushed through the SAME affine as the image):

    img            [B, 3, S, S]  float in [0, 1]
    pair_batch_idx [M]           long — image index of each pair
    boxes_a        [M, 4]        view-NORMALIZED xyxy of patch A
    boxes_b        [M, 4]        view-NORMALIZED xyxy of patch B
    log_r          [M]           float labels log(Z_A / Z_B)
    aug_theta      [B, 2, 3]     the exact affine used on each image
    image_id       list[str]     (passthrough)

Runtime guards (cross-team interface risks made loud):
    * tap stride asserted by shape ratio at the first batch (the documented
      silent-wrong-layer bug class),
    * ``aug_theta`` aspect-distortion bound (anisotropic scaling is the one
      aug that corrupts log_r) via pair_transform.assert_aspect_ok.

Loss terms returned to the trainer: ``l_scale`` and ``l_inv`` (the latter
pre-multiplied by ``lambda_inv``); the trainer applies the channel weight
``lambda_aux`` and backprops in the SAME optimizer step as the COCO replay
loss (R3). Diagnostics (n_pairs, sign_acc, pred_std, ...) live in
``self.last_logs`` — they must NOT enter the returned dict (the trainer sums
all values into the loss).

EXPORT (R8): all four heads are ~370K params of student-side scaffolding,
registered with the trainer (head-LR group + EMA) but NEVER submodules of
``model`` — they are discarded at export by construction.

Sentinels: :meth:`epoch_sentinels` evaluates a FIXED probe-pair batch from
the reserved 1% image holdout — probe smooth-L1, sign accuracy, Spearman,
prediction std (collapse flag < 0.05) and TWO SHORTCUT PROBES: R^2 of a
row-coordinate-only and of a box-size-only linear regressor vs R^2 of the
head (the head must beat the better baseline by epoch 4 or the channel is
flagged). The shared AnchoredJointTrainer invokes this automatically each
epoch through :meth:`on_epoch_end` (R9, structural); manual run loops may
still call ``epoch_sentinels(epoch)`` directly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..anchored.channel import AuxChannel, probe_tap_channels
from .config import ScaleRealConfig
from .heads import ContentProjector, PatchDescriptor, Predictor, ScaleHead
from .losses import (
    connected_zero,
    content_consistency_loss,
    scale_pair_loss,
    spearman_corr,
)
from .pair_transform import (
    assert_aspect_ok,
    boxes_to_padded,
    identity_theta,
    letterbox_to_square,
    sample_rrc_theta,
    transform_boxes_theta,
    filter_transformed_pairs,
    warp_images,
)

LOG = logging.getLogger(__name__)


# ── dataset + collate (module-level: picklable for DataLoader workers) ───────


class ScaleRealPoolDataset(torch.utils.data.Dataset):
    """Pool images + mined pairs, jointly augmented (R5 by construction).

    Each item letterboxes the native image to a square canvas (pure padding,
    no aspect change), samples ONE RRC+hflip theta constrained to the content
    region, warps the image with it and pushes BOTH boxes of every pair
    through the SAME theta — image/label misalignment is impossible by
    construction. Pairs clipped > ``max_clip_frac``, scaled below
    ``min_patch_px`` or left without a partner are dropped.

    Args:
        records: list of ``{"image_id", "path"}`` dicts (manifest subset).
        pair_index: :class:`~.pair_manifest.PairIndex`.
        imgsz: view side (multiple of 32 for YOLO).
        cfg: thresholds.
        augment: False -> identity theta (the deterministic probe path).
        seed: optional determinism for theta sampling.
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        pair_index,
        imgsz: int,
        cfg: Optional[ScaleRealConfig] = None,
        augment: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.pair_index = pair_index
        self.imgsz = int(imgsz)
        self.cfg = cfg or ScaleRealConfig()
        self.augment = bool(augment)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> torch.Tensor:
        from PIL import Image  # lazy (E2)

        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cfg = self.cfg
        rec = self.records[idx]
        image_id = rec["image_id"]
        img = self._load_image(rec["path"])
        padded, content = letterbox_to_square(img)

        gen = None
        if self.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self.seed + idx)
        if self.augment:
            theta = sample_rrc_theta(
                1, scale=cfg.rrc_scale, ratio=cfg.rrc_ratio,
                hflip_prob=cfg.hflip_prob, content_box=content, generator=gen,
            )
        else:
            theta = identity_theta(1)
        view = warp_images(padded.unsqueeze(0), theta, self.imgsz)[0]

        targets = self.pair_index.prepare_targets(image_id)
        if targets is None:
            boxes_a = torch.zeros(0, 4)
            boxes_b = torch.zeros(0, 4)
            log_r = torch.zeros(0)
        else:
            pa = torch.from_numpy(boxes_to_padded(targets["boxes_a"], content))
            pb = torch.from_numpy(boxes_to_padded(targets["boxes_b"], content))
            lr = torch.as_tensor(targets["log_r"], dtype=torch.float32)
            va = transform_boxes_theta(pa, theta[0])
            vb = transform_boxes_theta(pb, theta[0])
            kept = filter_transformed_pairs(
                va, vb, lr, self.imgsz,
                max_clip_frac=cfg.max_clip_frac, min_patch_px=cfg.min_patch_px,
            )
            boxes_a, boxes_b, log_r = kept["boxes_a"], kept["boxes_b"], kept["log_r"]

        return {
            "img": view,
            "boxes_a": boxes_a,
            "boxes_b": boxes_b,
            "log_r": log_r,
            "theta": theta[0],
            "image_id": image_id,
        }


def scalereal_collate(items: List[Dict[str, Any]],
                      max_pairs: int = 256) -> Dict[str, Any]:
    """Stack images; flatten pairs with per-pair image indices; cap pairs.

    Pairs surviving augmentation are capped at ``max_pairs`` per batch by
    UNIFORM subsampling (never sorted/truncated — that would skew the log_r
    distribution the miner stratified).
    """
    imgs = torch.stack([it["img"] for it in items])
    thetas = torch.stack([it["theta"] for it in items])
    idx_list, ba_list, bb_list, lr_list = [], [], [], []
    for b, it in enumerate(items):
        m = it["log_r"].shape[0]
        if m:
            idx_list.append(torch.full((m,), b, dtype=torch.long))
            ba_list.append(it["boxes_a"])
            bb_list.append(it["boxes_b"])
            lr_list.append(it["log_r"])
    if idx_list:
        pair_batch_idx = torch.cat(idx_list)
        boxes_a = torch.cat(ba_list)
        boxes_b = torch.cat(bb_list)
        log_r = torch.cat(lr_list)
    else:
        pair_batch_idx = torch.zeros(0, dtype=torch.long)
        boxes_a = torch.zeros(0, 4)
        boxes_b = torch.zeros(0, 4)
        log_r = torch.zeros(0)
    m = log_r.shape[0]
    if m > max_pairs:
        keep = torch.randperm(m)[:max_pairs]
        pair_batch_idx = pair_batch_idx[keep]
        boxes_a, boxes_b, log_r = boxes_a[keep], boxes_b[keep], log_r[keep]
    return {
        "img": imgs,
        "pair_batch_idx": pair_batch_idx,
        "boxes_a": boxes_a,
        "boxes_b": boxes_b,
        "log_r": log_r,
        "aug_theta": thetas,
        "image_id": [it["image_id"] for it in items],
    }


class _CollateWithCap:
    """Picklable collate wrapper (Windows DataLoader workers, E5 path hygiene)."""

    def __init__(self, max_pairs: int) -> None:
        self.max_pairs = int(max_pairs)

    def __call__(self, items):
        return scalereal_collate(items, max_pairs=self.max_pairs)


# ── the channel ───────────────────────────────────────────────────────────────


class ScaleRealChannel(AuxChannel):
    """GASP-Real natural-scale supervision as an anchored-trainer channel.

    Args:
        pairs_path: mined pair parquet (pair_manifest schema). Either this or
            ``pair_index`` is required for ``build_loader``.
        pool_manifest: SSL-pool manifest parquet (or DataFrame) providing
            ``image_id -> materialized_path``; required for ``build_loader``.
        config: :class:`ScaleRealConfig` (defaults to the calibrated spec).
        pair_index: prebuilt :class:`~.pair_manifest.PairIndex` (overrides
            ``pairs_path``; used by tests).
        p4_channels: explicit P4 width — skips the probe forward in
            ``attach`` (stub models in tests).
        loader_seed: deterministic theta sampling in the loader (tests).
    """

    name = "scalereal"
    #: tap levels this channel consumes from the shared MultiScaleFeatureTap.
    requires_taps = ("P4",)

    def __init__(
        self,
        pairs_path: Optional[str] = None,
        pool_manifest: Optional[Any] = None,
        config: Optional[ScaleRealConfig] = None,
        pair_index: Optional[Any] = None,
        p4_channels: Optional[int] = None,
        loader_seed: Optional[int] = None,
    ) -> None:
        self.cfg = config or ScaleRealConfig()
        self.pairs_path = pairs_path
        self.pool_manifest = pool_manifest
        self._pair_index = pair_index
        self._p4_channels = p4_channels
        self.loader_seed = loader_seed

        self.descriptor: Optional[PatchDescriptor] = None
        self.scale_head: Optional[ScaleHead] = None
        self.projector: Optional[ContentProjector] = None
        self.predictor: Optional[Predictor] = None

        self._model: Optional[nn.Module] = None
        self._taps: Optional[Any] = None
        self._stride_checked = False
        self._probe_batch: Optional[Dict[str, Any]] = None
        self._probe_build_failed = False
        self._loader_imgsz: Optional[int] = None
        self._epoch = 0

        self.last_logs: Dict[str, float] = {}
        self.sentinel_records: List[Dict[str, float]] = []

    # ── AuxChannel protocol ───────────────────────────────────────────────

    def attach(self, model: nn.Module, taps: Any) -> nn.ModuleList:
        """Build the four student heads sized from the live P4 tap width.

        C4 is inferred at runtime (never hardcoded — 128 for yolov8n); the
        heads are returned for the trainer's head-LR group + EMA and are NOT
        registered on ``model`` (R8: nothing leaks into the export)."""
        cfg = self.cfg
        if self._p4_channels is not None:
            c4 = int(self._p4_channels)
        else:
            c4 = probe_tap_channels(model, taps)["P4"]
        self.descriptor = PatchDescriptor(
            c4, output_size=cfg.roi_output_size,
            hidden=cfg.descriptor_hidden, out_dim=cfg.descriptor_dim,
        )
        self.scale_head = ScaleHead(cfg.descriptor_dim, hidden=cfg.scale_hidden)
        self.projector = ContentProjector(cfg.descriptor_dim, out_dim=cfg.proj_dim)
        self.predictor = Predictor(cfg.proj_dim)
        self._model, self._taps = model, taps
        heads = nn.ModuleList(
            [self.descriptor, self.scale_head, self.projector, self.predictor]
        )
        for p in heads.parameters():
            p.requires_grad_(True)  # E5: never inherit ships-frozen ambiguity
        return heads

    def loss(self, batch: Dict[str, Any], taps: Any) -> Dict[str, torch.Tensor]:
        """One channel step: pooled descriptors -> scale + invariance terms.

        The trainer already ran the single full-image forward; ``taps`` holds
        THIS batch's features. Returns ``{"l_scale", "l_inv"}`` (l_inv
        pre-weighted by ``lambda_inv``); diagnostics go to ``last_logs``.
        """
        p4 = taps.get_features()["P4"]
        self._check_contract(batch, p4)
        out = self._compute(p4, batch)
        self.last_logs = out["logs"]
        return {"l_scale": out["l_scale"], "l_inv": out["l_inv"]}

    def build_loader(self, cfg: Dict[str, Any]) -> Iterable:
        """DataLoader over training-eligible pool images (probe excluded)."""
        index = self._get_pair_index()
        records = self._eligible_records(index)
        if not records:
            raise ValueError(
                "scalereal: no training-eligible images — check pairs_path / "
                "pool_manifest overlap and the min_pairs_per_image gate"
            )
        self._loader_imgsz = int(cfg["imgsz"])  # lets on_epoch_end lazy-build the probe
        dataset = ScaleRealPoolDataset(
            records, index, imgsz=cfg["imgsz"], cfg=self.cfg,
            augment=True, seed=self.loader_seed,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["batch"],
            shuffle=True,
            num_workers=cfg.get("workers", 0),
            collate_fn=_CollateWithCap(self.cfg.max_pairs_per_batch),
            drop_last=False,
        )

    # ── core computation ──────────────────────────────────────────────────

    def _check_contract(self, batch: Dict[str, Any], p4: torch.Tensor) -> None:
        img = batch["img"]
        if not self._stride_checked:
            stride = img.shape[-1] / p4.shape[-1]
            exp = float(self.cfg.expected_stride)
            if not (0.95 * exp <= stride <= 1.05 * exp):
                raise ValueError(
                    f"scalereal: P4 tap stride {stride:.2f} != expected {exp:.0f} "
                    f"(img {img.shape[-1]} px -> feature {p4.shape[-1]}) — wrong tap "
                    f"layer? (documented silent-wrong-layer bug class)"
                )
            self._stride_checked = True
        theta = batch.get("aug_theta")
        if theta is not None:
            assert_aspect_ok(theta, self.cfg.max_aspect_distortion)

    def _connected_zero(self, p4: torch.Tensor) -> torch.Tensor:
        refs: List[torch.Tensor] = [p4]
        for head in (self.descriptor, self.scale_head):
            if head is not None:
                refs.extend(p for p in head.parameters())
        return connected_zero(refs)

    def _descriptors(
        self, p4: torch.Tensor, batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """RoIAlign both patch sets in ONE call; split A/B afterwards."""
        img = batch["img"]
        m = int(batch["log_r"].shape[0])
        scale = p4.shape[-1] / img.shape[-1]  # == 1/16 under the stride assert
        idx = batch["pair_batch_idx"].to(p4.device).float().unsqueeze(1)
        px = float(img.shape[-1])
        rois = torch.cat([
            torch.cat([idx, batch["boxes_a"].to(p4.device) * px], dim=1),
            torch.cat([idx, batch["boxes_b"].to(p4.device) * px], dim=1),
        ], dim=0)  # [2M, 5]
        phi = self.descriptor(p4, rois, spatial_scale=scale)
        return {"phi_a": phi[:m], "phi_b": phi[m:]}

    def _compute(self, p4: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        log_r = batch["log_r"]
        m = int(log_r.shape[0])
        if m == 0:
            zero = self._connected_zero(p4)
            return {
                "l_scale": zero,
                "l_inv": zero * 0.0,
                "logs": {"n_pairs": 0, "l_scale": 0.0, "l_inv": 0.0,
                         "sign_acc": 0.0, "pred_std": 0.0},
            }
        if m > cfg.max_pairs_per_batch:  # belt-and-braces (collate already caps)
            keep = torch.randperm(m, device=log_r.device)[: cfg.max_pairs_per_batch]
            batch = dict(batch)
            for k in ("pair_batch_idx", "boxes_a", "boxes_b", "log_r"):
                batch[k] = batch[k][keep]
            log_r = batch["log_r"]
            m = int(log_r.shape[0])

        d = self._descriptors(p4, batch)
        s_a = self.scale_head(d["phi_a"])
        s_b = self.scale_head(d["phi_b"])
        scale_out = scale_pair_loss(s_a, s_b, log_r.to(s_a.dtype),
                                    beta=cfg.smooth_l1_beta)

        z_a = self.projector(d["phi_a"])
        z_b = self.projector(d["phi_b"])
        q_a = self.predictor(z_a)
        q_b = self.predictor(z_b)
        inv_out = content_consistency_loss(q_a, q_b, z_a, z_b)

        l_scale = scale_out["loss"]
        l_inv = cfg.lambda_inv * inv_out["loss"]
        return {
            "l_scale": l_scale,
            "l_inv": l_inv,
            "logs": {
                "n_pairs": m,
                "l_scale": float(l_scale.detach()),
                "l_inv": float(l_inv.detach()),
                "sign_acc": scale_out["sign_acc"],
                "pred_std": scale_out["pred_std"],
            },
        }

    # ── data plumbing ─────────────────────────────────────────────────────

    def _get_pair_index(self):
        if self._pair_index is None:
            if self.pairs_path is None:
                raise ValueError("scalereal: pairs_path (or pair_index) is required")
            from .pair_manifest import PairIndex  # lazy: pandas (E2)

            self._pair_index = PairIndex.from_parquet(self.pairs_path)
        return self._pair_index

    def _manifest_paths(self) -> Dict[str, str]:
        if self.pool_manifest is None:
            raise ValueError("scalereal: pool_manifest is required to build loaders")
        import pandas as pd  # lazy (E2)

        df = (self.pool_manifest if isinstance(self.pool_manifest, pd.DataFrame)
              else pd.read_parquet(self.pool_manifest))
        return dict(zip(df["image_id"].astype(str), df["materialized_path"].astype(str)))

    def _eligible_records(self, index) -> List[Dict[str, str]]:
        paths = self._manifest_paths()
        ids = index.eligible_image_ids(
            min_pairs=self.cfg.min_pairs_per_image,
            exclude_probe=True,
            probe_fraction=self.cfg.probe_fraction,
        )
        return [{"image_id": i, "path": paths[i]} for i in ids if i in paths]

    # ── sentinels ─────────────────────────────────────────────────────────

    def on_epoch_end(self, epoch: int) -> Dict[str, float]:
        """R9 trainer hook: run :meth:`epoch_sentinels` on the fixed probe set.

        Called automatically by ``AnchoredJointTrainer.train`` (logged as
        ``sentinel/scalereal/*``). If no probe batch was installed yet, one is
        built lazily from the 1% holdout at the loader's imgsz; a failed build
        logs a warning once and the hook returns {} (sentinel ABSENCE is then
        visible in the run log, but a probe-data problem never kills a long
        training run).
        """
        if self._probe_batch is None and not self._probe_build_failed and \
                self._loader_imgsz is not None:
            try:
                self.build_probe_batch(self._loader_imgsz)
            except Exception as exc:  # noqa: BLE001 — sentinel infra must not kill the run
                self._probe_build_failed = True
                LOG.warning(
                    "scalereal: probe-batch build failed (%s) — channel "
                    "sentinels disabled for this run", exc,
                )
        return self.epoch_sentinels(epoch)

    def set_probe_batch(self, batch: Dict[str, Any]) -> None:
        """Install a FIXED probe-pair batch (same schema as training batches)."""
        self._probe_batch = batch

    def build_probe_batch(self, imgsz: int) -> Optional[Dict[str, Any]]:
        """Assemble the fixed probe batch from the 1% holdout (identity aug)."""
        index = self._get_pair_index()
        paths = self._manifest_paths()
        ids = [i for i in index.probe_image_ids(self.cfg.probe_fraction) if i in paths]
        if not ids:
            return None
        records = [{"image_id": i, "path": paths[i]} for i in ids]
        dataset = ScaleRealPoolDataset(records, index, imgsz=imgsz, cfg=self.cfg,
                                       augment=False)
        items, n_pairs = [], 0
        for k in range(len(dataset)):
            it = dataset[k]
            if it["log_r"].shape[0] == 0:
                continue
            items.append(it)
            n_pairs += int(it["log_r"].shape[0])
            if n_pairs >= self.cfg.probe_pairs:
                break
        if not items:
            return None
        self._probe_batch = scalereal_collate(items, max_pairs=self.cfg.probe_pairs)
        return self._probe_batch

    @torch.no_grad()
    def epoch_sentinels(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Per-epoch channel sentinels on the fixed probe-pair set.

        Returns (and appends to ``sentinel_records``)::

            probe_smooth_l1, sign_acc, spearman, pred_std,
            r2_head, r2_row, r2_size, row_probe_delta_r2,
            flag_pred_collapse, flag_row_shortcut

        TWO SHORTCUT PROBES — GASP-Real's analogues of the blur shortcut:

        * row baseline (``r2_row``): in road scenes depth correlates with
          image row, so a row-coordinate-only linear regressor is trivially
          informative;
        * size baseline (``r2_size``): under exact pinhole geometry the box
          size ratio alone already predicts log_r for matched same-physical-
          size content, so a ``[1, log(side_a), log(side_b)]`` regressor on
          the probe boxes is a second free lunch.

        The head must beat ``max(r2_row, r2_size)`` by epoch
        ``cfg.row_probe_deadline_epoch`` (``row_probe_delta_r2`` = head R^2
        minus that max) or ``flag_row_shortcut`` fires. Warnings
        (RuntimeWarning) are emitted on flags; aborting is left to the
        caller's policy.
        """
        self._epoch = int(epoch) if epoch is not None else self._epoch + 1
        if self._probe_batch is None or self._model is None or self._taps is None:
            return {}
        if self.descriptor is None:
            return {}
        batch = self._probe_batch
        model, taps = self._model, self._taps
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        was_training = model.training
        model.eval()
        taps.clear()
        try:
            img = batch["img"].to(device)
            _ = model(img)
            p4 = taps.get_features()["P4"]
            dev_batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            dev_batch["img"] = img
            d = self._descriptors(p4, dev_batch)
            s_a = self.scale_head(d["phi_a"]).float()
            s_b = self.scale_head(d["phi_b"]).float()
        finally:
            taps.clear()
            if was_training:
                model.train()

        log_r = batch["log_r"].to(s_a.device).float()
        pred = (s_b - s_a)
        m = int(log_r.shape[0])
        probe_l1 = float(F.smooth_l1_loss(pred, log_r, beta=self.cfg.smooth_l1_beta))
        sign_acc = float((torch.sign(pred) == torch.sign(log_r)).float().mean())
        pred_std = float(pred.std(unbiased=False)) if m > 1 else 0.0
        rho = spearman_corr(pred, log_r)

        ss_tot = float(((log_r - log_r.mean()) ** 2).sum())
        r2_head = 1.0 - float(((pred - log_r) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0

        y = log_r.cpu().unsqueeze(1)

        def _lstsq_r2(x: torch.Tensor) -> float:
            try:
                sol = torch.linalg.lstsq(x, y).solution
                resid = float(((x @ sol) - y).pow(2).sum())
                return 1.0 - resid / ss_tot if ss_tot > 0 else 0.0
            except RuntimeError:  # pragma: no cover - degenerate probe
                return 0.0

        # row-only baseline: predict log_r from view-row coordinates alone
        row_a = ((batch["boxes_a"][:, 1] + batch["boxes_a"][:, 3]) / 2.0).float()
        row_b = ((batch["boxes_b"][:, 1] + batch["boxes_b"][:, 3]) / 2.0).float()
        r2_row = _lstsq_r2(torch.stack([torch.ones(m), row_a, row_b], dim=1))

        # size-only baseline: predict log_r from the box sizes alone
        def _log_side(boxes: torch.Tensor) -> torch.Tensor:
            w = (boxes[:, 2] - boxes[:, 0]).abs().clamp_min(1e-6)
            h = (boxes[:, 3] - boxes[:, 1]).abs().clamp_min(1e-6)
            return 0.5 * torch.log(w * h)  # log geometric-mean side

        side_a = _log_side(batch["boxes_a"].cpu().float())
        side_b = _log_side(batch["boxes_b"].cpu().float())
        r2_size = _lstsq_r2(torch.stack([torch.ones(m), side_a, side_b], dim=1))

        delta = r2_head - max(r2_row, r2_size)
        flag_collapse = pred_std < self.cfg.pred_std_flag
        flag_row = self._epoch >= self.cfg.row_probe_deadline_epoch and delta <= 0.0
        if flag_collapse:
            warnings.warn(
                f"[scalereal sentinel] epoch {self._epoch}: probe prediction std "
                f"{pred_std:.4f} < {self.cfg.pred_std_flag} — scale head collapsing "
                f"to a constant.",
                RuntimeWarning, stacklevel=2,
            )
        if flag_row:
            warnings.warn(
                f"[scalereal sentinel] epoch {self._epoch}: head R^2 {r2_head:.3f} does "
                f"not beat the row-only regressor R^2 {r2_row:.3f} / size-only "
                f"regressor R^2 {r2_size:.3f} by the epoch-"
                f"{self.cfg.row_probe_deadline_epoch} deadline — row/size shortcut "
                f"suspected (the blur-shortcut analogue).",
                RuntimeWarning, stacklevel=2,
            )

        record = {
            "epoch": float(self._epoch),
            "n_probe_pairs": float(m),
            "probe_smooth_l1": probe_l1,
            "sign_acc": sign_acc,
            "spearman": rho,
            "pred_std": pred_std,
            "r2_head": r2_head,
            "r2_row": r2_row,
            "r2_size": r2_size,
            "row_probe_delta_r2": delta,
            "flag_pred_collapse": float(flag_collapse),
            "flag_row_shortcut": float(flag_row),
        }
        self.sentinel_records.append(record)
        return record

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScaleRealChannel(pairs={self.pairs_path!r}, "
            f"attached={self.descriptor is not None})"
        )
