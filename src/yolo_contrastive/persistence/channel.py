"""PersistenceChannel — REVISIT as an AuxChannel on the anchored joint trainer.

One channel, two label-like supervision signals computed from the SINGLE
student forward the trainer runs on the channel batch (A-views and B-views
stacked along the batch dim — the channel never forwards the model itself):

    Signal A (``corr``): positive-only SimSiam consistency between P3
        features sampled at homography-corresponding points across the two
        traversals. The "augmentation" between branches is REAL cross-session
        variation (months apart, weather, exposure) — per-image targets that
        change every batch (R2), no frozen teacher (R1), zero negatives (R4),
        zero teacher-side trainables (R6: projector/predictor are
        student-side, trained jointly).

    Signal B (``pers``): dense 3-class cross-entropy (background / persistent
        / transient, ignore=255) on P3 through a tiny conv head — persistence
        is exactly the property separating the downstream taxonomy
        (potholes/manholes/bumps are permanent) from distractors.

Both heads are discarded at export (R8). P3-only by design: alignment error
(~1.25 px @ 512) is far sub-cell at stride 8, while P4/P5 cells are too
coarse for decimeter-scale road furniture.

Batch contract (produced by ``pair_dataset.collate_pairs``):
    img    [2B, 3, S, S]  A-views then B-views
    pts    [B, 2, K, 2]   view-normalized correspondence coords (A, B)
    valid  [B, K]         bool point-validity mask
    labels [2B, g, g]     int64 P3-grid label maps (g = S / 8)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..anchored.channel import AuxChannel
from ..exceptions import FeatureTapError
from .heads import (
    CorrespondencePredictor,
    CorrespondenceProjector,
    PersistenceHead,
    simsiam_one_way,
)

__all__ = ["PersistenceChannel"]

#: P3 stride — the tap-sanity assertion (E5 wrong-layer guard) checks it.
P3_STRIDE = 8


class PersistenceChannel(AuxChannel):
    """REVISIT cross-traversal persistence channel (name: ``"persistence"``).

    Args:
        pairs_path / labels_path: parquet manifests consumed by the loader
            (see ``pair_dataset.PairDataset``). Required unless ``dataset``
            is injected.
        dataset: optional pre-built torch Dataset yielding PairDataset-style
            items (tests / custom pipelines).
        w_corr / w_pers: per-signal loss weights (the trainer applies the
            global ``lambda_aux`` on top).
        k_min: minimum valid correspondence points per pair — below this,
            Signal A skips the pair that step (skip-rate sentinel).
        proj_dim / proj_hidden: projector/predictor widths.
        batch_pairs: pool batch size in PAIRS (default: trainer cfg batch).
            Each pair contributes TWO images to the forward.
        label_cfg / aug / seed: forwarded to PairDataset.
        passthrough_heads: test hook — projector/predictor become identity
            so analytic loss-minimum tests can reason about raw features.

    Sentinels: :meth:`sentinel_metrics` exposes the SimSiam-collapse guard
    (projector per-dim std, abort < 0.044), persistence-head argmax class
    frequencies, and the Signal-A skip rate, accumulated since the last
    :meth:`reset_epoch_stats`. The trainer invokes them automatically each
    epoch through :meth:`on_epoch_end` (R9).
    """

    name = "persistence"
    required_levels = ("P3",)

    def __init__(
        self,
        pairs_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        dataset: Optional[Any] = None,
        w_corr: float = 1.0,
        w_pers: float = 1.0,
        k_min: int = 32,
        proj_dim: int = 128,
        proj_hidden: int = 256,
        batch_pairs: Optional[int] = None,
        label_cfg: Optional[Any] = None,
        aug: Optional[Any] = None,
        seed: Optional[int] = None,
        ignore_index: int = 255,
        passthrough_heads: bool = False,
    ) -> None:
        self.pairs_path = pairs_path
        self.labels_path = labels_path
        self._dataset = dataset
        self.w_corr = float(w_corr)
        self.w_pers = float(w_pers)
        self.k_min = int(k_min)
        self.proj_dim = int(proj_dim)
        self.proj_hidden = int(proj_hidden)
        self.batch_pairs = batch_pairs
        self.label_cfg = label_cfg
        self.aug = aug
        self.seed = seed
        self.ignore_index = int(ignore_index)
        self.passthrough_heads = bool(passthrough_heads)

        self.projector: Optional[CorrespondenceProjector] = None
        self.predictor: Optional[CorrespondencePredictor] = None
        self.pers_head: Optional[PersistenceHead] = None
        self.p3_channels: Optional[int] = None
        self.reset_epoch_stats()

    # ── AuxChannel interface ────────────────────────────────────────────────

    def attach(self, model: nn.Module, taps: Any) -> nn.ModuleList:
        """Probe the P3 tap (stride + channel width), build the three heads.

        E5 wrong-layer guard: a probe forward at 64 px must yield a P3 map of
        spatial size 64/8 = 8 — a tap hooked on the wrong layer (wrong
        stride) raises :class:`FeatureTapError` here, at construction, not as
        silently misaligned supervision later.
        """
        feats = self._probe(model, taps, imgsz=64)
        if "P3" not in feats:
            raise FeatureTapError(
                f"persistence channel requires a 'P3' tap; got {sorted(feats)}"
            )
        f3 = feats["P3"]
        expected = 64 // P3_STRIDE
        if f3.dim() != 4 or f3.shape[-2] != expected or f3.shape[-1] != expected:
            raise FeatureTapError(
                f"P3 tap stride check failed: expected [B, C, {expected}, {expected}] "
                f"for a 64 px probe (stride {P3_STRIDE}), got {tuple(f3.shape)} — "
                f"the tap is hooked on the wrong layer."
            )
        self.p3_channels = int(f3.shape[1])

        self.projector = CorrespondenceProjector(
            self.p3_channels, self.proj_hidden, self.proj_dim,
            passthrough=self.passthrough_heads,
        )
        self.predictor = CorrespondencePredictor(
            self.proj_dim, self.proj_hidden, passthrough=self.passthrough_heads,
        )
        self.pers_head = PersistenceHead(self.p3_channels, n_classes=3)
        return nn.ModuleList([self.projector, self.predictor, self.pers_head])

    def loss(self, batch: Dict[str, Any], taps: Any) -> Dict[str, torch.Tensor]:
        """Compute ``{"corr": ..., "pers": ...}`` from this batch's P3 tap.

        Either term may be absent (no valid points / no labeled cells); an
        empty dict skips the channel this step.
        """
        if self.projector is None:
            raise RuntimeError("PersistenceChannel.loss called before attach()")
        feats = taps.get_features()["P3"]
        n2 = feats.shape[0]
        if n2 % 2 != 0:
            raise ValueError(
                f"persistence batch must stack A and B views ([2B, ...]); "
                f"got {n2} images"
            )
        b = n2 // 2
        terms: Dict[str, torch.Tensor] = {}

        corr = self._corr_loss(feats[:b], feats[b:], batch)
        if corr is not None:
            terms["corr"] = self.w_corr * corr
        pers = self._pers_loss(feats, batch)
        if pers is not None:
            terms["pers"] = self.w_pers * pers
        return terms

    def build_loader(self, cfg: Dict[str, Any]) -> Iterable:
        """DataLoader over PairDataset (manifests + local JPEGs, offline)."""
        from torch.utils.data import DataLoader

        from .pair_dataset import PairDataset, collate_pairs

        ds = self._dataset
        if ds is None:
            if self.pairs_path is None:
                raise ValueError(
                    "PersistenceChannel needs pairs_path (or an injected dataset)"
                )
            ds = PairDataset(
                self.pairs_path, labels=self.labels_path, imgsz=cfg["imgsz"],
                label_cfg=self.label_cfg, aug=self.aug, seed=self.seed,
            )
        batch = int(self.batch_pairs or cfg["batch"])
        return DataLoader(
            ds, batch_size=batch, shuffle=True, num_workers=int(cfg["workers"]),
            collate_fn=collate_pairs, drop_last=len(ds) > batch,
        )

    # ── Signal A: positive-only correspondence consistency ─────────────────

    def _corr_loss(
        self, f_a: torch.Tensor, f_b: torch.Tensor, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        pts = batch.get("pts")
        valid = batch.get("valid")
        if pts is None or valid is None:
            return None
        b = f_a.shape[0]
        valid = valid.bool()
        pair_ok = valid.sum(dim=1) >= self.k_min
        self._stat_pairs += b
        self._stat_pairs_skipped += int((~pair_ok).sum())
        mask = valid & pair_ok[:, None]                       # [B, K]
        n = int(mask.sum())
        self._stat_points += n
        if n < 2:  # BN1d needs >= 2 samples; also nothing to learn from 0/1
            return None

        v_a = self._sample_points(f_a, pts[:, 0])             # [B, K, C]
        v_b = self._sample_points(f_b, pts[:, 1])
        v_a, v_b = v_a[mask], v_b[mask]                       # [N, C]

        z_a = self.projector(v_a)
        z_b = self.projector(v_b)
        p_a = self.predictor(z_a)
        p_b = self.predictor(z_b)
        loss = 0.5 * simsiam_one_way(p_a, z_b) + 0.5 * simsiam_one_way(p_b, z_a)

        with torch.no_grad():  # SimSiam-collapse sentinel input
            z = F.normalize(torch.cat([z_a, z_b]).float(), dim=-1)
            self._stat_proj_std = float(z.std(dim=0).mean())
        return loss

    @staticmethod
    def _sample_points(feats: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Bilinear point sampling: feats [B, C, h, w] + view-normalized
        coords [B, K, 2] (x, y in [0, 1]) -> [B, K, C]."""
        grid = (pts.to(feats.dtype) * 2.0 - 1.0).unsqueeze(1)  # [B, 1, K, 2]
        out = F.grid_sample(feats, grid, mode="bilinear", align_corners=False)
        return out.squeeze(2).permute(0, 2, 1)                 # [B, K, C]

    # ── Signal B: dense persistence classification ──────────────────────────

    def _pers_loss(
        self, feats: torch.Tensor, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        labels = batch.get("labels")
        if labels is None:
            return None
        labels = labels.long()
        if labels.shape[0] != feats.shape[0] or labels.shape[-2:] != feats.shape[-2:]:
            raise ValueError(
                f"label map shape {tuple(labels.shape)} does not match P3 features "
                f"{tuple(feats.shape)} — the dataset's imgsz must equal the trainer's."
            )
        if int((labels != self.ignore_index).sum()) == 0:
            return None
        logits = self.pers_head(feats)
        with torch.no_grad():  # class-frequency sentinel input
            pred = logits.argmax(dim=1)
            total = pred.numel()
            for c in range(3):
                self._stat_class_counts[c] += int((pred == c).sum())
            self._stat_class_total += total
        # equal class weights; the 3:1 background cap does the balancing
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)

    # ── sentinels (R9 channel extras) ───────────────────────────────────────

    def on_epoch_end(self, epoch: int) -> Dict[str, float]:
        """R9 trainer hook: report this epoch's sentinel readouts, then reset
        the accumulators for the next epoch. Called automatically by
        ``AnchoredJointTrainer.train`` (logged as ``sentinel/persistence/*``);
        manual loops may call :meth:`sentinel_metrics` /
        :meth:`reset_epoch_stats` directly instead."""
        metrics = self.sentinel_metrics()
        self.reset_epoch_stats()
        return metrics

    def reset_epoch_stats(self) -> None:
        """Reset the per-epoch sentinel accumulators."""
        self._stat_pairs = 0
        self._stat_pairs_skipped = 0
        self._stat_points = 0
        self._stat_proj_std = float("nan")
        self._stat_class_counts = [0, 0, 0]
        self._stat_class_total = 0

    def sentinel_metrics(self) -> Dict[str, float]:
        """Channel sentinel readouts since the last reset.

        ``proj_std`` < 0.044 (= 0.5 / sqrt(128)) signals SimSiam collapse;
        ``skip_rate`` > 0.30 signals Signal-A starvation; any
        ``class_freq_*`` < 0.01 after epoch 2 signals a collapsed
        persistence head.
        """
        out = {
            "proj_std": self._stat_proj_std,
            "skip_rate": (self._stat_pairs_skipped / self._stat_pairs
                          if self._stat_pairs else float("nan")),
            "valid_points": float(self._stat_points),
        }
        for c, name in enumerate(("background", "persistent", "transient")):
            out[f"class_freq_{name}"] = (
                self._stat_class_counts[c] / self._stat_class_total
                if self._stat_class_total else float("nan")
            )
        return out

    # ── resume (Risk-16-safe: value copies, never assign=True) ─────────────

    def state_dict(self) -> Dict[str, Any]:
        if self.projector is None:
            return {}
        return {
            "projector": self.projector.state_dict(),
            "predictor": self.predictor.state_dict(),
            "pers_head": self.pers_head.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Clone-based value copy into the existing head tensors (E5/Risk-16:
        never ``load_state_dict(assign=True)`` — no storage aliasing)."""
        if self.projector is None:
            raise RuntimeError("call attach() before load_state_dict()")
        for key, module in (("projector", self.projector),
                            ("predictor", self.predictor),
                            ("pers_head", self.pers_head)):
            sub = state.get(key)
            if sub is None:
                raise KeyError(f"channel state_dict missing {key!r}")
            own = module.state_dict()
            for k, v in sub.items():
                if k not in own:
                    raise KeyError(f"unexpected key {key}.{k}")
                own[k].copy_(torch.as_tensor(v).clone())

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _probe(model: nn.Module, taps: Any, imgsz: int = 64) -> Dict[str, torch.Tensor]:
        """One dummy forward through ``model`` capturing tap features; taps
        are cleared afterwards so no stale features leak into training."""
        try:
            p = next(model.parameters())
            device, dtype = p.device, p.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32
        dummy = torch.zeros(2, 3, imgsz, imgsz, device=device, dtype=dtype)
        was_training = model.training
        model.eval()
        taps.clear()
        try:
            with torch.no_grad():
                _ = model(dummy)
            feats = {k: v.detach() for k, v in taps.get_features().items()}
        finally:
            taps.clear()
            if was_training:
                model.train()
        return feats
