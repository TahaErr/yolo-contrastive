"""Sentinels — per-epoch training-health monitors for anchored joint training (R9).

The repo's documented silent-bug history (rank collapse by epoch 2, EMA
aliasing zeroing weights inside one epoch, frozen-teacher targets exhausted at
75-92%, wrong-layer hooks) demands cheap, always-on diagnostics:

    * effective rank of P5 features on a FIXED probe batch
      (rank collapse detector — failed SSL arms sat at eff_rank 3-10 vs the
      COCO init's 60.4),
    * linear CKA of those probe features vs the previous epoch
      (representation-churn detector — a sudden drop means the init is being
      destroyed despite the replay anchor),
    * replay cls-loss EMA drift vs its first-epoch baseline
      (catastrophic-forgetting detector for the COCO anchor, R3),
    * per-module head weight norms
      (fresh-head gradient-blast / dead-head detector).

Each sentinel has a warn level (``warnings.warn``) and an abort level
(:class:`SentinelAbort` raised with a self-explanatory message). The metrics
row is recorded (and flushed to CSV if configured) BEFORE any abort is raised,
so the forensic trail survives the crash.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class SentinelAbort(RuntimeError):
    """A sentinel crossed its abort threshold — training must stop.

    The message names the sentinel, its value and its threshold; the full
    per-epoch record was appended to ``SentinelLog.records`` (and the CSV,
    if enabled) before raising.
    """


# ── feature-statistics primitives ─────────────────────────────────────────────


def _to_matrix(features: torch.Tensor) -> torch.Tensor:
    """Flatten features to a [N, D] sample matrix.

    [B, C, H, W] -> [B*H*W, C] (each spatial position is a sample),
    [B, N, D]    -> [B*N, D],
    [N, D]       -> unchanged.
    """
    if features.dim() == 4:
        b, c, h, w = features.shape
        return features.permute(0, 2, 3, 1).reshape(b * h * w, c)
    if features.dim() == 3:
        return features.reshape(-1, features.shape[-1])
    if features.dim() == 2:
        return features
    raise ValueError(f"features must be 2D/3D/4D, got shape {tuple(features.shape)}")


def effective_rank(features: torch.Tensor, center: bool = True, eps: float = 1e-12) -> float:
    """Effective rank (Roy & Vetterli 2007): exp of the entropy of the
    normalized singular-value distribution of the (optionally centered)
    [N, D] sample matrix.

    Analytic anchors (used by the tests): a rank-1 matrix has effective rank
    1.0; a matrix with k equal nonzero singular values has effective rank k.
    """
    x = _to_matrix(features).float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


def linear_cka(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Linear CKA between two feature matrices with the same sample count.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F) on column-centered
    matrices. 1.0 = identical up to orthogonal transform + isotropic scaling;
    near 0 = unrelated representations.
    """
    x = _to_matrix(a).float()
    y = _to_matrix(b).float()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample counts differ: {x.shape[0]} vs {y.shape[0]}")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (y.t() @ x).norm().pow(2)
    denom = (x.t() @ x).norm() * (y.t() @ y).norm()
    return float(hsic / denom.clamp_min(eps))


# ── thresholds ────────────────────────────────────────────────────────────────


@dataclass
class SentinelThresholds:
    """Warn / abort levels. Defaults follow wf2_ac.md Stage-1 sentinels
    (abort: eff_rank < 20, replay cls drift > 30%); tune per run via the
    trainer's ``sentinel_thresholds`` argument (tests relax them — a 4-image
    64px probe cannot reach eff_rank 20).
    """

    eff_rank_warn: float = 30.0     # warn if probe eff_rank falls below
    eff_rank_abort: float = 20.0    # abort if probe eff_rank falls below
    cls_drift_warn: float = 0.15    # warn if replay cls EMA drifts up > 15%
    cls_drift_abort: float = 0.30   # abort if replay cls EMA drifts up > 30%
    cka_warn: float = 0.30          # warn if CKA vs previous epoch falls below
    head_norm_growth_warn: float = 3.0  # warn if a head's weight norm grows > 3x initial


# ── per-epoch log ─────────────────────────────────────────────────────────────


class SentinelLog:
    """Per-epoch sentinel computation + history + CSV.

    Args:
        model: the detector under training (forwarded on the probe batch in
            eval mode under ``no_grad``; train mode is restored afterwards).
        taps: shared ``MultiScaleFeatureTap`` (already set up) — or any object
            with ``clear()`` / ``get_features() -> {level: tensor}``.
        probe_batch: FIXED probe images [B, 3, H, W]; cloned at construction
            so later in-place edits by the caller cannot change the probe.
        thresholds: warn/abort levels (default :class:`SentinelThresholds`).
        level: which tap level to monitor (default "P5", the documented
            collapse site).
        cls_ema_momentum: EMA momentum for the replay cls-loss tracker.
        csv_path: optional CSV file; one row appended per epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        taps: Any,
        probe_batch: torch.Tensor,
        thresholds: Optional[SentinelThresholds] = None,
        level: str = "P5",
        cls_ema_momentum: float = 0.98,
        csv_path: Optional[str] = None,
    ) -> None:
        if probe_batch.dim() != 4:
            raise ValueError(f"probe_batch must be [B, 3, H, W], got {tuple(probe_batch.shape)}")
        if not 0.0 <= cls_ema_momentum < 1.0:
            raise ValueError(f"cls_ema_momentum must be in [0, 1), got {cls_ema_momentum}")
        self.model = model
        self.taps = taps
        self.probe = probe_batch.detach().clone()
        self.thresholds = thresholds if thresholds is not None else SentinelThresholds()
        self.level = level
        self._mom = float(cls_ema_momentum)
        self.csv_path = str(csv_path) if csv_path else None

        self._cls_ema: Optional[float] = None
        self._cls_baseline: Optional[float] = None
        self._prev_feats: Optional[torch.Tensor] = None
        self._init_head_norms: Dict[str, float] = {}
        self.records: List[Dict[str, float]] = []

    # ── per-step input ────────────────────────────────────────────────────

    def update_replay_cls(self, value: float) -> None:
        """Feed one replay-batch cls-loss value (call once per optimizer step)."""
        v = float(value)
        if not math.isfinite(v):
            return
        if self._cls_ema is None:
            self._cls_ema = v
        else:
            self._cls_ema = self._mom * self._cls_ema + (1.0 - self._mom) * v

    # ── probe forward ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _probe_features(self) -> torch.Tensor:
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        was_training = self.model.training
        self.model.eval()
        self.taps.clear()
        try:
            _ = self.model(self.probe.to(device))
            feats = self.taps.get_features()[self.level]
            return _to_matrix(feats).detach().float().cpu()
        finally:
            self.taps.clear()
            if was_training:
                self.model.train()

    # ── epoch boundary ────────────────────────────────────────────────────

    def epoch_end(
        self,
        epoch: int,
        head_modules: Optional[Dict[str, nn.Module]] = None,
    ) -> Dict[str, float]:
        """Compute all sentinels; record; warn or raise :class:`SentinelAbort`.

        Args:
            epoch: 1-based epoch index (used in the record + messages).
            head_modules: optional ``{name: module}`` of trainable heads whose
                total weight norm is tracked (channel heads + COCO Detect head).

        Returns:
            The metrics record (also appended to ``self.records``).
        """
        th = self.thresholds

        feats = self._probe_features()
        er = effective_rank(feats)
        cka = linear_cka(feats, self._prev_feats) if self._prev_feats is not None else float("nan")
        self._prev_feats = feats

        # Replay cls drift vs first-epoch baseline (signed: only an upward
        # drift — COCO forgetting — counts toward warn/abort).
        drift = float("nan")
        if self._cls_ema is not None:
            if self._cls_baseline is None:
                self._cls_baseline = self._cls_ema
                drift = 0.0
            elif self._cls_baseline > 0:
                drift = (self._cls_ema - self._cls_baseline) / self._cls_baseline

        record: Dict[str, float] = {
            "epoch": float(epoch),
            "eff_rank": er,
            "cka_prev_epoch": cka,
            "replay_cls_ema": float("nan") if self._cls_ema is None else self._cls_ema,
            "replay_cls_drift": drift,
        }

        norm_warnings: List[str] = []
        if head_modules:
            for name, mod in head_modules.items():
                norm = float(
                    sum(p.detach().float().norm().item() ** 2 for p in mod.parameters()) ** 0.5
                )
                record[f"head_norm/{name}"] = norm
                init = self._init_head_norms.setdefault(name, norm)
                if init > 0 and norm > th.head_norm_growth_warn * init:
                    norm_warnings.append(f"{name}: {norm:.3f} > {th.head_norm_growth_warn}x "
                                         f"initial {init:.3f}")

        # Record BEFORE any abort so the forensic trail survives the crash.
        self.records.append(record)
        self._append_csv(record)

        # ── warn / abort ladder ──────────────────────────────────────────
        if er < th.eff_rank_abort:
            raise SentinelAbort(
                f"[sentinel] epoch {epoch}: effective rank of {self.level} probe features "
                f"collapsed to {er:.2f} < abort threshold {th.eff_rank_abort} — this is the "
                f"documented rank-collapse failure signature; stopping."
            )
        if er < th.eff_rank_warn:
            warnings.warn(
                f"[sentinel] epoch {epoch}: eff_rank({self.level})={er:.2f} below warn "
                f"threshold {th.eff_rank_warn}.",
                RuntimeWarning, stacklevel=2,
            )
        if math.isfinite(drift):
            if drift > th.cls_drift_abort:
                raise SentinelAbort(
                    f"[sentinel] epoch {epoch}: replay cls-loss EMA drifted "
                    f"{drift:+.1%} above its baseline (> {th.cls_drift_abort:.0%}) — the COCO "
                    f"anchor is being forgotten (R3 violated); stopping."
                )
            if drift > th.cls_drift_warn:
                warnings.warn(
                    f"[sentinel] epoch {epoch}: replay cls-loss EMA drift {drift:+.1%} above "
                    f"warn threshold {th.cls_drift_warn:.0%}.",
                    RuntimeWarning, stacklevel=2,
                )
        if math.isfinite(cka) and cka < th.cka_warn:
            warnings.warn(
                f"[sentinel] epoch {epoch}: CKA vs previous epoch {cka:.3f} below warn "
                f"threshold {th.cka_warn} — heavy representation churn.",
                RuntimeWarning, stacklevel=2,
            )
        for msg in norm_warnings:
            warnings.warn(
                f"[sentinel] epoch {epoch}: head weight norm growth — {msg}.",
                RuntimeWarning, stacklevel=2,
            )

        return record

    # ── CSV ───────────────────────────────────────────────────────────────

    def _append_csv(self, record: Dict[str, float]) -> None:
        if self.csv_path is None:
            return
        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SentinelLog(level={self.level!r}, probe={tuple(self.probe.shape)}, "
            f"epochs_recorded={len(self.records)})"
        )
