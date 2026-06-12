"""GASP-Real loss terms — scale equivariance + positive-only content invariance.

LOSS 1 — SCALE EQUIVARIANCE (primary), :func:`scale_pair_loss`::

    L_scale = (1/M) Σ_pairs SmoothL1_β( [s(φ_B) − s(φ_A)] − log_r(A→B) ),  β=0.25

Smooth-L1 regression on a real-valued label is exactly the "label-like
target" R4 prescribes. The per-patch-scalar-DIFFERENCE form is antisymmetric
by construction (ŝ_AB = −ŝ_BA exactly, pair order irrelevant) and forces
per-patch scale-aware features. InfoNCE-over-log_r-bins is deliberately
absent — the GASP history shows that family is a calibration minefield
(oracle-below-chance logits §10.39, duplicate-candidate bug, silent 2-scale
MSE fallback, temperature fragility).

LOSS 2 — CONTENT INVARIANCE (positive-only, R4), :func:`content_consistency_loss`::

    L_inv = (1/2M) Σ_pairs [(1 − cos(q(z_A), sg(z_B))) + (1 − cos(q(z_B), sg(z_A)))]

SimSiam-style: predictor + stop-grad, NO negatives, no EMA second forward
(SimSiam shows EMA unnecessary; the per-step COCO replay loss is the real
anti-collapse anchor). Matched pairs contain similar content at genuinely
different scales, so this term teaches scale-INVARIANCE of content features
while the scalar head isolates scale-EQUIVARIANCE — the explicit resolution
of the SAPS-within vs GASP objective contradiction. No cross-scale negatives
anywhere (R4).

Zero-pair batches return a graph-connected zero (the gasp/losses.py
natural_loss zero-pattern, upgraded to keep a grad_fn so the trainer's
``backward()`` never special-cases an empty step).
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def connected_zero(refs: Iterable[torch.Tensor]) -> torch.Tensor:
    """A scalar 0.0 connected to the autograd graph of ``refs``.

    Built as ``sum(r.sum() * 0)`` over refs that participate in autograd, so
    ``backward()`` runs cleanly (contributing exactly zero gradient) even on
    steps with no valid pairs. Falls back to a plain zero if nothing in
    ``refs`` carries grad.
    """
    total = None
    for r in refs:
        if torch.is_tensor(r) and (r.requires_grad or r.grad_fn is not None):
            term = r.sum() * 0.0
            total = term if total is None else total + term
    if total is None:
        return torch.zeros(())
    return total


def pair_scale_pred(s_a: torch.Tensor, s_b: torch.Tensor) -> torch.Tensor:
    """ŝ(A→B) = s(φ_B) − s(φ_A) — exactly antisymmetric under A<->B swap."""
    return s_b - s_a


def scale_pair_loss(
    s_a: torch.Tensor,
    s_b: torch.Tensor,
    log_r: torch.Tensor,
    beta: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Smooth-L1 on the scalar-difference prediction vs the real log ratio.

    Args:
        s_a: [M] scale potentials of patches A.
        s_b: [M] scale potentials of patches B.
        log_r: [M] labels log(Z_A / Z_B) (= log apparent scale of B rel. A).
        beta: Smooth-L1 transition point (0.25 ~ 28% scale error).

    Returns:
        ``{"loss": scalar tensor (graph-connected even at M=0),
           "sign_acc": float, "pred_std": float, "n_pairs": int}``.
        ``sign_acc`` is the fraction of pairs whose predicted ratio sign
        matches the label sign; ``pred_std`` is the std of predictions
        (collapse-to-constant diagnostic).
    """
    if s_a.shape != s_b.shape or s_a.shape != log_r.shape:
        raise ValueError(
            f"shape mismatch: s_a {tuple(s_a.shape)}, s_b {tuple(s_b.shape)}, "
            f"log_r {tuple(log_r.shape)}"
        )
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    m = int(s_a.numel())
    if m == 0:
        return {
            "loss": connected_zero([s_a, s_b]),
            "sign_acc": 0.0,
            "pred_std": 0.0,
            "n_pairs": 0,
        }
    pred = pair_scale_pred(s_a, s_b)
    loss = F.smooth_l1_loss(pred, log_r.to(pred.dtype), beta=float(beta))
    with torch.no_grad():
        sign_acc = float((torch.sign(pred) == torch.sign(log_r)).float().mean())
        pred_std = float(pred.float().std(unbiased=False)) if m > 1 else 0.0
    return {"loss": loss, "sign_acc": sign_acc, "pred_std": pred_std, "n_pairs": m}


def content_consistency_loss(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Symmetric SimSiam cosine term — predictor side q, stop-grad side z.

    ``L = (1/2M) Σ [(1 − cos(q_a, sg(z_b))) + (1 − cos(q_b, sg(z_a)))]``.

    The stop-grad is applied HERE (``z.detach()``) so no caller can forget
    it; the q inputs must come from the predictor head. Positive-only (R4):
    no negatives, no queue, no EMA forward.

    Returns:
        ``{"loss": scalar tensor (graph-connected at M=0), "n_pairs": int}``.
    """
    shapes = {tuple(t.shape) for t in (q_a, q_b, z_a, z_b)}
    if len(shapes) != 1:
        raise ValueError(f"q/z shape mismatch: {sorted(shapes)}")
    m = int(q_a.shape[0]) if q_a.dim() >= 1 else 0
    if m == 0:
        return {"loss": connected_zero([q_a, q_b]), "n_pairs": 0}
    loss_ab = 1.0 - F.cosine_similarity(q_a, z_b.detach(), dim=-1)
    loss_ba = 1.0 - F.cosine_similarity(q_b, z_a.detach(), dim=-1)
    return {"loss": 0.5 * (loss_ab + loss_ba).mean(), "n_pairs": m}


def spearman_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman rank correlation (diagnostic; average-free tie handling).

    Implemented as Pearson correlation of double-argsort ranks — exact for
    distinct values, adequate for the sentinel diagnostics it feeds (no scipy
    dependency).
    """
    p = pred.detach().float().flatten()
    t = target.detach().float().flatten()
    if p.numel() != t.numel():
        raise ValueError(f"length mismatch: {p.numel()} vs {t.numel()}")
    n = p.numel()
    if n < 2:
        return 0.0
    rp = torch.argsort(torch.argsort(p)).float()
    rt = torch.argsort(torch.argsort(t)).float()
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = rp.norm() * rt.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((rp * rt).sum() / denom)
