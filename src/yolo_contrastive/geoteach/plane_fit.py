"""Road-plane fitting in uncalibrated inverse-depth space (TERRA Stage 0, step 2).

The load-bearing geometric fact: for a planar road, ideal inverse depth is an
AFFINE function of pixel coordinates, ``d_plane(u, v) = a*u + b*v + c``, and
any affine ambiguity ``(s, t)`` of a relative monocular depth model preserves
affinity (``s*d + t`` is still affine in ``(u, v)``). So the plane is fitted
directly in the depth model's native inverse-depth output space — no metric
calibration, no camera intrinsics, no pose. The basis is extended to
``[u, v, 1, v^2, u*v]`` (quadratic-in-v; Fan et al.'s road surface model) to
absorb road crown and grade.

Pipeline per image (all pure numpy, ~20 ms at half resolution, CPU):
    1. Seed points from a bottom trapezoid (rows 0.55H–0.97H) — the prior road
       region in forward-facing road imagery.
    2. RANSAC: 6-point minimal solves, 500 iterations, inlier threshold
       ``tau = 1.5 * sigma_hat`` (sigma_hat = robust MAD scale of an initial
       least-squares fit on the seed points).
    3. 3 refit rounds: least squares on inliers, re-estimate sigma, re-select.
    4. Final Huber IRLS refinement on the inlier set.
    5. ``sigma_MAD = 1.4826 * median|res|`` over inliers; the full-image
       inlier mask IS the road mask (no segmentation model in this pipeline).

Per-image trust gates (night / rain / fit failure) are flagged here and acted
on in residual_labels.py: inlier ratio < 40% of the seed trapezoid, or
``sigma_MAD`` above a caller-supplied cap (the pool's 95th percentile).

No torch in this module — it runs inside multiprocessing pools over the
manifest (design doc: wf2_designs.md, TERRA core_mechanism).
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

#: Number of plane-basis terms: [u, v, 1, v^2, u*v].
N_BASIS = 5

#: 1 / Phi^-1(3/4) — converts the median absolute deviation to a Gaussian
#: standard deviation estimate.
MAD_TO_SIGMA = 1.4826


@dataclasses.dataclass
class PlaneFitConfig:
    """All plane-fit knobs in one place (ablation grid: quadratic vs planar,
    trapezoid geometry, RANSAC budget, trust gates)."""

    # RANSAC
    ransac_iters: int = 500
    min_samples: int = 6                 # 6-pt minimal solve (5 unknowns)
    tau_scale: float = 1.5               # tau = tau_scale * sigma_hat
    refit_rounds: int = 3
    # Final robust refinement
    huber_delta: float = 1.345           # in units of sigma
    huber_iters: int = 5
    # Bottom-trapezoid seeding (fractions of image height/width)
    trapezoid_top: float = 0.55
    trapezoid_bottom: float = 0.97
    trapezoid_top_halfwidth: float = 0.20
    trapezoid_bottom_halfwidth: float = 0.45
    # Absolute tau cap, as a fraction of the seed region's robust disparity
    # span (5th-95th pct). A pure MAD-derived tau is scale-invariant — on
    # structureless input it inflates with the garbage and the inlier-ratio
    # trust gate could never fire. A clean road trapezoid spans a LARGE
    # disparity range top-to-bottom while its plane residuals are tiny, so
    # the cap is inactive on good scenes and decisive on bad ones.
    max_tau_range_frac: float = 0.05
    # Performance / determinism
    max_seed_points: int = 4000          # subsample the trapezoid for speed
    seed: Optional[int] = None           # rng seed (tests)
    # Surface model: True = [u, v, 1, v^2, u*v]; False = pure planar [u, v, 1]
    quadratic: bool = True
    # Trust gate (acted on downstream)
    min_inlier_ratio: float = 0.40       # of the seed trapezoid


@dataclasses.dataclass
class PlaneFitResult:
    """Fit output + trust diagnostics for one image.

    Attributes:
        params: basis weights, shape (5,) — ``[u, v, 1, v^2, u*v]`` order.
            For ``quadratic=False`` the last two entries are exactly 0.
        sigma_mad: robust residual scale over the final inliers
            (``1.4826 * median|res|``); the z-map denominator.
        tau: final inlier threshold used for the road mask.
        inlier_ratio: inlier fraction over the seed trapezoid (trust gate).
        inlier_mask: full-image boolean road mask (``|res| <= tau``).
        n_seed: number of seed points used.
        trusted: per-image trust gate (inlier ratio); sigma_MAD percentile
            gating needs pool statistics and is applied by the caller.
        reason: human-readable trust/failure reason.
    """

    params: np.ndarray
    sigma_mad: float
    tau: float
    inlier_ratio: float
    inlier_mask: np.ndarray
    n_seed: int
    trusted: bool
    reason: str = "ok"


# ── basis / surface evaluation ────────────────────────────────────────────────


def normalized_coords(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-center normalized coordinates ``u, v`` in [0, 1], each [H, W]."""
    h, w = int(shape[0]), int(shape[1])
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    return np.broadcast_to(u[None, :], (h, w)), np.broadcast_to(v[:, None], (h, w))


def design_basis(u: np.ndarray, v: np.ndarray, quadratic: bool = True) -> np.ndarray:
    """Stack the ``[u, v, 1, v^2, u*v]`` basis. Shape: u.shape + (5,).

    With ``quadratic=False`` the last two columns are zeroed so params keep a
    fixed (5,) layout across both surface models.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ones = np.ones_like(u)
    if quadratic:
        cols = [u, v, ones, v * v, u * v]
    else:
        cols = [u, v, ones, np.zeros_like(u), np.zeros_like(u)]
    return np.stack(cols, axis=-1)


def evaluate_surface(params: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Evaluate the fitted surface ``d_surf(u, v)`` over a full [H, W] grid."""
    params = np.asarray(params, dtype=np.float64).reshape(N_BASIS)
    u, v = normalized_coords(shape)
    # quadratic=True is safe even for planar fits (their tail params are 0)
    basis = design_basis(u, v, quadratic=True)
    return basis @ params


def standardized_residual(inv_depth: np.ndarray, fit: PlaneFitResult) -> np.ndarray:
    """Standardized plane residual ``z = (d - d_surf) / sigma_MAD`` [H, W].

    Sign convention (the physics): a pothole point is FARTHER than the road
    plane, so its inverse depth is SMALLER → residual < 0 → **depression has
    z < 0**; elevation (speed bump) has z > 0.
    """
    d = np.asarray(inv_depth, dtype=np.float64)
    d_surf = evaluate_surface(fit.params, d.shape)
    sigma = max(float(fit.sigma_mad), 1e-12)
    return ((d - d_surf) / sigma).astype(np.float32)


# ── trapezoid seeding ─────────────────────────────────────────────────────────


def trapezoid_mask(shape: Tuple[int, int], cfg: Optional[PlaneFitConfig] = None) -> np.ndarray:
    """Boolean [H, W] mask of the bottom road-prior trapezoid.

    Rows span ``[trapezoid_top, trapezoid_bottom] * H``; the horizontal
    half-width grows linearly from ``trapezoid_top_halfwidth`` to
    ``trapezoid_bottom_halfwidth`` (fractions of W) around the image center —
    the typical projection of a road ahead of a forward-facing camera.
    """
    cfg = cfg or PlaneFitConfig()
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    r0 = int(round(cfg.trapezoid_top * h))
    r1 = int(round(cfg.trapezoid_bottom * h))
    r0, r1 = max(0, min(r0, h - 1)), max(0, min(r1, h))
    if r1 <= r0:
        return mask
    cx = w / 2.0
    for r in range(r0, r1):
        t = (r - r0) / max(r1 - 1 - r0, 1)
        half = (cfg.trapezoid_top_halfwidth
                + t * (cfg.trapezoid_bottom_halfwidth - cfg.trapezoid_top_halfwidth)) * w
        c0 = int(round(cx - half))
        c1 = int(round(cx + half))
        mask[r, max(0, c0):min(w, c1)] = True
    return mask


# ── solvers ───────────────────────────────────────────────────────────────────


def _lstsq(basis: np.ndarray, d: np.ndarray,
           weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """(Weighted) least squares for the plane params; None if degenerate."""
    a = basis
    b = d
    if weights is not None:
        sw = np.sqrt(weights)[:, None]
        a = a * sw
        b = b * sw[:, 0]
    # Drop all-zero columns (planar model) to keep lstsq well-posed.
    active = np.any(a != 0.0, axis=0)
    if not np.any(active):
        return None
    sol, _, rank, _ = np.linalg.lstsq(a[:, active], b, rcond=None)
    if rank < int(active.sum()) or not np.all(np.isfinite(sol)):
        return None
    params = np.zeros(N_BASIS, dtype=np.float64)
    params[active] = sol
    return params


def _mad_sigma(res: np.ndarray) -> float:
    """Robust sigma from residuals: ``1.4826 * median|res - median(res)|``."""
    med = np.median(res)
    return MAD_TO_SIGMA * float(np.median(np.abs(res - med)))


def _phi(x: float) -> float:
    """Standard normal CDF."""
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Standard normal quantile via bisection (avoids a scipy dependency)."""
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _truncated_mad_correction(c: float) -> float:
    """Consistency factor for MAD-sigma computed over residuals truncated at
    ``±c*sigma`` (the inlier set). Without it, re-estimating sigma from
    inliers each refit round compounds truncation shrinkage (~0.85x per round
    at c=1.5) and silently inflates downstream |z| magnitudes.

    For a Gaussian truncated at c, ``median|res| = sigma * Phi^-1((2*Phi(c)+1)/4)``;
    the untruncated value is ``sigma * Phi^-1(0.75)``. The ratio corrects the
    estimate (c=1.5 → ~1.178).
    """
    denom = _phi_inv((2.0 * _phi(c) + 1.0) / 4.0)
    if denom <= 0:
        return 1.0
    return _phi_inv(0.75) / denom


def _huber_irls(basis: np.ndarray, d: np.ndarray, params: np.ndarray,
                cfg: PlaneFitConfig) -> np.ndarray:
    """Huber iteratively-reweighted least squares refinement."""
    for _ in range(cfg.huber_iters):
        res = d - basis @ params
        sigma = max(_mad_sigma(res), 1e-12)
        k = cfg.huber_delta * sigma
        a = np.abs(res)
        w = np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))
        new = _lstsq(basis, d, weights=w)
        if new is None:
            break
        params = new
    return params


# ── the fit ───────────────────────────────────────────────────────────────────


def fit_road_plane(
    inv_depth: np.ndarray,
    cfg: Optional[PlaneFitConfig] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> PlaneFitResult:
    """RANSAC + Huber fit of the road surface in inverse-depth space.

    Args:
        inv_depth: [H, W] float inverse-depth map (the depth model's native
            output space; arbitrary affine scaling is fine — see module doc).
        cfg: fit configuration.
        valid_mask: optional [H, W] boolean mask of usable pixels (e.g. finite
            depth). Non-finite pixels are always excluded.

    Returns:
        :class:`PlaneFitResult`. On failure (degenerate input, too few seeds)
        ``trusted=False`` with a NaN-free zero surface and empty inlier mask.
    """
    cfg = cfg or PlaneFitConfig()
    d_full = np.asarray(inv_depth, dtype=np.float64)
    if d_full.ndim != 2:
        raise ValueError(f"inv_depth must be [H, W], got shape {d_full.shape}")
    h, w = d_full.shape

    def _failure(reason: str) -> PlaneFitResult:
        return PlaneFitResult(
            params=np.zeros(N_BASIS), sigma_mad=0.0, tau=0.0, inlier_ratio=0.0,
            inlier_mask=np.zeros((h, w), dtype=bool), n_seed=0,
            trusted=False, reason=reason,
        )

    finite = np.isfinite(d_full)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)

    seed_region = trapezoid_mask((h, w), cfg) & finite
    ys, xs = np.nonzero(seed_region)
    if ys.size < max(cfg.min_samples * 4, 32):
        return _failure("too_few_seed_points")

    rng = np.random.default_rng(cfg.seed)
    if ys.size > cfg.max_seed_points:
        pick = rng.choice(ys.size, size=cfg.max_seed_points, replace=False)
        ys, xs = ys[pick], xs[pick]
    n_seed = int(ys.size)

    u = (xs.astype(np.float64) + 0.5) / w
    v = (ys.astype(np.float64) + 0.5) / h
    basis = design_basis(u, v, quadratic=cfg.quadratic)
    d = d_full[ys, xs]

    # Absolute tau cap from the seed region's robust disparity span (see
    # PlaneFitConfig.max_tau_range_frac).
    span = float(np.percentile(d, 95) - np.percentile(d, 5))
    tau_cap = cfg.max_tau_range_frac * span if span > 0 else np.inf

    def _tau(sigma: float) -> float:
        return min(cfg.tau_scale * sigma, tau_cap)

    # Initial LS on all seed points -> sigma_hat -> tau
    params0 = _lstsq(basis, d)
    if params0 is None:
        return _failure("degenerate_initial_fit")
    sigma_hat = _mad_sigma(d - basis @ params0)
    if sigma_hat <= 0:
        # Perfectly planar input (synthetic / constant) — accept the LS fit.
        sigma_hat = 1e-12
    tau = _tau(sigma_hat)

    # RANSAC: 6-pt minimal solves
    best_params = params0
    best_inliers = int(np.count_nonzero(np.abs(d - basis @ params0) <= tau))
    m = cfg.min_samples
    for _ in range(cfg.ransac_iters):
        idx = rng.choice(n_seed, size=m, replace=False)
        cand = _lstsq(basis[idx], d[idx])
        if cand is None:
            continue
        n_in = int(np.count_nonzero(np.abs(d - basis @ cand) <= tau))
        if n_in > best_inliers:
            best_inliers, best_params = n_in, cand

    # Refit rounds: LS on inliers, re-estimate sigma/tau from the INLIER
    # residuals (the consensus set — this is what lets the inlier-ratio trust
    # gate fire on minority-plane failures), with the truncated-MAD
    # consistency correction so shrinkage does not compound across rounds.
    mad_corr = _truncated_mad_correction(cfg.tau_scale)
    params = best_params
    for _ in range(cfg.refit_rounds):
        res = d - basis @ params
        inl = np.abs(res) <= tau
        if inl.sum() < cfg.min_samples:
            break
        new = _lstsq(basis[inl], d[inl])
        if new is None:
            break
        params = new
        res = d - basis @ params
        inl = np.abs(res) <= tau
        sigma_hat = max(_mad_sigma(res[inl]) * mad_corr, 1e-12)
        tau = _tau(sigma_hat)

    # Final Huber IRLS on the inlier set
    inl = np.abs(d - basis @ params) <= tau
    if inl.sum() >= cfg.min_samples:
        params = _huber_irls(basis[inl], d[inl], params, cfg)

    # Final statistics (sigma_MAD over inliers, truncation-corrected)
    res_seed = d - basis @ params
    inl = np.abs(res_seed) <= tau
    inlier_ratio = float(inl.mean())
    sigma_mad = (MAD_TO_SIGMA * float(np.median(np.abs(res_seed[inl]))) * mad_corr
                 if inl.any() else 0.0)
    sigma_mad = max(sigma_mad, 1e-12)

    # Full-image road mask: plane-consistent pixels (the inlier mask IS the
    # road mask — anomaly pixels fall outside it and are recovered by the
    # hole filling in residual_labels).
    d_surf = evaluate_surface(params, (h, w))
    res_full = d_full - d_surf
    inlier_mask = finite & (np.abs(res_full) <= tau)

    trusted = inlier_ratio >= cfg.min_inlier_ratio
    return PlaneFitResult(
        params=params.astype(np.float64),
        sigma_mad=sigma_mad,
        tau=float(tau),
        inlier_ratio=inlier_ratio,
        inlier_mask=inlier_mask,
        n_seed=n_seed,
        trusted=trusted,
        reason="ok" if trusted else f"inlier_ratio {inlier_ratio:.2f} < {cfg.min_inlier_ratio}",
    )
