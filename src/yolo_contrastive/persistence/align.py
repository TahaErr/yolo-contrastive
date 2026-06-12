"""Cross-traversal image alignment for REVISIT (road-plane homographies).

Offline stage 3 of the pair factory: ORB (SIFT fallback) keypoints restricted
to the bottom ``roi_bottom_frac`` of each image (the road region), MAGSAC
(RANSAC fallback) homography, hard trust gates, and conversion to a
resolution-independent NORMALIZED homography stored in the pairs manifest:

    x_b = H_norm @ x_a,  with x in [0, 1]^2 homogeneous coords, h22 := 1.

Everything the TRAINING path needs (point/box warping, point-in-quad overlap
tests) is pure numpy — cv2 is imported lazily and only by the offline
alignment functions (E2).

Error budget: the 2.5 px RMSE gate at the 1024 working resolution is
~1.25 px at 512 training resolution — far below one P3 cell (8 px), so
correspondence labels are sub-cell accurate by construction.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "AlignConfig",
    "AlignResult",
    "align_pair",
    "align_manifest",
    "to_normalized_h",
    "from_normalized_h",
    "warp_points_h",
    "warp_boxes_h",
    "roi_rect",
    "quad_from_rect",
    "points_in_quad",
    "is_convex_quad",
    "shrink_quad",
    "overlap_fraction",
    "degeneracy_ok",
    "h_from_row",
    "h_to_row",
]


@dataclasses.dataclass(frozen=True)
class AlignConfig:
    """All alignment knobs and trust gates (locked; see wf2 spec)."""

    long_side: int = 1024            # working resolution (long image side)
    roi_bottom_frac: float = 0.6     # keypoints only in the bottom fraction (road)
    orb_features: int = 4000
    lowe_ratio: float = 0.75
    ransac_px: float = 3.0           # reprojection threshold at working res
    max_iters: int = 5000
    confidence: float = 0.999
    # trust gates — ALL must pass, else the pair is rejected
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25   # inliers / post-ratio-test matches
    max_rmse_px: float = 2.5         # inlier reprojection RMSE at working res
    min_overlap_frac: float = 0.35   # A-ROI grid points landing inside B
    # degeneracy guard on the normalized H
    det_min: float = 0.1
    det_max: float = 10.0
    scale_min: float = 0.33
    scale_max: float = 3.0
    min_matches: int = 8             # bare minimum to attempt findHomography


@dataclasses.dataclass
class AlignResult:
    """Outcome of :func:`align_pair` (stats are at the working resolution)."""

    h_norm: Optional[np.ndarray]     # [3, 3] normalized homography or None
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_rmse: float = float("inf")
    overlap_frac: float = 0.0
    method: str = "none"             # "orb" | "sift" | "none"
    ok: bool = False
    reason: str = ""


# ── pure-numpy geometry (used by the training path — no cv2) ─────────────────


def warp_points_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to [N, 2] points. Points with |w| ~ 0 map to inf."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1) @ np.asarray(h, dtype=np.float64).T
    w = ph[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ph[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.inf
    return out


def warp_boxes_h(h: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Warp [N, 4] xyxy boxes: 4 corners through H, take the axis-aligned bbox."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes.copy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack(
        [np.stack([x1, y1], 1), np.stack([x2, y1], 1),
         np.stack([x2, y2], 1), np.stack([x1, y2], 1)], axis=1,
    )  # [N, 4, 2]
    warped = warp_points_h(h, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    out = np.stack(
        [warped[:, :, 0].min(1), warped[:, :, 1].min(1),
         warped[:, :, 0].max(1), warped[:, :, 1].max(1)], axis=1,
    )
    return out


def roi_rect(roi_bottom_frac: float) -> Tuple[float, float, float, float]:
    """Normalized xyxy rect of the bottom road ROI."""
    return (0.0, 1.0 - float(roi_bottom_frac), 1.0, 1.0)


def quad_from_rect(rect: Tuple[float, float, float, float]) -> np.ndarray:
    """Rect (x1, y1, x2, y2) -> [4, 2] corners (tl, tr, br, bl)."""
    x1, y1, x2, y2 = rect
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def points_in_quad(pts: np.ndarray, quad: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Pure-numpy point-in-convex-quad test via edge cross products.

    Works for either vertex orientation; non-finite quads or points are
    treated as outside. Returns bool [N].
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(quad)):
        return np.zeros(pts.shape[0], dtype=bool)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = np.where(finite[:, None], pts, 0.0)  # mask non-finite (reported False)
    edges = np.roll(quad, -1, axis=0) - quad                       # [4, 2]
    rel = pts[:, None, :] - quad[None, :, :]                       # [N, 4, 2]
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    inside = np.all(cross >= -eps, axis=1) | np.all(cross <= eps, axis=1)
    return inside & finite


def is_convex_quad(quad: np.ndarray, eps: float = 1e-12) -> bool:
    """True iff the 4 vertices form a finite convex quadrilateral."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(quad)):
        return False
    edges = np.roll(quad, -1, axis=0) - quad
    nxt = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
    if np.any(np.abs(cross) < eps):  # degenerate (collinear) corner
        return False
    return bool(np.all(cross > 0) or np.all(cross < 0))


def shrink_quad(quad: np.ndarray, margin: float) -> np.ndarray:
    """Shrink a quad toward its centroid by a fractional ``margin`` (the 2%
    overlap margin in the transient-evidence rule)."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = quad.mean(axis=0, keepdims=True)
    return c + (quad - c) * (1.0 - 2.0 * float(margin))


def overlap_fraction(
    h_norm: np.ndarray, roi_bottom_frac: float, grid_n: int = 20
) -> float:
    """Fraction of A's ROI grid points whose image under H lands inside B's
    bounds ([0, 1]^2). Pure numpy."""
    x1, y1, x2, y2 = roi_rect(roi_bottom_frac)
    xs = np.linspace(x1, x2, grid_n)
    ys = np.linspace(y1, y2, grid_n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    warped = warp_points_h(h_norm, pts)
    finite = np.all(np.isfinite(warped), axis=1)
    inside = (
        finite
        & (warped[:, 0] >= 0.0) & (warped[:, 0] <= 1.0)
        & (warped[:, 1] >= 0.0) & (warped[:, 1] <= 1.0)
    )
    return float(inside.mean())


def degeneracy_ok(
    h_norm: np.ndarray, cfg: AlignConfig
) -> Tuple[bool, str]:
    """Degeneracy guard on the NORMALIZED homography:

    det(upper 2x2) in (det_min, det_max); per-axis scale change (singular
    values of the upper 2x2) in (scale_min, scale_max); the warped ROI
    quadrilateral must stay convex.
    """
    h = np.asarray(h_norm, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        return False, "non-finite homography"
    a = h[:2, :2]
    det = float(np.linalg.det(a))
    if not (cfg.det_min < det < cfg.det_max):
        return False, f"det {det:.4f} outside ({cfg.det_min}, {cfg.det_max})"
    sv = np.linalg.svd(a, compute_uv=False)
    if not (cfg.scale_min < sv.min() and sv.max() < cfg.scale_max):
        return False, (f"scale change ({sv.min():.3f}, {sv.max():.3f}) outside "
                       f"({cfg.scale_min}, {cfg.scale_max})")
    quad = warp_points_h(h, quad_from_rect(roi_rect(cfg.roi_bottom_frac)))
    if not is_convex_quad(quad):
        return False, "warped ROI quad not convex"
    return True, ""


# ── pixel <-> normalized H conversion ─────────────────────────────────────────


def _scale_mat(w: float, h: float) -> np.ndarray:
    return np.diag([float(w), float(h), 1.0])


def to_normalized_h(
    h_px: np.ndarray, shape_a: Tuple[int, int], shape_b: Tuple[int, int]
) -> np.ndarray:
    """Pixel homography -> normalized: H_norm = S_b^{-1} @ H_px @ S_a, h22 := 1.

    ``shape_*`` are (height, width) of the images H_px was estimated on.
    Direction locked: x_b = H @ x_a in both spaces.
    """
    ha, wa = shape_a
    hb, wb = shape_b
    hn = np.linalg.inv(_scale_mat(wb, hb)) @ np.asarray(h_px, np.float64) @ _scale_mat(wa, ha)
    if abs(hn[2, 2]) < 1e-12:
        raise ValueError("degenerate homography: h22 ~ 0 after normalization")
    return hn / hn[2, 2]


def from_normalized_h(
    h_norm: np.ndarray, shape_a: Tuple[int, int], shape_b: Tuple[int, int]
) -> np.ndarray:
    """Inverse of :func:`to_normalized_h` (back to pixel coords, h22 := 1)."""
    ha, wa = shape_a
    hb, wb = shape_b
    hp = _scale_mat(wb, hb) @ np.asarray(h_norm, np.float64) @ np.linalg.inv(_scale_mat(wa, ha))
    if abs(hp[2, 2]) < 1e-12:
        raise ValueError("degenerate homography: h22 ~ 0 after denormalization")
    return hp / hp[2, 2]


def h_to_row(h_norm: np.ndarray) -> dict:
    """Flatten a normalized H (h22 == 1) into the 8 manifest columns."""
    h = np.asarray(h_norm, dtype=np.float64)
    h = h / h[2, 2]
    return {
        "h00": h[0, 0], "h01": h[0, 1], "h02": h[0, 2],
        "h10": h[1, 0], "h11": h[1, 1], "h12": h[1, 2],
        "h20": h[2, 0], "h21": h[2, 1],
    }


def h_from_row(row) -> np.ndarray:
    """Rebuild the normalized 3x3 H from a pairs-manifest row (dict or Series)."""
    return np.array(
        [[row["h00"], row["h01"], row["h02"]],
         [row["h10"], row["h11"], row["h12"]],
         [row["h20"], row["h21"], 1.0]],
        dtype=np.float64,
    )


# ── offline alignment (lazy cv2) ──────────────────────────────────────────────


def _resize_long_side(gray: np.ndarray, long_side: int):
    """Resize so the long side equals ``long_side``; returns (image, scale)."""
    import cv2  # lazy (E2)

    h, w = gray.shape[:2]
    s = long_side / max(h, w)
    if abs(s - 1.0) < 1e-9:
        return gray, 1.0
    return cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), s


def _roi_mask(shape: Tuple[int, int], roi_bottom_frac: float) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(round(h * (1.0 - roi_bottom_frac))):, :] = 255
    return mask


def _detect_and_match(gray_a, gray_b, mask_a, mask_b, cfg: AlignConfig, method: str):
    """Keypoints + ratio-test matches. Returns (pts_a, pts_b) float32 [N, 2]."""
    import cv2  # lazy

    if method == "orb":
        det = cv2.ORB_create(nfeatures=cfg.orb_features)
        norm = cv2.NORM_HAMMING
    elif method == "sift":
        det = cv2.SIFT_create()
        norm = cv2.NORM_L2
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown method {method!r}")

    kp_a, des_a = det.detectAndCompute(gray_a, mask_a)
    kp_b, des_b = det.detectAndCompute(gray_b, mask_b)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return None, None
    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < cfg.lowe_ratio * n.distance]
    if len(good) < cfg.min_matches:
        return None, None
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def _estimate_h(pts_a, pts_b, cfg: AlignConfig):
    """MAGSAC (RANSAC fallback) homography. Returns (H_px, inlier_mask)."""
    import cv2  # lazy

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    h, mask = cv2.findHomography(
        pts_a, pts_b, method=method,
        ransacReprojThreshold=cfg.ransac_px,
        maxIters=cfg.max_iters, confidence=cfg.confidence,
    )
    if h is None or mask is None:
        return None, None
    return h, mask.ravel().astype(bool)


def _attempt(gray_a, gray_b, cfg: AlignConfig, method: str) -> AlignResult:
    """One full alignment attempt with a given detector; gates applied."""
    mask_a = _roi_mask(gray_a.shape, cfg.roi_bottom_frac)
    mask_b = _roi_mask(gray_b.shape, cfg.roi_bottom_frac)
    pts_a, pts_b = _detect_and_match(gray_a, gray_b, mask_a, mask_b, cfg, method)
    if pts_a is None:
        return AlignResult(None, method=method, reason="too few matches")
    h_px, inl = _estimate_h(pts_a, pts_b, cfg)
    if h_px is None:
        return AlignResult(None, method=method, reason="findHomography failed")

    n_in = int(inl.sum())
    ratio = n_in / max(1, len(pts_a))
    proj = warp_points_h(h_px, pts_a[inl].astype(np.float64))
    rmse = float(np.sqrt(np.mean(np.sum((proj - pts_b[inl]) ** 2, axis=1)))) if n_in else math.inf

    h_norm = to_normalized_h(h_px, gray_a.shape[:2], gray_b.shape[:2])
    ovl = overlap_fraction(h_norm, cfg.roi_bottom_frac)

    res = AlignResult(h_norm, n_inliers=n_in, inlier_ratio=float(ratio),
                      reproj_rmse=rmse, overlap_frac=ovl, method=method)
    # trust gates (ALL must pass)
    if n_in < cfg.min_inliers:
        res.reason = f"n_inliers {n_in} < {cfg.min_inliers}"
    elif ratio < cfg.min_inlier_ratio:
        res.reason = f"inlier_ratio {ratio:.3f} < {cfg.min_inlier_ratio}"
    elif rmse > cfg.max_rmse_px:
        res.reason = f"rmse {rmse:.2f}px > {cfg.max_rmse_px}px"
    elif ovl < cfg.min_overlap_frac:
        res.reason = f"overlap {ovl:.2f} < {cfg.min_overlap_frac}"
    else:
        deg_ok, deg_reason = degeneracy_ok(h_norm, cfg)
        if not deg_ok:
            res.reason = f"degenerate: {deg_reason}"
        else:
            res.ok = True
    return res


def align_pair(
    gray_a: np.ndarray, gray_b: np.ndarray, cfg: Optional[AlignConfig] = None
) -> AlignResult:
    """Align two grayscale captures of the same location.

    ORB first; if any gate fails, one retry with SIFT (core cv2 >= 4.4).
    Returns the best attempt (preferring an ``ok`` one).
    """
    cfg = cfg or AlignConfig()
    a, _ = _resize_long_side(np.asarray(gray_a), cfg.long_side)
    b, _ = _resize_long_side(np.asarray(gray_b), cfg.long_side)

    res = _attempt(a, b, cfg, "orb")
    if res.ok:
        return res
    sift = _attempt(a, b, cfg, "sift")
    return sift if (sift.ok or sift.n_inliers > res.n_inliers) else res


# ── manifest-driven stage runner ──────────────────────────────────────────────


def _align_one(args) -> Tuple[str, AlignResult]:
    """Top-level worker (picklable for multiprocessing)."""
    import cv2  # lazy

    pair_id, path_a, path_b, cfg = args
    ga = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    gb = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if ga is None or gb is None:
        return pair_id, AlignResult(None, reason="image unreadable")
    return pair_id, align_pair(ga, gb, cfg)


def align_manifest(
    pairs_path,
    cfg: Optional[AlignConfig] = None,
    workers: int = 0,
    statuses: Tuple[str, ...] = ("downloaded",),
) -> dict:
    """Run alignment over every pair in ``statuses``; write H + stats back.

    Resumable: already-aligned/rejected pairs are untouched. Returns
    ``{"aligned": n, "rejected": n}``.
    """
    from . import pair_manifest as pm

    cfg = cfg or AlignConfig()
    df = pm.read_pairs(pairs_path)
    todo = df[df["status"].isin(statuses)]
    jobs = [(r["pair_id"], r["path_a"], r["path_b"], cfg) for _, r in todo.iterrows()]

    results = []
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_align_one, jobs))
    else:
        results = [_align_one(j) for j in jobs]

    updates, n_ok, n_rej = {}, 0, 0
    for pair_id, res in results:
        cols = {
            "n_inliers": int(res.n_inliers),
            "inlier_ratio": float(res.inlier_ratio),
            "reproj_rmse": float(res.reproj_rmse),
            "overlap_frac": float(res.overlap_frac),
            "align_method": res.method,
            "align_ok": bool(res.ok),
            "status": "aligned" if res.ok else "rejected",
        }
        if res.ok:
            cols.update(h_to_row(res.h_norm))
            n_ok += 1
        else:
            n_rej += 1
        updates[pair_id] = cols
    pm.update_pairs(pairs_path, updates)
    return {"aligned": n_ok, "rejected": n_rej}
