"""Depth-cache I/O for GASP-Real — uint16 inverse-depth PNGs + variant sidecars.

SHARED CACHE FORMAT (owned by geoteach/depth_cache.py; this module is the
scalereal-side consumer and implements the identical on-disk contract so the
two channels share one cache root):

    {cache_root}/depth/{variant}/{image_id}.png      uint16 PNG, half the
                                                     materialized resolution,
                                                     encoding INVERSE depth
    {cache_root}/depth/{variant}/{image_id}.json     per-image sidecar
    {cache_root}/depth/{variant}/metadata.json       per-variant metadata

Sidecar schema::

    {"v_min": float, "v_max": float, "encoding": "inverse_depth",
     "units": "1/m" | "affine", "model_tag": str, "variant": str,
     "cache_version": int}

Decode: ``inv_z = v_min + png / 65535 * (v_max - v_min)``.

Variants:
    * ``dav2_rel_base``               — TERRA's relative checkpoint
                                        (units "affine": d = s * (1/Z) + t
                                        with unknown per-image s, t).
    * ``dav2_metric_outdoor_small``   — the metric-outdoor checkpoint
                                        (units "1/m"; Z clipped to
                                        [0.5, 80] m before encoding — uint16
                                        over 1/Z concentrates precision
                                        near-field).

AFFINE-AMBIGUITY GUARD (load-bearing): on the relative checkpoint,
``(d_A - t) / (d_B - t)`` depends on the unknown per-image shift ``t``, so
depth RATIOS are mathematically meaningless. :func:`log_depth_ratio` therefore
HARD-FAILS (``ValueError``) unless the sidecar declares a metric variant
(units "1/m"). The weaker true requirement: any per-image MULTIPLICATIVE
error in Z cancels in the ratio; only additive/nonlinear distortion hurts.

image_id may contain '/' — slashes are preserved as sub-directories
(collision-free, mirrors dual_teacher/teacher_cache.py).

GEOTEACH DIALECT COMPATIBILITY: the geoteach/depth_cache.py writer that
landed alongside this module uses ``{root}/{tag}/`` layout (no ``depth/``
segment) and sidecar keys ``d_min``/``d_max`` plus ``metric``/``depth_unit``
flags instead of ``v_min``/``v_max`` + ``units``. The PNG payload and decode
formula are identical. :func:`normalize_sidecar` maps that dialect onto the
spec schema, every reader here accepts both, and ``DepthCache(subdir="")``
points at a geoteach-tag directory directly — the metric guard fires
correctly on caches written by either side.

PNG codec: cv2 imported lazily (E2), PIL fallback; both paths use in-memory
encode/decode + pathlib byte I/O so Windows unicode paths are safe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
CACHE_VERSION = 1

#: ``units`` value that marks a metric cache (depth ratios are valid).
METRIC_UNITS = "1/m"
#: ``units`` value of affine-ambiguous relative caches (ratios are INVALID).
AFFINE_UNITS = "affine"
#: ``encoding`` value — the PNG stores inverse depth.
INVERSE_DEPTH_ENCODING = "inverse_depth"

#: Default units per known variant (used when ``DepthCache.save`` is called
#: without explicit units).
VARIANT_UNITS = {
    "dav2_metric_outdoor_small": METRIC_UNITS,
    "dav2_rel_base": AFFINE_UNITS,
}

#: Metric-variant Z clip band [m] applied before encoding.
METRIC_Z_CLIP = (0.5, 80.0)


def normalize_sidecar(sidecar: Dict) -> Dict:
    """Map any known sidecar dialect onto the spec schema (non-destructive).

    Handles the geoteach/depth_cache.py dialect: ``d_min``/``d_max`` become
    ``v_min``/``v_max``; ``metric=True`` + ``depth_unit`` in {"meters", "m"}
    becomes ``units="1/m"`` (geoteach inverts metric depth before caching);
    a missing ``units`` on a non-metric geoteach sidecar becomes "affine".
    ``encoding`` defaults to inverse_depth when quantization bounds exist —
    both writers store inverse depth by construction.
    """
    out = dict(sidecar)
    if "v_min" not in out and "d_min" in out:
        out["v_min"] = out["d_min"]
    if "v_max" not in out and "d_max" in out:
        out["v_max"] = out["d_max"]
    if "units" not in out and ("metric" in out or "d_min" in out):
        is_metric = bool(out.get("metric")) and \
            str(out.get("depth_unit", "meters")).lower() in ("meters", "m", "metre")
        out["units"] = METRIC_UNITS if is_metric else AFFINE_UNITS
    if "encoding" not in out and "v_min" in out and "v_max" in out:
        out["encoding"] = INVERSE_DEPTH_ENCODING
    return out


# ── uint16 PNG codec (cv2 lazy, PIL fallback, Windows-safe byte I/O) ─────────


def _encode_png16(arr: np.ndarray) -> bytes:
    """Encode a uint16 2-D array as PNG bytes."""
    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ValueError(f"expected 2-D uint16 array, got {arr.dtype} {arr.shape}")
    try:
        import cv2  # lazy optional dep (E2)

        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf.tobytes())
    except ImportError:
        import io

        from PIL import Image  # PIL ships with ultralytics/torchvision installs

        bio = io.BytesIO()
        Image.fromarray(arr).save(bio, format="PNG")
        return bio.getvalue()


def _decode_png16(data: bytes) -> np.ndarray:
    """Decode PNG bytes to a uint16 2-D array."""
    try:
        import cv2  # lazy optional dep (E2)

        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError("cv2.imdecode failed")
    except ImportError:
        import io

        from PIL import Image

        arr = np.array(Image.open(io.BytesIO(data)))
    if arr.ndim != 2:
        raise ValueError(f"depth PNG must be single-channel, got shape {arr.shape}")
    return arr.astype(np.uint16, copy=False)


# ── encode / decode inverse depth <-> uint16 ─────────────────────────────────


def encode_inverse_depth(inv_depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Quantize a float inverse-depth array to uint16 + (v_min, v_max)."""
    inv = np.asarray(inv_depth, dtype=np.float64)
    if inv.ndim != 2:
        raise ValueError(f"inverse depth must be 2-D, got shape {inv.shape}")
    if not np.isfinite(inv).all():
        raise ValueError("inverse depth contains non-finite values")
    v_min = float(inv.min())
    v_max = float(inv.max())
    if v_max > v_min:
        q = np.round((inv - v_min) / (v_max - v_min) * 65535.0)
    else:
        q = np.zeros_like(inv)
    return q.astype(np.uint16), v_min, v_max


def decode_inverse_depth(png: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    """Invert :func:`encode_inverse_depth`: ``inv_z = v_min + png/65535*(v_max-v_min)``."""
    return (v_min + png.astype(np.float32) / 65535.0 * (v_max - v_min)).astype(np.float32)


# ── the cache ────────────────────────────────────────────────────────────────


class DepthCache:
    """Variant-tagged inverse-depth cache at ``{cache_root}/depth/{variant}/``.

    Resumable per-image caching following dual_teacher/teacher_cache.py:
    ``has()`` / ``save()`` / ``read()``, ``metadata.json``, slash-preserving
    image ids.

    Args:
        cache_root: cache root shared with TERRA (geoteach).
        variant: cache variant tag. GASP-Real requires the metric variant
            ``"dav2_metric_outdoor_small"`` — the relative variant is read-
            compatible but :func:`log_depth_ratio` will refuse its sidecars.
        model_tag: model identifier written into sidecars on ``save``.
        subdir: path segment between root and variant (default ``"depth"``
            per the spec layout). Pass ``""`` to read a geoteach-style
            ``{root}/{tag}/`` cache directly (variant = the geoteach tag).
    """

    def __init__(
        self,
        cache_root: str,
        variant: str = "dav2_metric_outdoor_small",
        model_tag: Optional[str] = None,
        subdir: str = "depth",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.variant = str(variant)
        self.model_tag = model_tag if model_tag is not None else self.variant
        self.cache_dir = (
            self.cache_root / subdir / self.variant if subdir
            else self.cache_root / self.variant
        )

    # ── paths ─────────────────────────────────────────────────────────────

    def _png_path(self, image_id: str) -> Path:
        return self.cache_dir / f"{image_id}.png"

    def _sidecar_path(self, image_id: str) -> Path:
        return self.cache_dir / f"{image_id}.json"

    # ── single-image I/O ──────────────────────────────────────────────────

    def has(self, image_id: str) -> bool:
        """True if both PNG and sidecar exist for ``image_id``."""
        return self._png_path(image_id).exists() and self._sidecar_path(image_id).exists()

    def __contains__(self, image_id: str) -> bool:
        return self.has(image_id)

    def save(
        self,
        image_id: str,
        inv_depth: np.ndarray,
        units: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Write one inverse-depth array + sidecar.

        Production writing is owned by the geoteach depth-cache CLI; this
        writer exists for tests, the synthetic generator and small local runs
        and produces byte-identical format.

        Args:
            image_id: cache key (slashes preserved as sub-dirs).
            inv_depth: float 2-D inverse-depth array (1/Z for metric units).
            units: ``"1/m"`` or ``"affine"``; defaults from VARIANT_UNITS.
            extra: optional additional sidecar fields.
        """
        if units is None:
            units = VARIANT_UNITS.get(self.variant)
            if units is None:
                raise ValueError(
                    f"unknown variant {self.variant!r} — pass units='1/m' or 'affine' explicitly"
                )
        inv = np.asarray(inv_depth, dtype=np.float32)
        if units == METRIC_UNITS:
            # Clip Z to the metric band before encoding (spec: [0.5, 80] m).
            lo, hi = METRIC_Z_CLIP
            inv = np.clip(inv, 1.0 / hi, 1.0 / lo)
        png, v_min, v_max = encode_inverse_depth(inv)
        sidecar = {
            "v_min": v_min,
            "v_max": v_max,
            "encoding": INVERSE_DEPTH_ENCODING,
            "units": units,
            "model_tag": self.model_tag,
            "variant": self.variant,
            "cache_version": CACHE_VERSION,
        }
        if extra:
            sidecar.update(extra)
        png_path = self._png_path(image_id)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(_encode_png16(png))
        self._sidecar_path(image_id).write_text(
            json.dumps(sidecar, indent=0), encoding="utf-8"
        )

    def read_sidecar(self, image_id: str) -> Dict:
        """Load the per-image sidecar dict (dialects normalized)."""
        path = self._sidecar_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"No depth sidecar for {image_id!r} ({path})")
        return normalize_sidecar(json.loads(path.read_text(encoding="utf-8")))

    def read(self, image_id: str) -> np.ndarray:
        """Decode the float32 inverse-depth array for ``image_id``."""
        png_path = self._png_path(image_id)
        if not png_path.exists():
            raise FileNotFoundError(f"Not cached: {image_id!r} ({png_path})")
        sidecar = self.read_sidecar(image_id)
        if sidecar.get("encoding") != INVERSE_DEPTH_ENCODING:
            raise ValueError(
                f"unsupported encoding {sidecar.get('encoding')!r} for {image_id!r}"
            )
        png = _decode_png16(png_path.read_bytes())
        return decode_inverse_depth(png, float(sidecar["v_min"]), float(sidecar["v_max"]))

    def __len__(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(1 for _ in self.cache_dir.rglob("*.png"))

    # ── metadata ──────────────────────────────────────────────────────────

    def save_metadata(self, extra: Optional[Dict] = None) -> None:
        """Write the per-variant ``metadata.json``."""
        meta = {
            "variant": self.variant,
            "model_tag": self.model_tag,
            "encoding": INVERSE_DEPTH_ENCODING,
            "cache_version": CACHE_VERSION,
        }
        if extra:
            meta.update(extra)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / _METADATA_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DepthCache(variant={self.variant!r}, dir={str(self.cache_dir)!r})"


# ── the soundness gate ───────────────────────────────────────────────────────


def require_metric_sidecar(sidecar: Dict) -> None:
    """Raise ``ValueError`` unless ``sidecar`` declares a metric depth cache.

    This is the affine-ambiguity guard enforced IN CODE: the relative DA-v2
    checkpoint outputs ``d = s * (1/Z) + t`` with unknown per-image (s, t),
    so ``(d_A - t) / (d_B - t)`` depends on the unknown ``t`` and depth ratios
    carry no scale information. Only metric caches (units "1/m") may feed
    ``log_r`` labels. Geoteach-dialect sidecars are normalized first, so the
    guard fires correctly on caches written by either module.
    """
    sidecar = normalize_sidecar(sidecar)
    units = sidecar.get("units")
    if units != METRIC_UNITS:
        raise ValueError(
            f"depth ratios are mathematically invalid on a non-metric depth cache "
            f"(variant={sidecar.get('variant')!r}, units={units!r}): the relative "
            f"checkpoint's output d = s*(1/Z) + t has an unknown per-image shift t, "
            f"so (d_A - t)/(d_B - t) is not a depth ratio. Use the metric variant "
            f"(units {METRIC_UNITS!r}, e.g. 'dav2_metric_outdoor_small')."
        )
    if sidecar.get("encoding") != INVERSE_DEPTH_ENCODING:
        raise ValueError(
            f"unsupported depth encoding {sidecar.get('encoding')!r}; "
            f"expected {INVERSE_DEPTH_ENCODING!r}"
        )


def log_depth_ratio(median_inv_a: float, median_inv_b: float, sidecar: Dict) -> float:
    """The pair label: ``log_r(A->B) = log(Z_A) - log(Z_B) = log(inv_b / inv_a)``.

    Under pinhole projection apparent scale s ∝ f/Z, so for same-physical-size
    content at metric depths Z_A, Z_B the apparent-scale ratio is
    ``s_B / s_A = Z_A / Z_B``; the ordered-pair label is antisymmetric under
    swap. Worked example: Z_A = 5 m, Z_B = 20 m -> B appears 4x smaller,
    log_r = -1.386.

    Args:
        median_inv_a: median inverse depth (1/Z_A) of patch A.
        median_inv_b: median inverse depth (1/Z_B) of patch B.
        sidecar: the per-image sidecar of the cache both medians came from.

    Raises:
        ValueError: if the sidecar is not a metric variant (units "1/m") —
            the affine-ambiguity unsoundness is a hard runtime error, not a
            docs footnote — or if either inverse depth is non-positive.
    """
    require_metric_sidecar(sidecar)
    a = float(median_inv_a)
    b = float(median_inv_b)
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"inverse depths must be positive, got {a}, {b}")
    return float(np.log(b) - np.log(a))


# ── per-patch depth statistics ───────────────────────────────────────────────


def central_sub_box(box_xyxy: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Shrink a normalized xyxy box around its center to ``fraction`` of each side."""
    box = np.asarray(box_xyxy, dtype=np.float64)
    cx = (box[..., 0] + box[..., 2]) / 2.0
    cy = (box[..., 1] + box[..., 3]) / 2.0
    hw = (box[..., 2] - box[..., 0]) / 2.0 * fraction
    hh = (box[..., 3] - box[..., 1]) / 2.0 * fraction
    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], axis=-1)


def patch_depth_stats(
    inv_depth: np.ndarray,
    box_xyxy_norm: np.ndarray,
    central_fraction: float = 0.5,
) -> Dict[str, float]:
    """Median + IQR statistics of inverse depth inside a patch.

    ``Z_patch`` is the median metric Z over the central ``central_fraction``
    sub-box (read from the half-res cache; boxes are normalized so resolution
    does not matter). The depth-coherence gate uses
    ``iqr_ratio = IQR(1/Z) / median(1/Z)``.

    Args:
        inv_depth: [H, W] float inverse-depth array (any resolution).
        box_xyxy_norm: normalized [x1, y1, x2, y2] in [0, 1] of the image.
        central_fraction: central sub-box fraction (default 0.5).

    Returns:
        ``{"median_inv": float, "iqr_ratio": float, "z": float, "n_px": int}``
        — ``z`` is 1 / median_inv (inf when median_inv <= 0).
    """
    inv = np.asarray(inv_depth, dtype=np.float32)
    if inv.ndim != 2:
        raise ValueError(f"inv_depth must be 2-D, got shape {inv.shape}")
    h, w = inv.shape
    sub = central_sub_box(np.asarray(box_xyxy_norm, dtype=np.float64), central_fraction)
    x1 = int(np.floor(np.clip(sub[0], 0.0, 1.0) * w))
    y1 = int(np.floor(np.clip(sub[1], 0.0, 1.0) * h))
    x2 = int(np.ceil(np.clip(sub[2], 0.0, 1.0) * w))
    y2 = int(np.ceil(np.clip(sub[3], 0.0, 1.0) * h))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    region = inv[y1:y2, x1:x2].reshape(-1)
    median = float(np.median(region))
    q25, q75 = np.percentile(region, [25.0, 75.0])
    iqr = float(q75 - q25)
    iqr_ratio = float(iqr / median) if median > 0 else float("inf")
    z = float(1.0 / median) if median > 0 else float("inf")
    return {"median_inv": median, "iqr_ratio": iqr_ratio, "z": z, "n_px": int(region.size)}
