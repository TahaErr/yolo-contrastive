"""Inverse-depth cache: Depth-Anything-V2 over the pool, resumable PNG cache.

TERRA Stage 0, step 1: run a foundation monocular depth model (default
Depth-Anything-V2-Small via the transformers ``depth-estimation`` pipeline,
fp16, batched) over the unlabeled pool and cache the inverse-depth maps as
**uint16 PNGs at half resolution** with a per-image JSON sidecar.

Why labels, not features (vs dual_teacher/teacher_cache.py): cached features
cannot be geometrically transformed, so cached-feature pipelines reproduce
the documented K2 spatial-misalignment bug class. Inverse-depth maps are
spatial LABELS — the pool loader transforms them jointly with the image (R5),
making that bug class impossible by construction.

Cache layout (mirrors teacher_cache.py; image_id slashes become sub-dirs)::

    {root}/{tag}/
        {image_id}.png      # uint16, min-max normalized inverse depth
        {image_id}.json     # {"d_min", "d_max", "metric", "model_name", ...}
        metadata.json

Sidecar fields: ``d_min/d_max`` undo the uint16 quantization; ``metric`` is
True when a metric Depth-Anything-V2 checkpoint was used — the sidecar then
also carries ``depth_unit`` and ``max_depth`` so |z| thresholds can be mapped
to meters. Plane fitting (plane_fit.py) is affine-invariant, so BOTH metric
and relative checkpoints feed the same downstream pipeline unchanged.

All heavy deps (transformers, PIL, cv2, torch) are imported lazily inside
functions (E2) — importing this module needs numpy only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_CACHE_VERSION = 1
_U16_MAX = 65535

#: Default relative-depth checkpoint (ViT-S). Metric variants (e.g.
#: "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf") are detected
#: by name and inverted to inverse depth before caching.
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

#: Metric-depth clip band [m] applied BEFORE inversion. Without it a single
#: sky/horizon return (hundreds of meters) or sub-half-meter hood pixel
#: stretches the per-image [d_min, d_max] quantization range and collapses
#: the road band onto a handful of uint16 codes. MUST stay in sync with
#: ``yolo_contrastive.scalereal.depth_io.METRIC_Z_CLIP`` (the shared-cache
#: consumer); the band is recorded in the sidecar as ``z_clip``.
METRIC_Z_CLIP = (0.5, 80.0)


class DepthCache:
    """Read/write uint16 inverse-depth PNGs + JSON sidecars.

    Args:
        root: cache root directory.
        tag: sub-directory naming this depth model configuration (different
            checkpoints/resolutions must use different tags).
    """

    def __init__(self, root, tag: str = "depth_anything_v2_small") -> None:
        self.root = Path(root)
        self.tag = str(tag)
        self.cache_dir = self.root / self.tag

    # ── paths ────────────────────────────────────────────────────────────

    def png_path(self, image_id: str) -> Path:
        return self.cache_dir / f"{image_id}.png"

    def sidecar_path(self, image_id: str) -> Path:
        return self.cache_dir / f"{image_id}.json"

    def has(self, image_id: str) -> bool:
        """True if both the PNG and its sidecar exist."""
        return self.png_path(image_id).exists() and self.sidecar_path(image_id).exists()

    def __contains__(self, image_id: str) -> bool:
        return self.has(image_id)

    def __len__(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(1 for _ in self.cache_dir.rglob("*.png"))

    def image_ids(self) -> List[str]:
        """All cached image ids (relative paths without the .png suffix)."""
        if not self.cache_dir.exists():
            return []
        return sorted(
            p.relative_to(self.cache_dir).with_suffix("").as_posix()
            for p in self.cache_dir.rglob("*.png")
        )

    # ── single-image I/O ─────────────────────────────────────────────────

    def save(self, image_id: str, inv_depth: np.ndarray,
             meta: Optional[Dict] = None) -> None:
        """Quantize a float inverse-depth map to uint16 PNG + JSON sidecar.

        Args:
            image_id: cache key (slashes become sub-directories).
            inv_depth: [H, W] float inverse depth (any affine scale).
            meta: extra sidecar fields (model name, metric info, original
                image size, ...). ``d_min``/``d_max`` are always written.
        """
        import cv2  # lazy: optional [pretrain] extra

        d = np.asarray(inv_depth, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError(f"inv_depth must be [H, W], got shape {d.shape}")
        finite = np.isfinite(d)
        if not finite.any():
            raise ValueError(f"inv_depth for {image_id!r} has no finite values")
        d_min = float(d[finite].min())
        d_max = float(d[finite].max())
        scale = (d_max - d_min) or 1.0
        q = np.zeros(d.shape, dtype=np.uint16)
        q[finite] = np.round((d[finite] - d_min) / scale * _U16_MAX).astype(np.uint16)

        png = self.png_path(image_id)
        png.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(png), q):
            raise IOError(f"cv2.imwrite failed for {png}")

        sidecar = {"d_min": d_min, "d_max": d_max,
                   "cache_h": int(d.shape[0]), "cache_w": int(d.shape[1])}
        if meta:
            sidecar.update(meta)
        self.sidecar_path(image_id).write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )

    def load(self, image_id: str) -> Tuple[np.ndarray, Dict]:
        """Load ``(inv_depth float32 [H, W], sidecar dict)`` for image_id."""
        import cv2  # lazy

        png = self.png_path(image_id)
        if not self.has(image_id):
            raise FileNotFoundError(f"Not cached: {image_id} ({png})")
        q = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if q is None:
            raise IOError(f"cv2.imread failed for {png}")
        if q.ndim == 3:  # some cv2 builds return [H, W, 1] for gray PNGs
            q = q[..., 0]
        meta = json.loads(self.sidecar_path(image_id).read_text(encoding="utf-8"))
        d_min, d_max = float(meta["d_min"]), float(meta["d_max"])
        scale = (d_max - d_min) or 1.0
        d = q.astype(np.float32) / _U16_MAX * scale + d_min
        return d, meta

    # ── metadata ─────────────────────────────────────────────────────────

    def save_metadata(self, extra: Optional[Dict] = None) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {"tag": self.tag, "cache_version": _CACHE_VERSION,
                "n_cached": len(self)}
        if extra:
            meta.update(extra)
        (self.cache_dir / _METADATA_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def load_metadata(self) -> Dict:
        path = self.cache_dir / _METADATA_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"No metadata at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DepthCache(dir={self.cache_dir}, n_cached={len(self)})"


# ── model runner ──────────────────────────────────────────────────────────────


def _is_metric_model(model_name: str) -> bool:
    return "metric" in model_name.lower()


def _load_image(src):
    """(lazy PIL) Load an image source into a PIL RGB image.

    Accepts a path/str or an ``[H, W, 3]`` uint8 RGB numpy array.
    """
    from PIL import Image  # lazy: ships with the ultralytics/transformers stack

    if isinstance(src, np.ndarray):
        return Image.fromarray(src).convert("RGB")
    return Image.open(str(src)).convert("RGB")


def _build_pipeline(model_name: str, device: Optional[str], fp16: bool):
    """(lazy transformers) Build the depth-estimation pipeline."""
    import torch  # local: keeps module import numpy-only
    from transformers import pipeline  # lazy: optional heavy dep

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (fp16 and "cuda" in str(device)) else torch.float32
    return pipeline("depth-estimation", model=model_name, device=device,
                    torch_dtype=dtype)


def run_depth_anything(
    images: Iterable[Tuple[str, object]],
    cache: DepthCache,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 8,
    fp16: bool = True,
    half_resolution: bool = True,
    skip_existing: bool = True,
    pipe=None,
    write_metadata: bool = True,
    log_every: int = 200,
) -> Dict[str, int]:
    """Populate ``cache`` with inverse-depth maps for an image stream.

    Resumable and idempotent: with ``skip_existing=True`` (default) already
    cached ids are skipped, so an interrupted run resumes cleanly — the
    teacher_cache.py pattern.

    Args:
        images: iterable of ``(image_id, source)`` where source is an image
            path or an ``[H, W, 3]`` uint8 RGB array.
        cache: target :class:`DepthCache`.
        model_name: HF checkpoint. Relative checkpoints output inverse depth
            directly; metric checkpoints (name contains "Metric") output
            metric depth, which is inverted to ``1/depth`` before caching and
            flagged in the sidecar (``metric=True, depth_unit="meters"``).
        device: "cuda" / "cpu" / None (auto).
        batch_size: pipeline batch size (fp16 ViT-S at 518px fits batch 8-16
            on an 8 GB GPU; A100 takes 32+).
        fp16: half-precision inference on CUDA.
        half_resolution: cache at half the ORIGINAL image resolution (the
            wf2 disk budget: ~40 GB for the 181K pool).
        skip_existing: resume support.
        pipe: pre-built/injected pipeline (tests use a stub; None = build the
            real transformers pipeline lazily).
        write_metadata: write metadata.json at the end.
        log_every: progress log cadence.

    Returns:
        ``{"scanned", "skipped", "computed", "errors"}`` counters.
    """
    stats = {"scanned": 0, "skipped": 0, "computed": 0, "errors": 0}
    metric = _is_metric_model(model_name)

    pending_ids: List[str] = []
    pending_imgs: List[object] = []

    def _flush() -> None:
        nonlocal pending_ids, pending_imgs
        if not pending_ids:
            return
        nonlocal pipe
        if pipe is None:
            pipe = _build_pipeline(model_name, device, fp16)
        try:
            pil_images = [_load_image(s) for s in pending_imgs]
            outputs = pipe(pil_images, batch_size=len(pil_images))
            if isinstance(outputs, dict):  # single-image pipelines
                outputs = [outputs]
            for image_id, pil, out in zip(pending_ids, pil_images, outputs):
                try:
                    _cache_one(cache, image_id, pil, out, model_name, metric,
                               half_resolution)
                    stats["computed"] += 1
                except Exception as exc:  # noqa: BLE001 — per-image isolation
                    LOG.warning("Depth cache failed on %s: %s", image_id, exc)
                    stats["errors"] += 1
        except Exception as exc:  # noqa: BLE001 — batch-level model failures
            LOG.warning("Depth batch failed (%d images): %s", len(pending_ids), exc)
            stats["errors"] += len(pending_ids)
        pending_ids, pending_imgs = [], []

    for image_id, src in images:
        stats["scanned"] += 1
        if skip_existing and cache.has(image_id):
            stats["skipped"] += 1
        else:
            pending_ids.append(image_id)
            pending_imgs.append(src)
            if len(pending_ids) >= batch_size:
                _flush()
        if log_every and stats["scanned"] % log_every == 0:
            LOG.info("Depth cache progress: %s", stats)
    _flush()

    if write_metadata:
        cache.save_metadata(extra={
            "model_name": model_name, "metric": metric,
            "half_resolution": half_resolution, "fp16": fp16,
        })
    LOG.info("Depth cache build done: %s", stats)
    return stats


def _cache_one(cache: DepthCache, image_id: str, pil_image, output,
               model_name: str, metric: bool, half_resolution: bool) -> None:
    """Convert one pipeline output to inverse depth and write it."""
    import torch  # local

    pred = output["predicted_depth"]
    if not torch.is_tensor(pred):
        pred = torch.as_tensor(np.asarray(pred))
    pred = pred.detach().float().cpu()
    if pred.ndim == 3:  # [1, H, W]
        pred = pred[0]

    orig_w, orig_h = pil_image.size
    out_h = max(1, orig_h // 2) if half_resolution else orig_h
    out_w = max(1, orig_w // 2) if half_resolution else orig_w
    if pred.shape != (out_h, out_w):
        pred = torch.nn.functional.interpolate(
            pred[None, None], size=(out_h, out_w),
            mode="bilinear", align_corners=False,
        )[0, 0]

    arr = pred.numpy()
    meta: Dict = {
        "model_name": model_name,
        "metric": metric,
        "orig_h": int(orig_h), "orig_w": int(orig_w),
    }
    if metric:
        # Metric checkpoints output depth in meters — clamp Z to the shared
        # [0.5, 80] m band (METRIC_Z_CLIP, = scalereal.depth_io.METRIC_Z_CLIP)
        # BEFORE inversion so far-field/sky returns cannot stretch the uint16
        # quantization range and collapse the road band, then invert to
        # inverse depth (plane fitting runs in inverse-depth space). The clip
        # band is recorded in the sidecar so |z| thresholds can be mapped
        # back to meters downstream.
        lo, hi = METRIC_Z_CLIP
        meta["depth_unit"] = "meters"
        meta["max_depth"] = float(np.nanmax(arr))
        meta["z_clip"] = [lo, hi]
        arr = 1.0 / np.clip(arr, lo, hi)
        meta["inverse_of_metric_depth"] = True
    cache.save(image_id, arr.astype(np.float32), meta=meta)
