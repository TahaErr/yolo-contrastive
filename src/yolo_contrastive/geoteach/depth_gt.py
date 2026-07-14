"""DepthGT — sparse/holed inverse-depth ground-truth store (B1 depth training).

The plain :class:`~yolo_contrastive.geoteach.depth_cache.DepthCache` stores a dense
inverse-depth map with NO per-pixel validity mask (it just drops non-finite pixels on
save). Real sensor GT is the opposite of dense: LiDAR is sparse and stereo disparity has
holes (sky, occlusion, `disparity==0` sentinel). Supervising a from-scratch depth net on
that GT REQUIRES the mask — you back-prop only where the measurement exists.

So B1 uses this sibling store: one ``.npz`` per image keyed by the same ``image_id`` scheme
as the pool, holding ``inv`` (float16 inverse depth) + ``valid`` (bool mask) + a small meta
dict. It is the training-GT store; the model's *predictions* over the 181K pool are cached
separately in the model-agnostic ``DepthCache`` (via the ``pipe=`` seam) so the geometry
consumer (`plane_fit`/`residual_labels`/`TerraChannel`) is reused unchanged.

Pure numpy; no torch, no cv2 at import (E2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


class DepthGT:
    """Per-image inverse-depth + validity-mask GT store (``.npz`` under ``{root}/{tag}/``)."""

    def __init__(self, root: str, tag: str = "depth_gt") -> None:
        self.root = Path(root)
        self.tag = tag
        self.dir = self.root / tag

    def path(self, image_id: str) -> Path:
        """``.npz`` path for ``image_id`` (slashes become sub-directories)."""
        return self.dir / f"{image_id}.npz"

    def has(self, image_id: str) -> bool:
        return self.path(image_id).exists()

    def __contains__(self, image_id: str) -> bool:
        return self.has(image_id)

    def save(
        self,
        image_id: str,
        inv_depth: np.ndarray,
        valid: np.ndarray,
        meta: Optional[Dict] = None,
    ) -> Path:
        """Save ``inv_depth`` (float, [H,W]) + ``valid`` (bool, [H,W]) for ``image_id``.

        ``inv_depth`` is stored as float16 (ample for a supervision target); invalid
        pixels are zeroed so the file compresses well. ``meta`` (e.g. camera params,
        source) is JSON-encoded alongside.
        """
        inv = np.asarray(inv_depth, dtype=np.float32)
        m = np.asarray(valid, dtype=bool)
        if inv.shape != m.shape:
            raise ValueError(f"inv {inv.shape} and valid {m.shape} shape mismatch")
        inv = np.where(m, inv, 0.0).astype(np.float16)
        p = self.path(image_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, inv=inv, valid=m.astype(np.uint8),
            meta=np.frombuffer(json.dumps(meta or {}).encode("utf-8"), dtype=np.uint8),
        )
        return p

    def load(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Return ``(inv_depth float32 [H,W], valid bool [H,W], meta dict)``."""
        with np.load(self.path(image_id), allow_pickle=False) as z:
            inv = z["inv"].astype(np.float32)
            valid = z["valid"].astype(bool)
            meta_bytes = z["meta"].tobytes() if "meta" in z else b"{}"
        meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
        return inv, valid, meta

    def image_ids(self) -> List[str]:
        """All stored ``image_id``s (relative to ``{root}/{tag}``, POSIX slashes)."""
        if not self.dir.exists():
            return []
        return sorted(
            p.relative_to(self.dir).with_suffix("").as_posix()
            for p in self.dir.rglob("*.npz")
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.image_ids())

    def __len__(self) -> int:
        return len(self.image_ids())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DepthGT(dir={self.dir}, n={len(self)})"


def masked_downsample(
    inv: np.ndarray,
    valid: np.ndarray,
    out_hw: Tuple[int, int],
    min_valid_frac: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample an inverse-depth map + validity mask WITHOUT bleeding across holes.

    Naive bilinear resize averages the ``valid==0`` sentinel (often 0) into real depth
    and corrupts hole borders. Instead we area-average the *masked* signal and the mask
    separately: ``inv_ds = area_pool(inv*valid) / area_pool(valid)``; a target pixel is
    valid iff its valid-fraction exceeds ``min_valid_frac``.

    Args:
        inv: [H,W] inverse depth. valid: [H,W] bool. out_hw: (h, w) target.
    Returns:
        (inv_ds [h,w] float32, valid_ds [h,w] bool).
    """
    import cv2  # lazy (E2)
    inv = np.asarray(inv, dtype=np.float32)
    m = np.asarray(valid, dtype=np.float32)
    oh, ow = int(out_hw[0]), int(out_hw[1])
    sum_ds = cv2.resize(inv * m, (ow, oh), interpolation=cv2.INTER_AREA)
    cnt_ds = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA)
    valid_ds = cnt_ds > min_valid_frac
    inv_ds = np.zeros_like(sum_ds)
    np.divide(sum_ds, cnt_ds, out=inv_ds, where=valid_ds)
    return inv_ds.astype(np.float32), valid_ds
