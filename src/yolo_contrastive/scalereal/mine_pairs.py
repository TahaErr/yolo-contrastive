"""Offline GASP-Real pair miner — natural scale pairs labeled by metric depth.

Per pool image: square candidate boxes on a 3-level grid are embedded with a
FROZEN injectable embedder (DINOv2-S/14 in production, a stub in tests);
pairs of spatially disjoint, similar-content candidates whose METRIC depths
differ by a 1.5-6x ratio survive the gate ladder and are appended to the pair
parquet with the real apparent-scale label ``log_r = log(Z_A / Z_B)``.

Design facts enforced here (each maps to a measured failure):
    * The embedder only SELECTS pairs; supervision content comes from depth
      (frozen, offline, never in the loss graph — no R6/R7 violation, and no
      collapsing-EMA-matcher feedback loop).
    * NO synthetic rescaling anywhere: candidates are COORDINATES into the
      native image; the miner never crops-and-resizes patch pixels — the
      resampling-artifact channel that solved GASP's task does not exist.
    * Depth ratios are taken through depth_io.log_depth_ratio(), which
      HARD-FAILS on non-metric cache variants (affine-ambiguity guard).
    * Per-image stratified selection over four |log_r| bins (<= 4/bin,
      <= 16/image) — GASP's measured mutual-NN distribution-skew lesson.

Resumable by image_id set-difference against already-written rows; images
that yielded zero pairs are cheaply re-scanned on resume (documented
trade-off of the rows-only resume source).

Heavy deps (PIL, torch.hub DINOv2, pandas via pair_manifest) are imported
lazily inside functions (E2).

CLI::

    python -m yolo_contrastive.scalereal.mine_pairs \
        --manifest data/ssl_pool/manifest.parquet \
        --depth-cache /content/cache \
        --out /content/cache/scalereal/pairs_v1.parquet \
        --embedder dinov2_vits14
    python -m yolo_contrastive.scalereal.mine_pairs --audit 200 ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import ScaleRealConfig
from .depth_io import DepthCache, log_depth_ratio, patch_depth_stats
from .pair_manifest import PairRecord, append_pairs, existing_image_ids, read_pairs

LOG = logging.getLogger(__name__)

#: embedder protocol: (image [H, W, 3] float in [0,1], boxes_norm [N, 4]) ->
#: [N, D] L2-normalized embeddings.
EmbedFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ── candidate grid ────────────────────────────────────────────────────────────


def grid_candidate_boxes(h: int, w: int, cfg: ScaleRealConfig) -> np.ndarray:
    """Square candidate boxes [N, 4] (normalized xyxy) on the 3-level grid.

    Side = ``frac * min(H, W)`` per level, stride ``grid_stride_frac * side``,
    restricted to the central region with a ``central_margin`` HORIZONTAL
    margin (bounds lens distortion; pinhole s ∝ 1/Z degrades off-axis).
    """
    boxes: List[List[float]] = []
    short = float(min(h, w))
    x_lo = cfg.central_margin * w
    x_hi = (1.0 - cfg.central_margin) * w
    for frac in cfg.grid_fractions:
        side = frac * short
        stride = max(cfg.grid_stride_frac * side, 1.0)
        if x_hi - x_lo < side or h < side:
            continue
        xs = np.arange(x_lo, x_hi - side + 1e-6, stride)
        ys = np.arange(0.0, h - side + 1e-6, stride)
        for y0 in ys:
            for x0 in xs:
                boxes.append([x0 / w, y0 / h, (x0 + side) / w, (y0 + side) / h])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


# ── per-patch + pair gates ────────────────────────────────────────────────────


def texture_std(image: np.ndarray, box_norm: np.ndarray) -> float:
    """Grayscale std (image in [0, 1]) inside a normalized box."""
    h, w = image.shape[:2]
    x1 = int(np.floor(box_norm[0] * w))
    y1 = int(np.floor(box_norm[1] * h))
    x2 = max(int(np.ceil(box_norm[2] * w)), x1 + 1)
    y2 = max(int(np.ceil(box_norm[3] * h)), y1 + 1)
    patch = image[y1:y2, x1:x2]
    gray = patch.mean(axis=2) if patch.ndim == 3 else patch
    return float(gray.std())


def _expand_box(box: np.ndarray, factor: float) -> np.ndarray:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    hw = (box[2] - box[0]) / 2.0 * factor
    hh = (box[3] - box[1]) / 2.0 * factor
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def boxes_disjoint(box_a: np.ndarray, box_b: np.ndarray, expand: float = 1.25) -> bool:
    """True iff the ``expand``-scaled boxes have ZERO intersection."""
    a = _expand_box(np.asarray(box_a, dtype=np.float64), expand)
    b = _expand_box(np.asarray(box_b, dtype=np.float64), expand)
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix <= 0.0 or iy <= 0.0


# ── mining statistics ─────────────────────────────────────────────────────────


class MiningStats:
    """Gate-attrition counters + log_r histogram + per-source yield."""

    def __init__(self, cfg: ScaleRealConfig) -> None:
        self.cfg = cfg
        self.counters: Dict[str, int] = {
            "images_total": 0,
            "images_skipped_source": 0,
            "images_skipped_pano": 0,
            "images_no_depth": 0,
            "images_failed_plane_gate": 0,
            "images_already_mined": 0,
            "images_processed": 0,
            "images_with_pairs": 0,
            "candidates_total": 0,
            "patches_failed_texture": 0,
            "patches_failed_depth_validity": 0,
            "patches_failed_iqr": 0,
            "pairs_considered": 0,
            "pairs_failed_sim": 0,
            "pairs_failed_band": 0,
            "pairs_failed_overlap": 0,
            "pairs_eligible": 0,
            "pairs_written": 0,
        }
        n_bins = len(cfg.ratio_bin_edges) - 1
        self.log_r_hist = [0] * n_bins
        self.per_source: Dict[str, Dict[str, int]] = {}

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def record_pair(self, log_r: float, source: str = "unknown") -> None:
        edges = self.cfg.ratio_bin_edges
        a = abs(log_r)
        for i in range(len(edges) - 1):
            if edges[i] <= a < edges[i + 1] or (i == len(edges) - 2 and a == edges[-1]):
                self.log_r_hist[i] += 1
                break
        src = self.per_source.setdefault(source, {"images": 0, "pairs": 0})
        src["pairs"] += 1

    def record_image(self, source: str = "unknown", with_pairs: bool = False) -> None:
        src = self.per_source.setdefault(source, {"images": 0, "pairs": 0})
        src["images"] += 1
        if with_pairs:
            self.bump("images_with_pairs")

    def to_dict(self) -> Dict:
        processed = max(self.counters.get("images_processed", 0), 1)
        return {
            "counters": dict(self.counters),
            "log_r_hist_edges": list(self.cfg.ratio_bin_edges),
            "log_r_hist": list(self.log_r_hist),
            "per_source": self.per_source,
            "image_yield": self.counters.get("images_with_pairs", 0) / processed,
            "miner_version": self.cfg.miner_version,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ── core per-image miner (file-free, fully testable) ─────────────────────────


def mine_image_pairs(
    image: np.ndarray,
    inv_depth: np.ndarray,
    sidecar: Dict,
    embed_fn: EmbedFn,
    cfg: Optional[ScaleRealConfig] = None,
    on_road_mask: Optional[np.ndarray] = None,
    stats: Optional[MiningStats] = None,
    source: str = "unknown",
) -> List[Dict]:
    """Mine the stratified scale pairs of ONE image.

    Args:
        image: [H, W, 3] float array in [0, 1] (native materialized image).
        inv_depth: metric inverse-depth array from the shared cache (any
            resolution; boxes are normalized).
        sidecar: the image's depth sidecar — must be a METRIC variant
            (``log_depth_ratio`` hard-fails otherwise).
        embed_fn: frozen patch embedder (see :data:`EmbedFn`).
        cfg: thresholds (default :class:`ScaleRealConfig`).
        on_road_mask: optional bool [H', W'] TERRA plane-fit inlier mask;
            fills ``on_road_a/b`` (else NaN).
        stats: optional :class:`MiningStats` to update.
        source: dataset name for per-source stats.

    Returns:
        List of pair dicts (keys = PairRecord fields minus image_id/pair_id),
        ordered by descending similarity within the stratified budget.
    """
    cfg = cfg or ScaleRealConfig()
    stats = stats or MiningStats(cfg)
    h, w = image.shape[:2]

    candidates = grid_candidate_boxes(h, w, cfg)
    stats.bump("candidates_total", len(candidates))
    if len(candidates) < 2:
        return []

    # per-patch gates --------------------------------------------------------
    tex = np.array([texture_std(image, b) for b in candidates])
    depth = [patch_depth_stats(inv_depth, b, cfg.patch_central_fraction) for b in candidates]
    z = np.array([d["z"] for d in depth])
    iqr = np.array([d["iqr_ratio"] for d in depth])
    median_inv = np.array([d["median_inv"] for d in depth])

    ok_tex = tex >= cfg.texture_std_min
    ok_z = (z >= cfg.z_min_m) & (z <= cfg.z_max_m)
    ok_iqr = iqr <= cfg.iqr_ratio_max
    stats.bump("patches_failed_texture", int((~ok_tex).sum()))
    stats.bump("patches_failed_depth_validity", int((~ok_z).sum()))
    stats.bump("patches_failed_iqr", int((~ok_iqr).sum()))
    valid = np.flatnonzero(ok_tex & ok_z & ok_iqr)
    if len(valid) < 2:
        return []

    # embeddings only for surviving candidates (tokens computed once upstream
    # in the production DINO embedder; the interface hides that detail)
    emb = np.asarray(embed_fn(image, candidates[valid]), dtype=np.float32)
    if emb.shape[0] != len(valid):
        raise ValueError(
            f"embedder returned {emb.shape[0]} rows for {len(valid)} boxes"
        )
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, None)
    sim_mat = emb @ emb.T

    # pair gates --------------------------------------------------------------
    eligible: List[Dict] = []
    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            stats.bump("pairs_considered")
            i, j = int(valid[ii]), int(valid[jj])
            sim = float(sim_mat[ii, jj])
            if sim < cfg.sim_threshold:
                stats.bump("pairs_failed_sim")
                continue
            log_r = log_depth_ratio(median_inv[i], median_inv[j], sidecar)
            if not (cfg.log_ratio_min <= abs(log_r) <= cfg.log_ratio_max):
                stats.bump("pairs_failed_band")
                continue
            if not boxes_disjoint(candidates[i], candidates[j], cfg.box_expand):
                stats.bump("pairs_failed_overlap")
                continue
            stats.bump("pairs_eligible")
            eligible.append({
                "i": i, "j": j, "sim": sim, "log_r": float(log_r),
            })

    if not eligible:
        return []

    # stratified greedy selection over |log_r| bins ---------------------------
    eligible.sort(key=lambda p: -p["sim"])
    edges = cfg.ratio_bin_edges
    bin_counts = [0] * (len(edges) - 1)
    total = 0
    selected: List[Dict] = []
    for p in eligible:
        if total >= cfg.max_pairs_per_image:
            break
        a = abs(p["log_r"])
        b = None
        for k in range(len(edges) - 1):
            if edges[k] <= a < edges[k + 1] or (k == len(edges) - 2 and a == edges[-1]):
                b = k
                break
        if b is None or bin_counts[b] >= cfg.max_pairs_per_bin:
            continue
        bin_counts[b] += 1
        total += 1
        selected.append(p)

    if len(selected) < cfg.min_pairs_per_image:
        return []

    def _on_road(box: np.ndarray) -> float:
        if on_road_mask is None:
            return float("nan")
        mh, mw = on_road_mask.shape[:2]
        x1, y1 = int(box[0] * mw), int(box[1] * mh)
        x2, y2 = max(int(box[2] * mw), x1 + 1), max(int(box[3] * mh), y1 + 1)
        return float(on_road_mask[y1:y2, x1:x2].mean() >= 0.5)

    rows: List[Dict] = []
    for p in selected:
        i, j = p["i"], p["j"]
        ba, bb = candidates[i], candidates[j]
        rows.append({
            "box_a_x1": float(ba[0]), "box_a_y1": float(ba[1]),
            "box_a_x2": float(ba[2]), "box_a_y2": float(ba[3]),
            "box_b_x1": float(bb[0]), "box_b_y1": float(bb[1]),
            "box_b_x2": float(bb[2]), "box_b_y2": float(bb[3]),
            "log_r": p["log_r"],
            "z_a": float(z[i]), "z_b": float(z[j]),
            "sim": p["sim"],
            "texture_a": float(tex[i]), "texture_b": float(tex[j]),
            "depth_iqr_a": float(iqr[i]), "depth_iqr_b": float(iqr[j]),
            "on_road_a": _on_road(ba), "on_road_b": _on_road(bb),
            "miner_version": cfg.miner_version,
        })
        stats.record_pair(p["log_r"], source)
    return rows


# ── pool-level driver ─────────────────────────────────────────────────────────


def _load_image(path: str) -> np.ndarray:
    from PIL import Image  # lazy (E2)

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def mine_pool(
    manifest,
    depth_cache: DepthCache,
    out_path,
    embed_fn: EmbedFn,
    cfg: Optional[ScaleRealConfig] = None,
    plane_gate_fn: Optional[Callable[[str], Optional[bool]]] = None,
    on_road_mask_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None,
    limit: Optional[int] = None,
    flush_every: int = 200,
) -> MiningStats:
    """Mine the whole pool into the pair parquet (resumable).

    Args:
        manifest: SSL-pool manifest DataFrame or parquet path (needs
            ``image_id`` + ``materialized_path``; ``dataset`` used for
            per-source stats / source exclusion; an ``is_pano`` column is
            honored when present).
        depth_cache: METRIC-variant :class:`~.depth_io.DepthCache`.
        out_path: pair parquet output; ``mining_stats.json`` is written next
            to it.
        embed_fn: frozen patch embedder.
        plane_gate_fn: optional per-image TERRA plane-consistency trust gate
            (``False`` -> skip image; ``None``/``True`` -> keep). Graceful
            null when TERRA artifacts are absent.
        on_road_mask_fn: optional per-image plane-inlier mask provider for
            the ``on_road_*`` columns.
        limit: optional max images processed this call (smoke runs).
        flush_every: parquet append cadence (images).

    Returns:
        The final :class:`MiningStats` (also saved as JSON).
    """
    import pandas as pd  # lazy (E2)

    cfg = cfg or ScaleRealConfig()
    out_path = Path(out_path)
    stats = MiningStats(cfg)
    df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_parquet(manifest)
    already = existing_image_ids(out_path)

    buffer: List[PairRecord] = []
    processed = 0

    def _flush() -> None:
        nonlocal buffer
        if buffer:
            n = append_pairs(out_path, buffer)
            stats.bump("pairs_written", n)
            buffer = []
        stats.save(out_path.parent / "mining_stats.json")

    for row in df.itertuples(index=False):
        stats.bump("images_total")
        image_id = str(row.image_id)
        source = str(getattr(row, "dataset", "unknown"))
        if source in set(cfg.exclude_sources):
            stats.bump("images_skipped_source")
            continue
        if bool(getattr(row, "is_pano", False)):
            stats.bump("images_skipped_pano")
            continue
        if image_id in already:
            stats.bump("images_already_mined")
            continue
        if not depth_cache.has(image_id):
            stats.bump("images_no_depth")
            continue
        if plane_gate_fn is not None and plane_gate_fn(image_id) is False:
            stats.bump("images_failed_plane_gate")
            continue
        if limit is not None and processed >= limit:
            break

        image = _load_image(str(row.materialized_path))
        inv_depth = depth_cache.read(image_id)
        sidecar = depth_cache.read_sidecar(image_id)
        mask = on_road_mask_fn(image_id) if on_road_mask_fn is not None else None

        pair_dicts = mine_image_pairs(
            image, inv_depth, sidecar, embed_fn, cfg,
            on_road_mask=mask, stats=stats, source=source,
        )
        processed += 1
        stats.bump("images_processed")
        stats.record_image(source, with_pairs=bool(pair_dicts))
        for k, d in enumerate(pair_dicts):
            buffer.append(PairRecord(image_id=image_id,
                                     pair_id=f"{image_id}#p{k:03d}", **d))
        if processed % flush_every == 0:
            _flush()
            LOG.info("mined %d images (%d pairs written)", processed,
                     stats.counters["pairs_written"])

    _flush()
    return stats


# ── embedders ─────────────────────────────────────────────────────────────────


class GridStatsEmbedder:
    """Offline demo embedder: RoI-pooled coarse color/gradient statistics.

    A weak stand-in for DINOv2 used by the example script and smoke runs —
    similar textures land near each other, but real mining quality REQUIRES a
    semantic embedder. Pools a precomputed cell-statistics grid per box (no
    patch crop-and-resize anywhere, matching the DINO token-pooling shape of
    the production path).
    """

    def __init__(self, cell_px: int = 16, dim: int = 24) -> None:
        self.cell_px = int(cell_px)
        self.dim = int(dim)

    def __call__(self, image: np.ndarray, boxes_norm: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        c = self.cell_px
        gh, gw = max(h // c, 1), max(w // c, 1)
        cells = image[: gh * c, : gw * c].reshape(gh, c, gw, c, -1)
        mean_rgb = cells.mean(axis=(1, 3))                       # [gh, gw, 3]
        gray = cells.mean(axis=-1)                               # [gh, c, gw, c]
        std = gray.std(axis=(1, 3))[..., None]                   # [gh, gw, 1]
        gx = np.abs(np.diff(gray.mean(axis=(1, 3)), axis=1))
        gx = np.pad(gx, ((0, 0), (0, 1)))[..., None]
        gy = np.abs(np.diff(gray.mean(axis=(1, 3)), axis=0))
        gy = np.pad(gy, ((0, 1), (0, 0)))[..., None]
        grid = np.concatenate([mean_rgb, std, gx, gy], axis=-1)  # [gh, gw, 6]

        out = np.zeros((len(boxes_norm), grid.shape[-1]), dtype=np.float32)
        for k, b in enumerate(np.asarray(boxes_norm, dtype=np.float64)):
            x1 = int(np.floor(b[0] * gw))
            y1 = int(np.floor(b[1] * gh))
            x2 = max(int(np.ceil(b[2] * gw)), x1 + 1)
            y2 = max(int(np.ceil(b[3] * gh)), y1 + 1)
            out[k] = grid[y1:y2, x1:x2].mean(axis=(0, 1))
        out = out - out.mean(axis=0, keepdims=True)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-8, None)


class DINOv2Embedder:
    """Production embedder: frozen DINOv2-S/14 patch tokens, RoI-pooled.

    The token grid is computed ONCE per image (518 px, fp16 on CUDA) and each
    candidate's embedding is the L2-normalized mean of bilinearly RoI-pooled
    tokens — features are never persisted, and no patch pixels are resized.

    Heavy path: downloads via torch.hub on first use; GPU strongly
    recommended. Tests never construct this class (@slow territory).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: Optional[str] = None,
        input_size: int = 518,
        fp16: bool = True,
        roi_cells: int = 3,
    ) -> None:
        import torch  # local: keep module import light

        self.torch = torch
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.input_size = int(input_size)
        self.patch = 14
        self.roi_cells = int(roi_cells)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()
        for p in self.model.parameters():
            p.requires_grad_(False)  # frozen teacher-side selector (R6-safe)

    def __call__(self, image: np.ndarray, boxes_norm: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from torchvision.ops import roi_align  # lazy heavy dep (E2)

        side = (self.input_size // self.patch) * self.patch
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
        x = F.interpolate(x.float(), size=(side, side), mode="bilinear",
                          align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)
        if self.fp16:
            x = x.half()
        with torch.no_grad():
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # [1, N, D]
        g = side // self.patch
        grid = tokens.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()      # [1, D, g, g]
        boxes_px = torch.as_tensor(
            np.asarray(boxes_norm, dtype=np.float32) * g, device=self.device
        )
        rois = torch.cat([torch.zeros(len(boxes_px), 1, device=self.device),
                          boxes_px], dim=1)
        pooled = roi_align(grid, rois, output_size=self.roi_cells,
                           spatial_scale=1.0, aligned=True)
        emb = pooled.mean(dim=(2, 3))                                       # [N, D]
        emb = F.normalize(emb, dim=1)
        return emb.cpu().numpy()


def build_embedder(name: str, **kw) -> EmbedFn:
    """CLI embedder factory: ``"stub"`` (offline) or a DINOv2 hub name."""
    if name == "stub":
        return GridStatsEmbedder(**kw)
    return DINOv2Embedder(model_name=name, **kw)


# ── audit rendering ───────────────────────────────────────────────────────────


def render_audit(
    pairs_path,
    manifest,
    out_dir,
    n: int = 200,
    cfg: Optional[ScaleRealConfig] = None,
    seed: int = 0,
) -> List[str]:
    """Render N stratified pairs side-by-side with log_r overlays (PIL).

    Crops are pasted at NATIVE resolution (no resizing — the human auditor
    must see the true apparent-scale difference). Go/no-go per wf2_ac.md:
    >= 60% judged same-content with plausible relative scale, >= 40% of
    images yielding >= 2 pairs, all four ratio bins populated.
    """
    import pandas as pd  # lazy
    from PIL import Image, ImageDraw  # lazy

    cfg = cfg or ScaleRealConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(pairs_path)
    if pairs.empty:
        return []
    mdf = manifest if isinstance(manifest, pd.DataFrame) else pd.read_parquet(manifest)
    path_by_id = dict(zip(mdf["image_id"].astype(str), mdf["materialized_path"]))

    rng = np.random.default_rng(seed)
    edges = cfg.ratio_bin_edges
    abs_lr = pairs["log_r"].abs().to_numpy()
    per_bin = max(n // (len(edges) - 1), 1)
    chosen: List[int] = []
    for k in range(len(edges) - 1):
        idx = np.flatnonzero((abs_lr >= edges[k]) & (abs_lr < edges[k + 1] + 1e-12))
        if len(idx):
            take = min(per_bin, len(idx))
            chosen.extend(rng.choice(idx, size=take, replace=False).tolist())

    written: List[str] = []
    for rank, ridx in enumerate(chosen):
        row = pairs.iloc[int(ridx)]
        img_path = path_by_id.get(str(row["image_id"]))
        if img_path is None or not Path(str(img_path)).exists():
            continue
        with Image.open(str(img_path)) as im:
            im = im.convert("RGB")
            w, h = im.size
            crops = []
            for pfx in ("box_a", "box_b"):
                x1 = int(row[f"{pfx}_x1"] * w)
                y1 = int(row[f"{pfx}_y1"] * h)
                x2 = int(row[f"{pfx}_x2"] * w)
                y2 = int(row[f"{pfx}_y2"] * h)
                crops.append(im.crop((x1, y1, x2, y2)))
        gap, pad = 8, 22
        cw = crops[0].width + crops[1].width + gap
        ch = max(crops[0].height, crops[1].height) + pad
        canvas = Image.new("RGB", (cw, ch), (24, 24, 24))
        canvas.paste(crops[0], (0, pad))
        canvas.paste(crops[1], (crops[0].width + gap, pad))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (4, 4),
            f"log_r={row['log_r']:+.3f}  z_a={row['z_a']:.1f}m  "
            f"z_b={row['z_b']:.1f}m  sim={row['sim']:.2f}",
            fill=(255, 255, 0),
        )
        path = out_dir / f"audit_{rank:03d}_{abs(row['log_r']):.2f}.png"
        canvas.save(path)
        written.append(str(path))
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="GASP-Real offline pair miner (metric-depth scale labels)"
    )
    parser.add_argument("--manifest", required=True,
                        help="SSL-pool manifest parquet")
    parser.add_argument("--depth-cache", required=True,
                        help="shared cache root (expects depth/{variant}/ under it)")
    parser.add_argument("--variant", default="dav2_metric_outdoor_small",
                        help="metric depth-cache variant")
    parser.add_argument("--out", required=True, help="output pair parquet")
    parser.add_argument("--embedder", default="dinov2_vits14",
                        help="'stub' (offline demo) or a DINOv2 torch.hub name")
    parser.add_argument("--sim-threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="max images this run (smoke)")
    parser.add_argument("--exclude-sources", nargs="*", default=None)
    parser.add_argument("--audit", type=int, default=0, metavar="N",
                        help="render N stratified audit pairs instead of mining")
    args = parser.parse_args(argv)

    cfg = ScaleRealConfig()
    if args.sim_threshold is not None:
        cfg.sim_threshold = args.sim_threshold
    if args.exclude_sources is not None:
        cfg.exclude_sources = tuple(args.exclude_sources)

    if args.audit:
        out_dir = Path(args.out).parent / "audit"
        files = render_audit(args.out, args.manifest, out_dir, n=args.audit, cfg=cfg)
        print(f"[scalereal] wrote {len(files)} audit mosaics to {out_dir}")
        return 0

    cache = DepthCache(args.depth_cache, variant=args.variant)
    embed_fn = build_embedder(args.embedder)
    stats = mine_pool(args.manifest, cache, args.out, embed_fn, cfg, limit=args.limit)
    d = stats.to_dict()
    print(f"[scalereal] pairs_written={d['counters']['pairs_written']} "
          f"image_yield={d['image_yield']:.1%} log_r_hist={d['log_r_hist']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
