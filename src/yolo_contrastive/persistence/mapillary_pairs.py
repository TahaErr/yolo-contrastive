"""Mapillary Graph API pair mining + download for REVISIT (offline stages 1-2).

Mines candidate co-located image pairs from repeated traversals of the same
streets (different sessions, months apart) and downloads their thumbnails.
GPS/compass only PROPOSE candidates — the homography trust gates in align.py
are the actual filter, so bad pairs cost only mining/alignment compute.

LOCKED PAIR GATE (see :class:`PairGateConfig`): haversine distance <= 10 m,
circular |delta heading| <= 25 deg, different sequence ids, delta-t >= 30
days, no panoramic/fisheye cameras. Greedy midpoint NMS (8 m) per sequence
pair plus a 3-pairs-per-location-cell cap keep one co-traversed street from
flooding the manifest.

Network access is fully injectable (``fetch_json`` / ``fetch_bytes``
callables) so unit tests run offline against canned Graph-API JSON; the
default transport lazily imports ``requests`` (E2) and reads the token from
the ``MAPILLARY_TOKEN`` env var. Thumbnail URLs are short-lived, so the
downloader re-queries them by image id at download time and is
skip-if-exists resumable.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "PairGateConfig",
    "DEFAULT_CITIES",
    "GRAPH_URL",
    "IMAGE_FIELDS",
    "get_token",
    "haversine_m",
    "heading_diff_deg",
    "iter_tiles",
    "fetch_tile_images",
    "mine_tile_pairs",
    "mine_pairs",
    "download_images",
]

GRAPH_URL = "https://graph.mapillary.com"
IMAGE_FIELDS = (
    "id,captured_at,compass_angle,computed_compass_angle,geometry,"
    "computed_geometry,sequence,is_pano,camera_type"
)
_MS_PER_DAY = 86_400_000.0


@dataclasses.dataclass(frozen=True)
class PairGateConfig:
    """Locked candidate-pair gates + anti-flooding knobs."""

    max_dist_m: float = 10.0
    max_heading_deg: float = 25.0
    min_dt_days: float = 30.0       # pre-registered fallback: 14 (never geometry)
    nms_radius_m: float = 8.0       # midpoint NMS per (seq_a, seq_b)
    cell_m: float = 10.0            # location-cell size for the per-cell cap
    max_pairs_per_cell: int = 3
    bad_camera_types: Tuple[str, ...] = ("spherical", "equirectangular", "fisheye")
    tile_m: float = 200.0
    tile_buffer_m: float = 20.0
    api_limit: int = 500


#: ~12 seed cities: RDD2022 countries (Japan, India, Czechia, Norway, US)
#: mixed with high-Mapillary-density cities. (name, lat, lon, radius_km)
DEFAULT_CITIES: List[Dict] = [
    {"name": "tokyo", "lat": 35.6762, "lon": 139.6503, "radius_km": 2.0},
    {"name": "nagoya", "lat": 35.1815, "lon": 136.9066, "radius_km": 2.0},
    {"name": "pune", "lat": 18.5204, "lon": 73.8567, "radius_km": 2.0},
    {"name": "prague", "lat": 50.0755, "lon": 14.4378, "radius_km": 2.0},
    {"name": "oslo", "lat": 59.9139, "lon": 10.7522, "radius_km": 2.0},
    {"name": "washington_dc", "lat": 38.9072, "lon": -77.0369, "radius_km": 2.0},
    {"name": "san_francisco", "lat": 37.7749, "lon": -122.4194, "radius_km": 2.0},
    {"name": "amsterdam", "lat": 52.3676, "lon": 4.9041, "radius_km": 2.0},
    {"name": "berlin", "lat": 52.5200, "lon": 13.4050, "radius_km": 2.0},
    {"name": "sao_paulo", "lat": -23.5505, "lon": -46.6333, "radius_km": 2.0},
    {"name": "melbourne", "lat": -37.8136, "lon": 144.9631, "radius_km": 2.0},
    {"name": "bangkok", "lat": 13.7563, "lon": 100.5018, "radius_km": 2.0},
]


def get_token() -> str:
    """Read the Mapillary access token from the MAPILLARY_TOKEN env var."""
    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "MAPILLARY_TOKEN is not set. Create a (free) token at "
            "https://www.mapillary.com/dashboard/developers and export it: "
            "set MAPILLARY_TOKEN=MLY|... (Windows) / export MAPILLARY_TOKEN=... (POSIX)."
        )
    return token


# ── geo helpers ───────────────────────────────────────────────────────────────


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def heading_diff_deg(a: float, b: float) -> float:
    """Circular absolute heading difference in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def iter_tiles(
    lat: float, lon: float, radius_km: float,
    tile_m: float = 200.0, buffer_m: float = 20.0,
) -> Iterator[Tuple[str, Tuple[float, float, float, float]]]:
    """Tile a square of half-side ``radius_km`` around (lat, lon) into
    ~``tile_m`` bboxes with a ``buffer_m`` overlap (edge-loss guard).

    Yields ``(tile_id, (min_lon, min_lat, max_lon, max_lat))``.
    """
    r_m = radius_km * 1000.0
    n = max(1, math.ceil(2.0 * r_m / tile_m))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * max(0.05, math.cos(math.radians(lat)))
    for iy in range(n):
        for ix in range(n):
            x0 = -r_m + ix * tile_m - buffer_m
            x1 = -r_m + (ix + 1) * tile_m + buffer_m
            y0 = -r_m + iy * tile_m - buffer_m
            y1 = -r_m + (iy + 1) * tile_m + buffer_m
            bbox = (
                lon + x0 / m_per_deg_lon, lat + y0 / m_per_deg_lat,
                lon + x1 / m_per_deg_lon, lat + y1 / m_per_deg_lat,
            )
            yield f"{ix}_{iy}", bbox


# ── default transports (lazy requests, E2) ────────────────────────────────────


def _default_fetch_json(token: str) -> Callable[[str, Dict], Dict]:
    def fetch(url: str, params: Dict) -> Dict:
        import requests  # lazy (persistence extra)

        params = dict(params)
        params["access_token"] = token
        for attempt in range(6):
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(60.0, 2.0 ** attempt))  # polite backoff
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()  # pragma: no cover - raise_for_status throws first

    return fetch


def _default_fetch_bytes() -> Callable[[str], bytes]:
    def fetch(url: str) -> bytes:
        import requests  # lazy

        for attempt in range(6):
            resp = requests.get(url, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            resp.raise_for_status()
            return resp.content
        resp.raise_for_status()
        return resp.content  # pragma: no cover

    return fetch


# ── API entry normalization ───────────────────────────────────────────────────


def _normalize_entry(e: Dict) -> Optional[Dict]:
    """Pick SfM-corrected geometry/heading when present; None if unusable."""
    geom = e.get("computed_geometry") or e.get("geometry")
    if not geom or "coordinates" not in geom:
        return None
    lon, lat = geom["coordinates"][:2]
    heading = e.get("computed_compass_angle")
    if heading is None:
        heading = e.get("compass_angle")
    if heading is None or e.get("captured_at") is None or e.get("id") is None:
        return None
    return {
        "id": str(e["id"]),
        "lon": float(lon), "lat": float(lat),
        "heading": float(heading),
        "captured_at": int(e["captured_at"]),
        "sequence": str(e.get("sequence", "")),
        "is_pano": bool(e.get("is_pano", False)),
        "camera_type": str(e.get("camera_type", "")).lower(),
    }


def fetch_tile_images(
    bbox: Tuple[float, float, float, float],
    fetch_json: Callable[[str, Dict], Dict],
    cfg: Optional[PairGateConfig] = None,
) -> List[Dict]:
    """GET /images for a bbox; returns normalized usable entries."""
    cfg = cfg or PairGateConfig()
    data = fetch_json(
        f"{GRAPH_URL}/images",
        {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "fields": IMAGE_FIELDS,
            "limit": cfg.api_limit,
        },
    )
    out = []
    for e in data.get("data", []):
        n = _normalize_entry(e)
        if n is not None:
            out.append(n)
    return out


# ── pair mining ───────────────────────────────────────────────────────────────


def _passes_gate(a: Dict, b: Dict, cfg: PairGateConfig) -> Optional[Dict]:
    """Apply the locked pair gate; returns the candidate dict or None."""
    if a["id"] == b["id"]:
        return None
    if a["is_pano"] or b["is_pano"]:
        return None
    if a["camera_type"] in cfg.bad_camera_types or b["camera_type"] in cfg.bad_camera_types:
        return None
    if a["sequence"] == b["sequence"]:
        return None
    dt_days = abs(a["captured_at"] - b["captured_at"]) / _MS_PER_DAY
    if dt_days < cfg.min_dt_days:
        return None
    hd = heading_diff_deg(a["heading"], b["heading"])
    if hd > cfg.max_heading_deg:
        return None
    dist = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
    if dist > cfg.max_dist_m:
        return None
    lo, hi = (a, b) if a["id"] <= b["id"] else (b, a)
    return {
        "pair_id": f"{lo['id']}_{hi['id']}",
        "a": lo, "b": hi,
        "dist_m": dist, "heading_diff": hd, "dt_days": dt_days,
        "mid_lat": (a["lat"] + b["lat"]) / 2.0,
        "mid_lon": (a["lon"] + b["lon"]) / 2.0,
    }


def mine_tile_pairs(images: List[Dict], cfg: Optional[PairGateConfig] = None) -> List[Dict]:
    """All gate-passing pairs in one tile, midpoint-NMS'd and cell-capped.

    NMS is greedy per (seq_a, seq_b) on pair midpoints (closest pairs first,
    8 m suppression radius); afterwards at most ``max_pairs_per_cell`` pairs
    survive per ~10 m location cell (multi-traversal streets).
    """
    cfg = cfg or PairGateConfig()
    cands: Dict[str, Dict] = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            c = _passes_gate(images[i], images[j], cfg)
            if c is not None:
                cands.setdefault(c["pair_id"], c)

    # greedy midpoint NMS per sequence pair
    by_seq: Dict[Tuple[str, str], List[Dict]] = {}
    for c in cands.values():
        key = tuple(sorted((c["a"]["sequence"], c["b"]["sequence"])))
        by_seq.setdefault(key, []).append(c)
    kept: List[Dict] = []
    for group in by_seq.values():
        group.sort(key=lambda c: (c["dist_m"], c["pair_id"]))
        accepted: List[Dict] = []
        for c in group:
            if all(
                haversine_m(c["mid_lat"], c["mid_lon"], k["mid_lat"], k["mid_lon"])
                >= cfg.nms_radius_m
                for k in accepted
            ):
                accepted.append(c)
        kept.extend(accepted)

    # per-location-cell cap
    cell_deg_lat = cfg.cell_m / 111_320.0
    out: List[Dict] = []
    counts: Dict[Tuple[int, int], int] = {}
    for c in sorted(kept, key=lambda c: (c["dist_m"], c["pair_id"])):
        m_per_deg_lon = 111_320.0 * max(0.05, math.cos(math.radians(c["mid_lat"])))
        cell = (
            int(c["mid_lat"] / cell_deg_lat),
            int(c["mid_lon"] / (cfg.cell_m / m_per_deg_lon)),
        )
        if counts.get(cell, 0) >= cfg.max_pairs_per_cell:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        out.append(c)
    return out


def mine_pairs(
    pairs_path,
    cities: Optional[Iterable[Dict]] = None,
    token: Optional[str] = None,
    fetch_json: Optional[Callable[[str, Dict], Dict]] = None,
    cfg: Optional[PairGateConfig] = None,
    max_pairs_per_city: int = 2500,
    sleep_s: float = 0.0,
) -> int:
    """Mine candidate pairs for every city tile; append to pairs.parquet.

    Idempotent (dedup on pair_id) and resumable. ``fetch_json`` is injectable
    for offline tests; the default needs MAPILLARY_TOKEN. Returns the number
    of NEW pairs written.
    """
    from . import pair_manifest as pm

    cfg = cfg or PairGateConfig()
    cities = list(cities) if cities is not None else DEFAULT_CITIES
    if fetch_json is None:
        fetch_json = _default_fetch_json(token or get_token())

    n_new = 0
    for city in cities:
        rows: List[Dict] = []
        for tile_id, bbox in iter_tiles(
            city["lat"], city["lon"], city.get("radius_km", 2.0),
            tile_m=cfg.tile_m, buffer_m=cfg.tile_buffer_m,
        ):
            images = fetch_tile_images(bbox, fetch_json, cfg)
            for c in mine_tile_pairs(images, cfg):
                a, b = c["a"], c["b"]
                rows.append(pm.new_pair_row(
                    pair_id=c["pair_id"],
                    img_a_id=a["id"], img_b_id=b["id"],
                    lon_a=a["lon"], lat_a=a["lat"],
                    lon_b=b["lon"], lat_b=b["lat"],
                    dist_m=c["dist_m"],
                    heading_a=a["heading"], heading_b=b["heading"],
                    heading_diff=c["heading_diff"],
                    captured_at_a=a["captured_at"], captured_at_b=b["captured_at"],
                    dt_days=c["dt_days"],
                    seq_a=a["sequence"], seq_b=b["sequence"],
                    city=city["name"], tile_id=f"{city['name']}_{tile_id}",
                ))
            if len(rows) >= max_pairs_per_city:
                rows = rows[:max_pairs_per_city]
                break
            if sleep_s:
                time.sleep(sleep_s)
        n_new += pm.append_pairs(pairs_path, rows)
    return n_new


# ── download ──────────────────────────────────────────────────────────────────


def download_images(
    pairs_path,
    root,
    token: Optional[str] = None,
    fetch_json: Optional[Callable[[str, Dict], Dict]] = None,
    fetch_bytes: Optional[Callable[[str], bytes]] = None,
    sleep_s: float = 0.05,
) -> int:
    """Download thumbnails for every queued pair into ``root/images/{id}.jpg``.

    Thumbnail URLs are short-lived, so each image's URL is re-queried by id
    at download time (thumb_2048_url, fallback thumb_1024_url). Skip-if-exists
    resumable; pairs whose two files exist get path_a/path_b filled and
    ``status="downloaded"``. Returns the number of files fetched.
    """
    from . import pair_manifest as pm

    if fetch_json is None:
        fetch_json = _default_fetch_json(token or get_token())
    if fetch_bytes is None:
        fetch_bytes = _default_fetch_bytes()

    img_dir = Path(root) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pm.read_pairs(pairs_path)
    todo = df[df["status"] == "queued"]
    fetched = 0
    updates: Dict[str, Dict] = {}

    def _ensure(image_id: str) -> Optional[Path]:
        nonlocal fetched
        path = img_dir / f"{image_id}.jpg"
        if path.exists():
            return path
        meta = fetch_json(f"{GRAPH_URL}/{image_id}",
                          {"fields": "thumb_2048_url,thumb_1024_url"})
        url = meta.get("thumb_2048_url") or meta.get("thumb_1024_url")
        if not url:
            return None
        data = fetch_bytes(url)
        if not data:
            return None
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic-ish: no truncated JPEGs on interrupt
        fetched += 1
        if sleep_s:
            time.sleep(sleep_s)
        return path

    for _, row in todo.iterrows():
        pa = _ensure(str(row["img_a_id"]))
        pb = _ensure(str(row["img_b_id"]))
        if pa is not None and pb is not None:
            updates[row["pair_id"]] = {
                "path_a": str(pa), "path_b": str(pb), "status": "downloaded",
            }
    pm.update_pairs(pairs_path, updates)
    return fetched
