"""Common helpers shared by all SSL-pool dataset adapters.

Three concerns live here:

1. **Image normalization** — every dataset hands us images at different
   resolutions, modes, and formats. We materialize them all the same way:
   resize so the long side is ``DEFAULT_LONG_SIDE`` (preserving aspect, never
   upscaling), convert to RGB, save as JPEG quality ``DEFAULT_JPEG_QUALITY``.
2. **Integrity** — quick "is this file actually an image" check used both
   during ingestion and as a sanity sweep over the materialized pool.
3. **Download** — HTTP GET with Range-based resume, since dataset zips are
   multi-GB and Colab disconnects.

All three are deliberately small and side-effect-isolated to keep adapter
code thin.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageFile

# Allow loading slightly truncated images rather than raising. Some dataset
# archives contain images with off-by-one byte trailers; we prefer salvaging
# them to dropping rows.
ImageFile.LOAD_TRUNCATED_IMAGES = True

LOG = logging.getLogger(__name__)

#: Long side of materialized images. Matches downstream finetune ``imgsz``.
DEFAULT_LONG_SIDE = 640

#: JPEG quality for materialized images. Empirically a good size/quality knee.
DEFAULT_JPEG_QUALITY = 90


def resize_long_side(img: Image.Image, long_side: int) -> Image.Image:
    """Resize so ``max(h, w) == long_side``, preserving aspect.

    Never upscales — if the input is already smaller than ``long_side`` along
    its long edge, the image is returned unchanged. Upscaling would add no
    information and would inflate disk usage.
    """
    w, h = img.size
    longest = max(h, w)
    if longest <= long_side:
        return img
    scale = long_side / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert ``img`` to RGB mode.

    RGBA is composited over an opaque white background; other modes (L, CMYK,
    P, etc.) go through PIL's standard conversion. RGB inputs pass through.
    """
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def resize_and_save(
    src,
    dest_path: Path,
    long_side: int = DEFAULT_LONG_SIDE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Tuple[Tuple[int, int], Tuple[int, int], str]:
    """Open ``src``, normalize, save JPEG to ``dest_path``.

    ``src`` may be a path or any binary file-like object that PIL accepts
    (e.g. an ``io.BytesIO`` wrapping bytes read out of a zip/tar member).
    This lets adapters stream archive entries directly through without ever
    extracting them to a scratch file.

    Returns ``((orig_w, orig_h), (mat_w, mat_h), sha256_hex)`` so the caller
    can build a ``ManifestRow`` without re-opening the file. The hash is
    computed over the materialized JPEG bytes — bit-identical materialization
    yields a bit-identical hash, which is what we want for the cheap
    exact-duplicate check that runs ahead of pHash dedup in Faz 4.2.

    Raises whatever PIL raises on unreadable input; callers decide whether to
    skip or abort.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as img:
        img.load()  # force decode now so errors surface here, not later
        original_size = img.size  # (w, h)
        rgb = to_rgb(img)
        resized = resize_long_side(rgb, long_side)

        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        data = buf.getvalue()

    dest_path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    return original_size, resized.size, sha


def is_readable_image(path: Path) -> bool:
    """Best-effort check whether ``path`` can be opened as an image.

    Uses PIL's ``verify`` which decodes the header but not the full pixel
    stream — fast enough to run over the whole pool.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001 — verification failures take many shapes
        return False


def download_with_resume(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 1024,
    expected_size: Optional[int] = None,
    timeout: int = 60,
) -> int:
    """Download ``url`` to ``dest``, resuming from a partial file if present.

    Sends a ``Range: bytes=N-`` header when ``dest`` already has bytes on
    disk. Treats HTTP 416 ("range not satisfiable") as "already complete" and
    returns the existing size — this is what most CDNs return when the client
    asks for a range past EOF.

    If ``expected_size`` is given, raises ``IOError`` on mismatch after
    download completes. The caller is responsible for knowing the expected
    size (e.g. from a prior HEAD request); we don't HEAD here because some
    signed CDN URLs reject HEAD even when GET succeeds.
    """
    import requests  # local import keeps the module importable without it

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code == 416:
            LOG.info("Server reports complete download for %s (size=%d)", dest.name, existing)
            return existing
        r.raise_for_status()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    final = dest.stat().st_size
    if expected_size is not None and final != expected_size:
        raise IOError(
            f"Download size mismatch for {dest.name}: got {final}, expected {expected_size}"
        )
    return final
