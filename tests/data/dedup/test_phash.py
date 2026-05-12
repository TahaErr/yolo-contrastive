"""Tests for the pHash module.

We use natural-looking pattern images (gradients, checkerboards) rather
than solid colors. Solid-color pHashes collapse to the same DC-only DCT
output and would make our different-image assertions misleading.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from yolo_contrastive.data.dedup.phash import (
    compute_phash,
    compute_pool_phashes,
    hamming_distance,
    load_phashes,
)
from yolo_contrastive.data.ssl_pool.manifest import ManifestRow, append_rows


# ----------------------------- image fixtures -----------------------------


def _gradient_image(size=(128, 128), direction: str = "horizontal") -> Image.Image:
    """Smooth gradient — gives a meaningfully non-uniform pHash."""
    img = Image.new("L", size)
    px = img.load()
    w, h = size
    for x in range(w):
        for y in range(h):
            v = int(255 * (x / w if direction == "horizontal" else y / h))
            px[x, y] = v
    return img.convert("RGB")


def _checker_image(size=(128, 128), cell: int = 16) -> Image.Image:
    img = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                d.rectangle([x, y, x + cell, y + cell], fill=(255, 255, 255))
    return img


def _save_jpg(img: Image.Image, path: Path, quality: int = 90) -> Path:
    img.save(path, "JPEG", quality=quality)
    return path


# ----------------------------- compute_phash ------------------------------


class TestComputePhash:
    def test_returns_16_char_hex(self, tmp_path):
        p = _save_jpg(_gradient_image(), tmp_path / "g.jpg")
        h = compute_phash(p)
        assert isinstance(h, str)
        assert len(h) == 16
        # All hex chars
        int(h, 16)  # raises if not valid hex

    def test_same_image_same_hash(self, tmp_path):
        a = _save_jpg(_gradient_image(direction="horizontal"), tmp_path / "a.jpg")
        b = _save_jpg(_gradient_image(direction="horizontal"), tmp_path / "b.jpg")
        assert compute_phash(a) == compute_phash(b)

    def test_different_patterns_differ(self, tmp_path):
        a = _save_jpg(_gradient_image(direction="horizontal"), tmp_path / "a.jpg")
        b = _save_jpg(_checker_image(), tmp_path / "b.jpg")
        ha, hb = compute_phash(a), compute_phash(b)
        # Visually very different images should have a meaningful Hamming gap
        assert ha != hb
        assert hamming_distance(ha, hb) >= 8

    def test_jpeg_quality_robust(self, tmp_path):
        """pHash should be stable across reasonable JPEG quality differences."""
        img = _gradient_image()
        p_hi = _save_jpg(img, tmp_path / "hi.jpg", quality=95)
        p_lo = _save_jpg(img, tmp_path / "lo.jpg", quality=70)
        # Either identical or very close — not testing exact equality because
        # tiny JPEG artifacts can flip a bit or two
        assert hamming_distance(compute_phash(p_hi), compute_phash(p_lo)) <= 4

    def test_accepts_bytesio(self, tmp_path):
        img = _gradient_image()
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        buf.seek(0)
        h_buf = compute_phash(buf)
        h_path = compute_phash(_save_jpg(img, tmp_path / "x.jpg"))
        assert h_buf == h_path


# --------------------------- hamming_distance -----------------------------


class TestHammingDistance:
    def test_identical(self):
        assert hamming_distance("ffff0000ffff0000", "ffff0000ffff0000") == 0

    def test_complete_inversion(self):
        # 0x0...0 vs 0xf...f = all 64 bits differ
        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_single_bit(self):
        assert hamming_distance("0000000000000000", "0000000000000001") == 1

    def test_known_value(self):
        # 0xff = 11111111 (8 bits), 0x00 = 00000000 → distance 8
        assert hamming_distance("ff00000000000000", "0000000000000000") == 8


# ---------------------- compute_pool_phashes integration ------------------


def _build_pool(tmp_path: Path, images: dict[str, Image.Image]) -> tuple[Path, Path]:
    """Build a minimal pool: writes images, returns (pool_root, manifest_path)."""
    pool = tmp_path / "pool"
    manifest = pool / "manifest.parquet"
    rows = []
    for image_id, img in images.items():
        # image_id format: <dataset>/<split>/<stem>
        materialized_rel = f"images/{image_id}.jpg"
        full = pool / materialized_rel
        full.parent.mkdir(parents=True, exist_ok=True)
        _save_jpg(img, full)
        rows.append(
            ManifestRow(
                image_id=image_id,
                dataset=image_id.split("/")[0],
                original_split="x",
                materialized_path=materialized_rel,
                original_h=img.size[1],
                original_w=img.size[0],
                materialized_h=img.size[1],
                materialized_w=img.size[0],
                image_hash="a" * 64,
                original_filename=f"{image_id.split('/')[-1]}.png",
            )
        )
    append_rows(manifest, rows)
    return pool, manifest


class TestComputePoolPhashes:
    def test_end_to_end(self, tmp_path):
        pool, manifest = _build_pool(
            tmp_path,
            {
                "bdd100k/train/a": _gradient_image(direction="horizontal"),
                "bdd100k/train/b": _gradient_image(direction="vertical"),
                "bdd100k/train/c": _checker_image(),
            },
        )
        out = pool / "phash.parquet"
        stats = compute_pool_phashes(pool, manifest, out)

        assert stats["scanned"] == 3
        assert stats["computed"] == 3
        assert stats["errors"] == 0

        df = pd.read_parquet(out)
        assert len(df) == 3
        assert set(df["image_id"]) == {
            "bdd100k/train/a",
            "bdd100k/train/b",
            "bdd100k/train/c",
        }
        # All hashes are 16-char hex
        for h in df["phash"]:
            assert len(h) == 16
            int(h, 16)

    def test_idempotent_rerun(self, tmp_path):
        pool, manifest = _build_pool(
            tmp_path,
            {
                "bdd100k/train/a": _gradient_image(),
                "bdd100k/train/b": _checker_image(),
            },
        )
        out = pool / "phash.parquet"
        first = compute_pool_phashes(pool, manifest, out)
        second = compute_pool_phashes(pool, manifest, out)

        assert first["computed"] == 2
        assert second["computed"] == 0
        assert second["skipped_existing"] == 2
        assert len(pd.read_parquet(out)) == 2

    def test_resume_after_partial(self, tmp_path):
        pool, manifest = _build_pool(
            tmp_path,
            {
                "bdd100k/train/a": _gradient_image(),
                "bdd100k/train/b": _checker_image(),
                "bdd100k/train/c": _gradient_image(direction="vertical"),
            },
        )
        out = pool / "phash.parquet"

        # First pass: limit to 1
        first = compute_pool_phashes(pool, manifest, out, limit=1)
        assert first["computed"] == 1
        assert len(pd.read_parquet(out)) == 1

        # Second pass: no limit — should compute the remaining 2
        second = compute_pool_phashes(pool, manifest, out)
        assert second["computed"] == 2
        assert second["skipped_existing"] == 1
        assert len(pd.read_parquet(out)) == 3

    def test_missing_image_does_not_abort(self, tmp_path):
        pool, manifest = _build_pool(
            tmp_path,
            {"bdd100k/train/a": _gradient_image()},
        )
        # Append a manifest row pointing to a non-existent file
        ghost = ManifestRow(
            image_id="bdd100k/train/ghost",
            dataset="bdd100k",
            original_split="train",
            materialized_path="images/bdd100k/train/ghost.jpg",  # doesn't exist
            original_h=1, original_w=1, materialized_h=1, materialized_w=1,
            image_hash="b" * 64,
            original_filename="ghost.jpg",
        )
        append_rows(manifest, [ghost])

        out = pool / "phash.parquet"
        stats = compute_pool_phashes(pool, manifest, out)

        assert stats["scanned"] == 2
        assert stats["computed"] == 1  # only the real image
        assert stats["errors"] == 1     # the ghost
        df = pd.read_parquet(out)
        assert set(df["image_id"]) == {"bdd100k/train/a"}


# -------------------------------- load ------------------------------------


class TestLoadPhashes:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_phashes(tmp_path / "nope.parquet") == {}

    def test_roundtrip(self, tmp_path):
        pool, manifest = _build_pool(
            tmp_path, {"bdd100k/train/a": _gradient_image()}
        )
        out = pool / "phash.parquet"
        compute_pool_phashes(pool, manifest, out)
        d = load_phashes(out)
        assert "bdd100k/train/a" in d
        assert len(d["bdd100k/train/a"]) == 16
