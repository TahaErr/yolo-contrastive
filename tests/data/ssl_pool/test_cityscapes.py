"""Tests for the Cityscapes adapter.

We build small fake zip archives in-process — the real 44 GB and 11 GB
packages are never touched in CI.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from yolo_contrastive.data.ssl_pool.cityscapes import (
    CANONICAL_PREFIX,
    KNOWN_SPLITS,
    _is_canonical_image,
    _parse_entry,
    count_canonical_images,
    ingest,
)
from yolo_contrastive.data.ssl_pool.manifest import (
    existing_image_ids,
    read_manifest,
)


def _png_bytes(size=(2048, 1024), color=(70, 110, 150)) -> bytes:
    """Return PNG-encoded bytes; default size mimics Cityscapes 2048×1024."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_coarse_like(zip_path: Path, *, cities=("heilbronn", "konstanz"),
                       per_city=2, include_noise=True) -> Path:
    """Fake of leftImg8bit_trainextra.zip (single split: train_extra)."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for city in cities:
            for i in range(per_city):
                stem = f"{city}_{i:06d}_000000_leftImg8bit"
                z.writestr(f"{CANONICAL_PREFIX}train_extra/{city}/{stem}.png",
                           _png_bytes(color=(i * 30, 80, 120)))
        if include_noise:
            z.writestr("README", b"readme contents")
            z.writestr("license.txt", b"license terms")
    return zip_path


def _build_fine_like(zip_path: Path) -> Path:
    """Fake of leftImg8bit_trainvaltest.zip (splits: train/val/test)."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for split, city in (("train", "jena"), ("val", "frankfurt"), ("test", "berlin")):
            stem = f"{city}_000001_000019_leftImg8bit"
            z.writestr(f"{CANONICAL_PREFIX}{split}/{city}/{stem}.png", _png_bytes())
        z.writestr("README", b"readme")
        z.writestr("license.txt", b"license")
    return zip_path


class TestIsCanonicalImage:
    def test_train_extra_accepted(self):
        assert _is_canonical_image(f"{CANONICAL_PREFIX}train_extra/heilbronn/x.png")

    def test_train_val_test_accepted(self):
        for split in ("train", "val", "test"):
            assert _is_canonical_image(f"{CANONICAL_PREFIX}{split}/jena/x.png"), split

    def test_uppercase_extension_accepted(self):
        assert _is_canonical_image(f"{CANONICAL_PREFIX}train/jena/x.PNG")

    def test_rejects_unknown_split(self):
        assert not _is_canonical_image(f"{CANONICAL_PREFIX}weird/jena/x.png")

    def test_rejects_non_png(self):
        assert not _is_canonical_image(f"{CANONICAL_PREFIX}train/jena/x.jpg")
        assert not _is_canonical_image(f"{CANONICAL_PREFIX}train/jena/x.json")

    def test_rejects_wrong_prefix(self):
        assert not _is_canonical_image("rightImg8bit/train/jena/x.png")
        assert not _is_canonical_image("README")
        assert not _is_canonical_image("license.txt")

    def test_rejects_unexpected_depth(self):
        # 2 segments after prefix — missing city
        assert not _is_canonical_image(f"{CANONICAL_PREFIX}train/x.png")
        # 4 segments after prefix — extra nesting
        assert not _is_canonical_image(f"{CANONICAL_PREFIX}train/jena/sub/x.png")


class TestParseEntry:
    def test_basic(self):
        n = f"{CANONICAL_PREFIX}train/jena/jena_000078_000019_leftImg8bit.png"
        assert _parse_entry(n) == (
            "train", "jena", "jena_000078_000019_leftImg8bit.png"
        )


class TestCountCanonicalImages:
    def test_coarse_count(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip", cities=("a", "b", "c"),
                                per_city=3)
        # 3 cities × 3 images = 9, noise (README, license) excluded
        assert count_canonical_images(zp) == 9

    def test_fine_count(self, tmp_path):
        zp = _build_fine_like(tmp_path / "fine.zip")
        assert count_canonical_images(zp) == 3  # train + val + test


class TestIngest:
    def test_coarse_end_to_end(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip",
                                cities=("heilbronn", "konstanz"), per_city=2)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)

        assert stats == {
            "scanned": 4,
            "skipped_existing": 0,
            "materialized": 4,
            "errors": 0,
        }

        df = read_manifest(manifest)
        assert len(df) == 4
        assert set(df["dataset"]) == {"cityscapes"}
        assert set(df["original_split"]) == {"train_extra"}

        # Per-city subdirs preserved on disk
        h = list((pool / "images" / "cityscapes" / "train_extra" / "heilbronn").glob("*.jpg"))
        k = list((pool / "images" / "cityscapes" / "train_extra" / "konstanz").glob("*.jpg"))
        assert len(h) == 2
        assert len(k) == 2

    def test_fine_end_to_end_three_splits(self, tmp_path):
        zp = _build_fine_like(tmp_path / "fine.zip")
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)
        assert stats["materialized"] == 3

        df = read_manifest(manifest)
        assert set(df["original_split"]) == {"train", "val", "test"}

    def test_skips_readme_and_license(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip", cities=("a",), per_city=1,
                                include_noise=True)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)
        # Just the 1 canonical image, noise filtered
        assert stats["scanned"] == 1
        assert stats["materialized"] == 1

    def test_idempotent_rerun_is_noop(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip",
                                cities=("a", "b"), per_city=2)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        first = ingest(zp, pool, manifest)
        second = ingest(zp, pool, manifest)
        assert first["materialized"] == 4
        assert second["materialized"] == 0
        assert second["skipped_existing"] == 4
        assert len(read_manifest(manifest)) == 4

    def test_coarse_and_fine_into_same_manifest(self, tmp_path):
        """A user typically ingests both packages back-to-back."""
        coarse_zp = _build_coarse_like(tmp_path / "coarse.zip",
                                       cities=("a",), per_city=2)
        fine_zp = _build_fine_like(tmp_path / "fine.zip")
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(coarse_zp, pool, manifest)
        ingest(fine_zp, pool, manifest)

        df = read_manifest(manifest)
        # 2 coarse + 3 fine = 5
        assert len(df) == 5
        assert set(df["original_split"]) == {"train_extra", "train", "val", "test"}

    def test_image_id_includes_split_and_city(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip", cities=("heilbronn",),
                                per_city=1, include_noise=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(zp, pool, manifest)
        ids = existing_image_ids(manifest)
        assert len(ids) == 1
        only = next(iter(ids))
        assert only.startswith("cityscapes/train_extra/heilbronn/")

    def test_materialized_resolution_is_640_long_side(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip", cities=("a",),
                                per_city=1, include_noise=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        ingest(zp, pool, manifest)
        jpg = next((pool / "images" / "cityscapes").rglob("*.jpg"))
        with Image.open(jpg) as img:
            # Source PNG is 2048×1024, resize to 640 long → (640, 320)
            assert img.size == (640, 320)

    def test_limit_caps_materialized(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip",
                                cities=("a",), per_city=5, include_noise=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest, limit=3)
        assert stats["materialized"] == 3

    def test_corrupt_entry_is_skipped(self, tmp_path):
        zp = tmp_path / "c.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr(f"{CANONICAL_PREFIX}train/a/good_leftImg8bit.png", _png_bytes())
            z.writestr(f"{CANONICAL_PREFIX}train/a/bad_leftImg8bit.png", b"not a png")
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)
        assert stats["materialized"] == 1
        assert stats["errors"] == 1

    def test_flush_every_does_not_lose_rows(self, tmp_path):
        zp = _build_coarse_like(tmp_path / "coarse.zip",
                                cities=("a",), per_city=7, include_noise=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest, flush_every=3)
        assert stats["materialized"] == 7
        assert len(read_manifest(manifest)) == 7
