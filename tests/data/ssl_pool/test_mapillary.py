"""Tests for the Mapillary Vistas adapter.

Vistas's defining feature for us is the heavy annotation noise: 120K
annotation PNGs sit alongside 25K RGB JPGs. We test the rejection logic
exhaustively because if any annotation leaks through it both wastes disk
and could mix label masks into the SSL pool — a subtle, hard-to-detect bug
in downstream training.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from yolo_contrastive.data.ssl_pool.mapillary import (
    IMAGE_SUBDIR,
    SPLITS,
    _is_canonical_image,
    _parse_entry,
    count_canonical_images,
    ingest,
)
from yolo_contrastive.data.ssl_pool.manifest import (
    existing_image_ids,
    read_manifest,
)


def _jpeg_bytes(size=(3000, 2000), color=(60, 90, 130)) -> bytes:
    """JPEG bytes; default size is in the Vistas range (multi-megapixel)."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _png_bytes(size=(3000, 2000)) -> bytes:
    img = Image.new("RGB", size, color=(200, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_fake_vistas(zip_path: Path, *, n_per_split=2, include_annotations=True,
                       include_configs=True) -> Path:
    """Build a fake Vistas zip with RGB jpgs + plenty of annotation noise."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for split in SPLITS:
            for i in range(n_per_split):
                stem = f"{split[:3]}_image_{i:04d}"
                z.writestr(f"{split}/{IMAGE_SUBDIR}/{stem}.jpg",
                           _jpeg_bytes(color=(i * 25, 80, 150)))
                if include_annotations and split != "testing":
                    # v1.2 family
                    for sub in ("labels", "panoptic", "instances"):
                        z.writestr(f"{split}/v1.2/{sub}/{stem}.png", _png_bytes())
                    # v2.0 family
                    for sub in ("labels", "panoptic", "instances"):
                        z.writestr(f"{split}/v2.0/{sub}/{stem}.png", _png_bytes())
        if include_configs:
            z.writestr("config_v1.2.json", b'{"version":"1.2"}')
            z.writestr("config_v2.0.json", b'{"version":"2.0"}')
    return zip_path


class TestIsCanonicalImage:
    def test_each_split_accepted(self):
        for split in SPLITS:
            assert _is_canonical_image(f"{split}/{IMAGE_SUBDIR}/x.jpg"), split

    def test_uppercase_extension_accepted(self):
        assert _is_canonical_image(f"training/{IMAGE_SUBDIR}/x.JPG")

    def test_rejects_v1_2_annotation(self):
        for sub in ("labels", "panoptic", "instances"):
            n = f"training/v1.2/{sub}/x.png"
            assert not _is_canonical_image(n), n

    def test_rejects_v2_0_annotation(self):
        for sub in ("labels", "panoptic", "instances"):
            n = f"training/v2.0/{sub}/x.png"
            assert not _is_canonical_image(n), n

    def test_rejects_png_under_images_dir(self):
        # If a stray PNG ends up under images/ it shouldn't pass — we want JPG only
        assert not _is_canonical_image(f"training/{IMAGE_SUBDIR}/x.png")

    def test_rejects_config_json(self):
        assert not _is_canonical_image("config_v1.2.json")
        assert not _is_canonical_image("config_v2.0.json")

    def test_rejects_unknown_split(self):
        assert not _is_canonical_image(f"weird/{IMAGE_SUBDIR}/x.jpg")

    def test_rejects_wrong_subdir(self):
        assert not _is_canonical_image("training/v1.2/x.jpg")
        assert not _is_canonical_image("training/somethingelse/x.jpg")

    def test_rejects_unexpected_depth(self):
        # 2 segments — missing images/
        assert not _is_canonical_image("training/x.jpg")
        # 4 segments — nested under images/
        assert not _is_canonical_image(f"training/{IMAGE_SUBDIR}/sub/x.jpg")


class TestParseEntry:
    def test_basic(self):
        n = f"training/{IMAGE_SUBDIR}/N5P8kCNrivgmUWKc7YSa3A.jpg"
        assert _parse_entry(n) == ("training", "N5P8kCNrivgmUWKc7YSa3A.jpg")


class TestCountCanonicalImages:
    def test_counts_only_rgb_images(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=3)
        # 3 splits × 3 RGB images = 9. Annotations (PNGs) and configs excluded.
        assert count_canonical_images(zp) == 9


class TestIngest:
    def test_basic_end_to_end(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=2)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)

        # 3 splits × 2 = 6 RGB, all annotations and configs filtered out
        assert stats == {
            "scanned": 6,
            "skipped_existing": 0,
            "materialized": 6,
            "errors": 0,
        }
        df = read_manifest(manifest)
        assert len(df) == 6
        assert set(df["dataset"]) == {"mapillary"}
        assert set(df["original_split"]) == {"training", "validation", "testing"}

    def test_rejects_all_annotations_in_realistic_zip(self, tmp_path):
        """Heavy annotation noise (6 PNGs per training/val image) must not leak.

        This is the killer test: in the real zip there are 120K annotation
        entries vs 25K RGB images. Any filter weakness here would multiply.
        """
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=5,
                                include_annotations=True, include_configs=True)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest)

        # 3 splits × 5 = 15 canonical images
        assert stats["scanned"] == 15
        assert stats["materialized"] == 15
        # On-disk pool contains only the 15 JPGs — no PNGs leaked
        all_files = list((pool / "images" / "mapillary").rglob("*"))
        jpgs = [f for f in all_files if f.is_file() and f.suffix.lower() == ".jpg"]
        non_jpgs = [f for f in all_files if f.is_file() and f.suffix.lower() != ".jpg"]
        assert len(jpgs) == 15
        assert non_jpgs == []

    def test_idempotent_rerun_is_noop(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=2)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        first = ingest(zp, pool, manifest)
        second = ingest(zp, pool, manifest)
        assert first["materialized"] == 6
        assert second["materialized"] == 0
        assert second["skipped_existing"] == 6
        assert len(read_manifest(manifest)) == 6

    def test_image_id_convention(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=1,
                                include_annotations=False, include_configs=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        ingest(zp, pool, manifest)

        ids = existing_image_ids(manifest)
        # <dataset>/<split>/<stem>
        assert ids == {
            "mapillary/training/tra_image_0000",
            "mapillary/validation/val_image_0000",
            "mapillary/testing/tes_image_0000",
        }

    def test_materialized_resolution_is_640_long_side(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=1,
                                include_annotations=False, include_configs=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        ingest(zp, pool, manifest)

        jpg = next((pool / "images" / "mapillary").rglob("*.jpg"))
        with Image.open(jpg) as img:
            # 3000×2000 → long-side 640 → (640, round(640*2000/3000)) = (640, 427)
            assert img.size[0] == 640
            assert img.size[1] == round(640 * 2000 / 3000)

    def test_limit_caps_materialized(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=5,
                                include_annotations=False, include_configs=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest, limit=4)
        assert stats["materialized"] == 4

    def test_corrupt_entry_is_skipped(self, tmp_path):
        zp = tmp_path / "v.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr(f"training/{IMAGE_SUBDIR}/good.jpg", _jpeg_bytes())
            z.writestr(f"training/{IMAGE_SUBDIR}/bad.jpg", b"not a real jpeg")
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"

        stats = ingest(zp, pool, manifest)
        assert stats["scanned"] == 2
        assert stats["materialized"] == 1
        assert stats["errors"] == 1

    def test_flush_every_does_not_lose_rows(self, tmp_path):
        zp = _build_fake_vistas(tmp_path / "v.zip", n_per_split=3,
                                include_annotations=False, include_configs=False)
        pool = tmp_path / "pool"
        manifest = pool / "manifest.parquet"
        stats = ingest(zp, pool, manifest, flush_every=2)
        assert stats["materialized"] == 9
        assert len(read_manifest(manifest)) == 9
